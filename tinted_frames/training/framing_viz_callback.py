"""
FramingVizCallback — TensorBoard visualization callback for framing consistency training.

Every viz_log_steps training steps, runs a mini-eval on viz_n_samples fixed samples
and logs a per-sample comparison figure to TensorBoard.

Each figure has 4 rows:
  Row 0: original image | open-ended Q + GT + model prediction
  Row 1: open_c overlay | open_c heatmap | open_c Q + GT + pred
  Row 2: yesno  overlay | yesno  heatmap | yesno  Q + GT + pred
  Row 3: mcq    overlay | mcq    heatmap | mcq    Q + GT + pred
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, BatchFeature
from tqdm import tqdm

from ..engine.attention import AttentionRollout

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class FramingVizCallback(TrainerCallback):
    """
    TrainerCallback that logs per-framing attention comparison figures to TensorBoard.

    Args:
        dataset:        FramingDataset instance (already constructed).
        processor:      AutoProcessor (for decoding token ids -> text).
        viz_log_steps:  Log figures every this many training steps.
        viz_n_samples:  Number of fixed samples to visualize.
        layer_start:    First layer index (inclusive) for attention rollout.
        layer_end:      Last layer index (exclusive) for attention rollout.
        logging_dir:    TensorBoard log directory.
    """

    def __init__(
        self,
        dataset,
        processor,
        viz_log_steps: int,
        viz_n_samples: int,
        layer_start: int,
        layer_end: int,
        logging_dir: str,
        image_size: Optional[int] = 448,
    ):
        n = min(viz_n_samples, len(dataset))
        self.viz_samples = [dataset[i] for i in range(n)]
        self.processor = processor
        self.viz_log_steps = viz_log_steps
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.logging_dir = logging_dir
        self.image_size = image_size
        self.rollout = AttentionRollout(
            residual_weight=0.5,
            start_layer=layer_start,
            end_layer=layer_end,
        )

    # ------------------------------------------------------------------
    # TrainerCallback interface
    # ------------------------------------------------------------------

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.viz_log_steps != 0:
            return
        if not state.is_world_process_zero:
            return

        writer = SummaryWriter(self.logging_dir)
        model.eval()

        with torch.no_grad():
            for i, sample in enumerate(
                tqdm(self.viz_samples, desc="Visualizing samples")
            ):
                fig = self._build_figure(model, sample)
                writer.add_figure(f"viz/sample_{i}", fig, state.global_step)
                plt.close(fig)

        writer.flush()
        writer.close()
        model.train()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model):
        if hasattr(model, "module"):
            return model.module
        return model

    def _build_figure(self, model, sample) -> plt.Figure:
        """Build comparison figure from pre-processed dataset sample."""
        device = next(model.parameters()).device
        base_model = self._unwrap(model)

        img_pil = sample["image"]
        img_array = np.array(img_pil)

        grid_hw = None
        attn_maps = {}
        framing_info = {}

        for fkey, display_key in [
            ("open", "open"),
            ("open_constrained", "open_c"),
            ("yesno", "yesno"),
            ("mcq", "mcq"),
        ]:
            fdata = sample[fkey]
            input_ids = fdata["input_ids"].unsqueeze(0).to(device)
            attention_mask = fdata["attention_mask"].unsqueeze(0).to(device)
            pixel_values = fdata["pixel_values"].to(device)

            image_grid_thw = None
            if "image_grid_thw" in fdata:
                image_grid_thw = fdata["image_grid_thw"].unsqueeze(0).to(device)

            img_start = int(fdata["img_start"])
            img_end = int(fdata["img_end"])

            if grid_hw is None:
                if image_grid_thw is not None:
                    thw = image_grid_thw[0]
                    grid_hw = (int(thw[1]) // 2, int(thw[2]) // 2)
                else:
                    n_img = img_end - img_start
                    side = int(n_img**0.5)
                    grid_hw = (side, side) if side > 0 else (1, 1)

            prompt_len = int(fdata["answer_start"])

            gen_kwargs = {
                "input_ids": input_ids[:, :prompt_len],
                "attention_mask": attention_mask[:, :prompt_len],
                "pixel_values": pixel_values,
                "max_new_tokens": 64,
                "output_attentions": True,
                "return_dict_in_generate": True,
            }
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw
            gen_output = base_model.generate(**gen_kwargs)

            pred_ids = gen_output.sequences[0, prompt_len:]
            pred_text = self.processor.decode(pred_ids, skip_special_tokens=True)
            gt_text = fdata["answer_text"]

            if display_key != "open" and gen_output.attentions:
                proc_inputs_dict = {
                    "input_ids": input_ids[:, :prompt_len],
                    "pixel_values": pixel_values,
                }
                if image_grid_thw is not None:
                    proc_inputs_dict["image_grid_thw"] = image_grid_thw
                proc_inputs = BatchFeature(proc_inputs_dict)

                attn_maps[display_key] = self.rollout.compute_rollout_simple(
                    gen_dict=gen_output,
                    inputs=proc_inputs,
                    token_indices={
                        "img_start_idx": img_start,
                        "img_end_idx": img_end,
                        "input_len": prompt_len,
                    },
                    return_mode="o-to-img",
                )

            framing_info[display_key] = {
                "question": fdata["question_text"],
                "gt": gt_text,
                "pred": pred_text,
            }

        return self._create_comparison_figure(img_array, grid_hw, attn_maps, framing_info)

    def _create_comparison_figure(self, img_array, grid_hw, attn_maps, framing_info):
        """Create a matplotlib figure comparing attention across framings."""
        fig, axes = plt.subplots(4, 3, figsize=(22, 18))

        # Row 0: original image
        axes[0, 0].imshow(img_array)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
        if "open" in framing_info:
            info = framing_info["open"]
            axes[0, 2].text(
                0.1, 0.5,
                f"Q: {info['question']}\nGT: {info['gt']}\nPred: {info['pred']}",
                transform=axes[0, 2].transAxes, fontsize=10, verticalalignment="center",
                wrap=True,
            )
        axes[0, 2].axis("off")

        # Rows 1-3: open_c, yesno, mcq
        for row, key in enumerate(["open_c", "yesno", "mcq"], start=1):
            if key in attn_maps and grid_hw is not None:
                attn = attn_maps[key]
                if isinstance(attn, torch.Tensor):
                    attn = attn.cpu().numpy()
                if attn.ndim == 1:
                    h, w = grid_hw
                    attn_2d = attn[: h * w].reshape(h, w)
                else:
                    attn_2d = attn

                # Overlay
                from PIL import Image as PILImage

                attn_resized = np.array(
                    PILImage.fromarray(
                        (attn_2d / attn_2d.max() * 255).astype(np.uint8)
                    ).resize((img_array.shape[1], img_array.shape[0]))
                )
                axes[row, 0].imshow(img_array)
                axes[row, 0].imshow(attn_resized, alpha=0.5, cmap="jet")
                axes[row, 0].set_title(f"{key} overlay")
                axes[row, 0].axis("off")

                # Heatmap
                axes[row, 1].imshow(attn_2d, cmap="jet")
                axes[row, 1].set_title(f"{key} heatmap")
                axes[row, 1].axis("off")
            else:
                axes[row, 0].axis("off")
                axes[row, 1].axis("off")

            if key in framing_info:
                info = framing_info[key]
                axes[row, 2].text(
                    0.1, 0.5,
                    f"Q: {info['question']}\nGT: {info['gt']}\nPred: {info['pred']}",
                    transform=axes[row, 2].transAxes, fontsize=10,
                    verticalalignment="center", wrap=True,
                )
            axes[row, 2].axis("off")

        plt.tight_layout()
        return fig
