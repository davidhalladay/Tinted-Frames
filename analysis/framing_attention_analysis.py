"""Framing-wise Visual Attention Analysis

Reproduces the attention metric comparison across question framings (Figure 3 / Section 4)
from the paper.

Studies how a VLM's visual attention changes across three question framings
(open-ended, yes/no, MCQ) for two datasets: GQA (reframed) and VstarBench (reframed).

For each framing x dataset combination collects five metrics per sample:
  - visual_energy
  - bbox_attention
  - sink_attention (vit + llm combined)
  - entropy
  - dispersion

Outputs:
  - results_raw.jsonl  : per-sample records
  - results_averaged.json : metrics averaged over each (dataset, framing) group
  - *_comparison.png : metric comparison plots
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

from tinted_frames.engine.inference import ModelManager, InferenceEngine
from tinted_frames.engine.attention import AttentionRollout
from tinted_frames.engine.metrics import BBoxAttentionAnalyzer, VisualEnergyCalculator
from tinted_frames.data.samplers import ReframedGQASampler, VstarFramingSampler

DATASET_COLORS = {'gqa': '#8B9FCD', 'vstar': '#4DBFB6'}
DATASET_LABELS = {'gqa': 'GQA', 'vstar': 'VstarBench'}
FRAMING_MARKERS = {'open_ended': 'o', 'yes_no': 's', 'mcq': '^'}
FRAMING_LABELS = {'open_ended': 'Open-Ended', 'yes_no': 'Yes/No', 'mcq': 'MCQ'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Simple case-insensitive substring containment check."""
    if predicted is None or ground_truth is None:
        return False
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    return gt in pred or pred in gt


def process_sample(
    sample: dict,
    dataset_label: str,
    framing_label: str,
    inference_engine,
    attention_collector,
    bbox_analyzer,
    ve_calculator,
    use_answer_ranking: bool = False,
) -> dict:
    """
    Run inference + attention analysis for a single sample.

    Returns a dict with all recorded metrics (or None values on failure).
    """
    image_path = sample['image']
    question = sample['text']
    gt_answer = sample.get('answer', None)

    # Inference
    gen_dict, inputs, output_text, token_indices = (
        inference_engine.run_inference_from_path(
            image_path=image_path,
            question=question,
            max_new_tokens=16,
        )
    )

    if use_answer_ranking and sample.get('options'):
        if sample.get('task_type', '') == 'mcq':
            my_option = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample['options'])]
        else:
            my_option = sample['options']

        _, _, ranked_text, _, *_ = inference_engine.run_inference_from_path(
            image_path=image_path,
            question=question,
            max_new_tokens=16,
            use_answer_ranking=my_option,
        )
        output_text = InferenceEngine.option_to_answer(
            ranked_text, my_option, sample.get('task_type', '')
        )

    is_correct = check_correctness(output_text, gt_answer)

    # Visual attention map [H, W]
    attention_weights = attention_collector.compute_rollout_simple(
        gen_dict, inputs, token_indices, return_mode='o-to-img'
    )

    # Stitched attention for sink detection (stacked, grid_h, grid_w)
    attn_stitched, grid_h, grid_w = attention_collector.compute_rollout_simple(
        gen_dict, inputs, token_indices, return_mode='stitched'
    )

    # Normalised attention map (H x W, sums to ~1)
    attn_norm = attention_weights / (attention_weights.sum() + 1e-8)

    # Metric 1: Visual energy
    visual_energy = ve_calculator.compute_visual_energy_simple(
        attention_weights, normalize=False
    )

    # Metric 2: BBox attention
    bbox_attention = None
    all_bboxes = sample.get('all_bboxes', None)
    if all_bboxes:
        img = Image.open(image_path)
        image_size = (img.width, img.height)
        bbox_attention = 0.0
        for bbox in all_bboxes:
            if isinstance(bbox, (list, tuple)):
                bbox = {'x': bbox[0], 'y': bbox[1], 'w': bbox[2], 'h': bbox[3]}
            bbox_attention += bbox_analyzer.compute_bbox_attention_by_grid(
                attn_map=attn_norm,
                bbox=bbox,
                image_size=image_size,
            )

    # Metric 3 & 4: Entropy and Dispersion
    entropy = ve_calculator.compute_entropy(attention_weights)
    dispersion = ve_calculator.compute_dispersion(attention_weights)

    return {
        'dataset': dataset_label,
        'framing': framing_label,
        'question_id': sample.get('question_id', ''),
        'image_path': image_path,
        'question': question,
        'answer': output_text,
        'gt_answer': gt_answer,
        'is_correct': is_correct,
        'visual_energy': visual_energy,
        'bbox_attention': float(bbox_attention) if bbox_attention is not None else None,
        'entropy': entropy,
        'dispersion': dispersion,
    }


def remove_outliers(data: list, keep_percentile: float = 0.4) -> list:
    """Keep only the middle percentile of data."""
    if len(data) < 4:
        return data
    lower_bound = (1 - keep_percentile) / 2
    upper_bound = 1 - lower_bound
    lower_val = np.percentile(data, lower_bound * 100)
    upper_val = np.percentile(data, upper_bound * 100)
    return [x for x in data if lower_val <= x <= upper_val]


def compute_averages(records: list) -> dict:
    """Average numeric metrics grouped by (dataset, framing)."""
    groups = defaultdict(list)
    for r in records:
        key = f"{r['dataset']}_{r['framing']}"
        groups[key].append(r)

    averaged = {}
    metric_keys = [
        'visual_energy', 'bbox_attention',
        'entropy', 'dispersion',
    ]
    for key, recs in groups.items():
        entry = {'n': len(recs)}
        for m in metric_keys:
            vals = [r[m] for r in recs if r[m] is not None]
            if vals:
                vals_clean = remove_outliers(vals)
                entry[f'{m}_mean'] = float(np.mean(vals_clean))
                entry[f'{m}_std'] = float(np.std(vals_clean))
                entry[f'{m}_n_clean'] = len(vals_clean)
                entry[m] = entry[f'{m}_mean']
            else:
                entry[f'{m}_mean'] = None
                entry[f'{m}_std'] = None
                entry[f'{m}_n_clean'] = 0
                entry[m] = None
        averaged[key] = entry
    return averaged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(averaged: dict, output_dir: str) -> None:
    """Generate comparison plots for all metrics across datasets and framings."""
    metrics = ['visual_energy', 'bbox_attention', 'entropy', 'dispersion']
    metric_titles = {
        'visual_energy': 'Visual Energy',
        'bbox_attention': 'BBox Attention',
        'entropy': 'Entropy',
        'dispersion': 'Dispersion',
    }

    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for dataset in ['gqa', 'vstar']:
            dataset_color = DATASET_COLORS[dataset]
            dataset_label = DATASET_LABELS[dataset]

            x_positions = []
            y_values = []
            labels = []

            for i, framing in enumerate(['open_ended', 'yes_no', 'mcq']):
                key = f"{dataset}_{framing}"
                if key not in averaged:
                    continue

                value = averaged[key].get(metric)
                if value is None:
                    continue

                x_positions.append(i)
                y_values.append(value)
                labels.append(FRAMING_LABELS[framing])

            if x_positions:
                ax.plot(x_positions, y_values,
                       color=dataset_color,
                       marker='o',
                       markersize=8,
                       linewidth=2,
                       label=dataset_label)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Open-Ended', 'Yes/No', 'MCQ'])
        ax.set_xlabel('Question Framing')
        ax.set_ylabel(metric_titles[metric])
        ax.set_title(f'{metric_titles[metric]} Across Framings')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(output_dir, f'{metric}_comparison.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Framing-wise Visual Attention Analysis")
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='HuggingFace model path or local model directory')
    parser.add_argument('--n_samples', type=int, default=100, help='Samples per framing per dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='exp/framing_attention_analysis')
    parser.add_argument('--use_answer_ranking', action='store_true', default=False,
                       help='Use answer ranking (log-likelihood scoring) instead of generation')
    # GQA dataset paths
    parser.add_argument('--gqa_questions_json', type=str, required=True,
                        help='Path to GQA questions JSON')
    parser.add_argument('--reframed_gqa_questions_json', type=str, required=True,
                        help='Path to reframed GQA questions JSON')
    parser.add_argument('--gqa_images', type=str, required=True,
                        help='Path to GQA images directory')
    parser.add_argument('--gqa_scene_graphs', type=str, required=True,
                        help='Path to GQA scene graphs JSON')
    # VstarBench dataset paths
    parser.add_argument('--vstar_framing_json', type=str, required=True,
                        help='Path to VstarBench framing JSON')
    parser.add_argument('--vstar_image_root', type=str, required=True,
                        help='Path to VstarBench images directory')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    N_SAMPLES = args.n_samples
    SEED = args.seed
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("Framing-wise Visual Attention Analysis")
    print("=" * 60)
    print(f"Model:      {MODEL_PATH}")
    print(f"N samples:  {N_SAMPLES}")
    print(f"Seed:       {SEED}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Sample data
    print("Sampling GQA ...")
    gqa_sampler = ReframedGQASampler(
        questions_path=args.gqa_questions_json,
        reframed_questions_path=args.reframed_gqa_questions_json,
        image_root=args.gqa_images,
        scene_graphs_path=args.gqa_scene_graphs,
        seed=SEED,
    )
    gqa_yes_no, gqa_mcq, gqa_open_ended = gqa_sampler.sample(
        n_samples=N_SAMPLES, single_obj_only=False
    )
    print(f"  GQA: {len(gqa_open_ended)} samples x 3 framings")

    print("Sampling VstarBench ...")
    vstar_sampler = VstarFramingSampler(
        framing_json_path=args.vstar_framing_json,
        image_root=args.vstar_image_root,
        seed=SEED,
    )
    vstar_yes_no, vstar_mcq, vstar_open_ended = vstar_sampler.sample(n_samples=N_SAMPLES)
    print(f"  VstarBench: {len(vstar_open_ended)} samples x 3 framings")

    # Initialize inference components
    print("\nLoading model ...")
    model_manager = ModelManager(MODEL_PATH)
    inference_engine = InferenceEngine(model_manager)
    attention_collector = AttentionRollout()
    bbox_analyzer = BBoxAttentionAnalyzer()
    ve_calculator = VisualEnergyCalculator()

    # Processing loop
    combos = [
        ('gqa',   'open_ended', gqa_open_ended),
        ('gqa',   'yes_no',     gqa_yes_no),
        ('gqa',   'mcq',        gqa_mcq),
        ('vstar', 'open_ended', vstar_open_ended),
        ('vstar', 'yes_no',     vstar_yes_no),
        ('vstar', 'mcq',        vstar_mcq),
    ]

    raw_path = os.path.join(OUTPUT_DIR, 'results_raw.jsonl')
    all_records = []
    accuracy_counts = defaultdict(lambda: {'total': 0, 'correct': 0})

    with open(raw_path, 'w') as f_out:
        for dataset_label, framing_label, samples in combos:
            desc = f"{dataset_label}/{framing_label}"
            print(f"\nProcessing {desc} ...")
            for sample in tqdm(samples, desc=desc):
                record = process_sample(
                    sample=sample,
                    dataset_label=dataset_label,
                    framing_label=framing_label,
                    inference_engine=inference_engine,
                    attention_collector=attention_collector,
                    bbox_analyzer=bbox_analyzer,
                    ve_calculator=ve_calculator,
                    use_answer_ranking=args.use_answer_ranking,
                )
                if record is None:
                    continue
                acc_key = f"{record['dataset']}_{record['framing']}"
                accuracy_counts[acc_key]['total'] += 1
                if record.get('is_correct', False):
                    accuracy_counts[acc_key]['correct'] += 1
                if record.get('_skip'):
                    continue
                all_records.append(record)
                f_out.write(json.dumps(record) + '\n')
                f_out.flush()

    print(f"\nWrote {len(all_records)} records to {raw_path}")

    # Averaged summary
    averaged = compute_averages(all_records)
    for key, counts in accuracy_counts.items():
        if key not in averaged:
            averaged[key] = {'n': 0}
        total = counts['total']
        correct = counts['correct']
        averaged[key]['accuracy'] = correct / total if total > 0 else 0.0
        averaged[key]['accuracy_correct'] = correct
        averaged[key]['accuracy_total'] = total
    avg_path = os.path.join(OUTPUT_DIR, 'results_averaged.json')
    with open(avg_path, 'w') as f:
        json.dump(averaged, f, indent=2)
    print(f"Wrote averaged results to {avg_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary (means +/- std, outliers removed)")
    print("=" * 60)
    for key, vals in sorted(averaged.items()):
        acc = vals.get('accuracy')
        acc_str = f"  accuracy={acc:.4f} ({vals.get('accuracy_correct',0)}/{vals.get('accuracy_total',0)})" if acc is not None else ""
        print(f"\n  [{key}]  n={vals['n']}{acc_str}")
        for m in ['visual_energy', 'bbox_attention', 'entropy', 'dispersion']:
            mean = vals.get(f'{m}_mean')
            std = vals.get(f'{m}_std')
            n_clean = vals.get(f'{m}_n_clean', 0)
            if mean is not None and std is not None:
                print(f"    {m:<25} {mean:.4f} +/- {std:.4f}  (n={n_clean})")
            else:
                print(f"    {m:<25} N/A")

    # Plots
    print("\nGenerating plots ...")
    plot_metrics(averaged, OUTPUT_DIR)

    print("\nFraming attention analysis completed.")


if __name__ == '__main__':
    main()
