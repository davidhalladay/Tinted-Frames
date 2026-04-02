"""Evaluate Finetuned Models

Reproduces the mitigation evaluation results (Table 2 / Section 5) from the paper.

Evaluates finetuned models (prompt-tuning prefix/postfix or LoRA) on all supported
datasets, collecting accuracy metrics.

Supported datasets: MME, POPE, V*Bench, HRBench (4K/8K),
HallusionBench, RealWorldQA, MMMU-Pro.
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from tinted_frames.engine.inference import ModelManager, InferenceEngine
from tinted_frames.data.samplers import (
    MMESampler,
    POPESampler,
    VstarSampler,
    HRBenchSampler,
    HallusionBenchSampler,
    RealWorldQASampler,
    MMMUProSampler,
)

# Dataset-to-framing_idx mapping
DATASET_FRAMING_IDX = {
    'realworldqa': 2,
    'mme': 1,
    'mmmupro': 2,
    'hallusion': 1,
    'pope': 1,
    'hrbench4k': 2,
    'hrbench8k': 2,
    'vstar': 2,
}


def is_correct_answer(pred_answer, true_answer):
    pred_answer_norm = str(pred_answer).strip().lower()
    true_answer_norm = str(true_answer).strip().lower()
    return true_answer_norm in pred_answer_norm


def evaluate_model(args, inference_engine, samples, dataset_name,
                   framing_idx=-1, use_answer_ranking=False):
    """Evaluate finetuned model on a dataset."""
    results = []
    correct_count = 0

    for idx, sample in enumerate(tqdm(samples, desc=f"{dataset_name}"), 1):
        if use_answer_ranking and sample.get('options') and sample.get('task_type', '') in ('mcq'):
            if sample.get('task_type', '') == 'mcq':
                my_option = [f"{chr(65+i)}" for i, opt in enumerate(sample['options'])]
            else:
                my_option = sample['options']

            _, _, ranked_text, _, *_ = inference_engine.run_inference_from_path(
                image_path=sample['image'],
                question=sample.get('text', ''),
                max_new_tokens=16,
                use_answer_ranking=my_option,
                framing_idx=framing_idx,
            )
            output_text = InferenceEngine.option_to_answer(
                ranked_text, my_option, sample.get('task_type', '')
            )
        else:
            _, _, output_text, _, *_ = inference_engine.run_inference_from_path(
                image_path=sample['image'],
                question=sample['text'],
                max_new_tokens=16,
                framing_idx=framing_idx,
            )

        is_correct = is_correct_answer(output_text, sample['answer'])
        if is_correct:
            correct_count += 1

        print(output_text, sample['answer'])

        result = {
            'task_type': dataset_name,
            'dataset': sample.get('dataset', dataset_name),
            'question_id': sample.get('question_id', f"{dataset_name}_{idx}"),
            'question': sample['text'],
            'answer': output_text,
            'gt_answer': sample['answer'],
            'is_correct': is_correct,
            'image_path': sample['image'],
        }
        results.append(result)

    accuracy = correct_count / len(samples) if len(samples) > 0 else 0

    return {
        'results': results,
        'accuracy': accuracy,
        'correct': correct_count,
        'total': len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Finetuned Models')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='HuggingFace model path or local model directory')
    parser.add_argument('--dataset', type=str, default='mme',
                       choices=['realworldqa', 'mme', 'mmmupro', 'hallusion',
                                'pope', 'hrbench4k', 'hrbench8k', 'vstar'],
                       help='Dataset to use for evaluation')
    parser.add_argument('--finetune_mode', type=str, default='none',
                       help='Finetune mode: none, lora, ptuning_8_prefix, ptuning_8_postfix, etc.')
    parser.add_argument('--aug_model_path', type=str, default=None,
                       help='Path to finetuned checkpoint (ptuning or LoRA adapter)')
    parser.add_argument('--n_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='exp/evaluate_finetuned',
                       help='Output directory for results')
    parser.add_argument('--is_framing_shared', action='store_true', default=False,
                       help='If True, use framing_idx=1 for all datasets')
    parser.add_argument('--use_answer_ranking', action='store_true', default=False,
                       help='Use answer ranking (log-likelihood scoring) instead of generation')

    args = parser.parse_args()

    MODEL_PATH = args.model_path

    # Hardcoded dataset paths (relative to project root)
    DATA_ROOT = 'data'
    MME_JSONL        = f'{DATA_ROOT}/MME/llava_mme_original.jsonl'
    MME_IMAGES       = f'{DATA_ROOT}/MME/MME_Benchmark_release_version'
    MME_ANSWERS      = f'{DATA_ROOT}/MME/MME_Benchmark_release_version'
    POPE_ROOT        = f'{DATA_ROOT}/pope/coco'
    POPE_IMAGES      = f'{DATA_ROOT}/pope/val2014'
    VSTAR_ROOT       = f'{DATA_ROOT}/vstar_bench'
    HRBENCH_ROOT     = f'{DATA_ROOT}/hrbench'
    HALLUSION_ROOT   = f'{DATA_ROOT}/HallusionBench'
    REALWORLDQA_ROOT = f'{DATA_ROOT}/realworldqa'
    MMMUPRO_ROOT     = f'{DATA_ROOT}/mmmu_pro'

    # Look up framing_idx from dataset
    if args.is_framing_shared:
        framing_idx = 1
    else:
        framing_idx = DATASET_FRAMING_IDX.get(args.dataset, -1)

    print("="*60)
    print("Evaluate Finetuned Models")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Finetune Mode: {args.finetune_mode}")
    print(f"Aug Model Path: {args.aug_model_path}")
    print(f"Framing Idx: {framing_idx}")
    print(f"Answer Ranking: {args.use_answer_ranking}")
    print(f"Samples: {args.n_samples}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("Loading model...")
    model_manager = ModelManager(
        MODEL_PATH,
        aug_model_path=args.aug_model_path,
        load_finetune_model_mode=args.finetune_mode
    )
    inference_engine = InferenceEngine(model_manager)

    # Sample data
    print("\n" + "="*60)
    print("Step 1: Sampling Data")
    print("="*60)

    if args.dataset == 'realworldqa':
        sampler = RealWorldQASampler(
            realworldqa_root=REALWORLDQA_ROOT,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = 'realworldqa'

    elif args.dataset == 'mme':
        sampler = MMESampler(
            jsonl_path=MME_JSONL,
            image_root=MME_IMAGES,
            answer_root=MME_ANSWERS,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = 'mme'

    elif args.dataset == 'mmmupro':
        sampler = MMMUProSampler(
            mmmupro_root=MMMUPRO_ROOT,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = 'mmmupro'

    elif args.dataset == 'hallusion':
        sampler = HallusionBenchSampler(
            hallusion_root=HALLUSION_ROOT,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = 'hallusion'

    elif args.dataset == 'pope':
        sampler = POPESampler(
            pope_root=POPE_ROOT,
            image_root=POPE_IMAGES,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = 'pope'

    elif args.dataset in ('hrbench4k', 'hrbench8k'):
        resolution = '4k' if args.dataset == 'hrbench4k' else '8k'
        sampler = HRBenchSampler(
            hrbench_root=HRBENCH_ROOT,
            resolution=resolution,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = f'hrbench_{resolution}'

    elif args.dataset == 'vstar':
        sampler = VstarSampler(
            vstar_root=VSTAR_ROOT,
            seed=args.seed
        )
        samples = sampler.sample(n_samples=args.n_samples)
        dataset_name = 'vstar'

    print(f"{args.dataset.upper()} samples: {len(samples)} samples")

    task_type_counts = {}
    for sample in samples:
        task_type = sample['task_type']
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    for task_type, count in task_type_counts.items():
        print(f"  {task_type}: {count} samples")

    # Run evaluation
    print("\n" + "="*60)
    print("Step 2: Running Inference")
    print("="*60)

    print(f"\nProcessing {dataset_name}...")

    eval_results = evaluate_model(
        args=args,
        inference_engine=inference_engine,
        samples=samples,
        dataset_name=dataset_name,
        framing_idx=framing_idx,
        use_answer_ranking=args.use_answer_ranking,
    )

    print(f"  {dataset_name} Accuracy: {eval_results['accuracy']:.2%} "
          f"({eval_results['correct']}/{eval_results['total']})")

    # Save results
    print("\n" + "="*60)
    print("Step 3: Saving Results")
    print("="*60)

    results_path = f"{args.output_dir}/results.jsonl"
    with open(results_path, 'w') as f:
        for result in eval_results['results']:
            f.write(json.dumps(result) + '\n')

    print(f"Saved detailed results to: {results_path}")
    print(f"\nOverall Accuracy: {eval_results['accuracy']:.2%} "
          f"({eval_results['correct']}/{eval_results['total']})")

    # Save summary
    summary = {
        'model': MODEL_PATH,
        'finetune_mode': args.finetune_mode,
        'aug_model_path': args.aug_model_path,
        'framing_idx': framing_idx,
        'dataset': args.dataset,
        'parameters': {
            'n_samples': args.n_samples,
            'seed': args.seed,
            'use_answer_ranking': args.use_answer_ranking,
        },
        'overall_accuracy': eval_results['accuracy'],
        'overall_correct': eval_results['correct'],
        'overall_total': eval_results['total'],
    }

    summary_path = f"{args.output_dir}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to: {summary_path}")

    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)


if __name__ == "__main__":
    main()
