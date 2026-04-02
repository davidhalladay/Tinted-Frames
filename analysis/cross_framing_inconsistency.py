"""Cross-Framing Consistency Analysis

Reproduces the cross-framing inconsistency results (Table 1 / Section 3) from the paper.

Tests whether VLMs are consistent across different question framings.
If a model truly grounds its answers visually, it should be consistent across framings.

Pipeline:
1. Sample open-ended questions from GQA or SeedBench
2. Get model's answer for open-ended question
3. If answer is correct:
   - Use an LLM to generate yes/no and MCQ versions from the answer
   - Inference model again on both framings
   - Check if answers remain correct
4. Report consistency accuracy for MCQ and yes/no framings
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from tinted_frames.engine.inference import ModelManager, InferenceEngine
from tinted_frames.data.samplers import GQASampler, SeedBenchSampler

from tinted_frames.data.samplers import MCQ_POSTFIX, YESNO_POSTFIX, OPEN_ENDED_POSTFIX

FRAMING_SYSTEM = "You are a helpful assistant that reframes questions into yes/no and MCQ formats. Always respond with valid JSON only."

def normalize_answer(text):
    """Normalize answer text for comparison."""
    if text is None:
        return ""
    return str(text).strip().lower()


def check_answer_correctness(predicted, ground_truth):
    """Check if predicted answer matches ground truth (case-insensitive contains check)."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # For yes/no questions
    if gt_norm in ["yes", "no"]:
        return gt_norm in pred_norm # pred_norm in gt_norm or gt_norm in pred_norm

    # For other answers, check if ground truth is contained in prediction
    return gt_norm in pred_norm # or pred_norm in gt_norm

JUDGE_SYSTEM = "You are a strict but fair judge. Respond with only 0 or 1."

def judge_answer_with_llm(question, ground_truth, model_output, client, model, backend):
    """Use an LLM to judge whether the model's open-ended answer is correct.

    Returns:
        bool: True if the answer is judged correct, False otherwise
    """
    prompt = (
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Model's Answer: {model_output}\n\n"
        "Is the model's answer correct? Be lenient with minor phrasing differences but strict about factual accuracy. "
        "The answer is correct if it conveys the same meaning as the ground truth.\n"
        "Output 1 if correct, 0 if incorrect."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
        )
        result_text = response.choices[0].message.content.strip()
        # Handle Qwen3 think tags
        if "vecinf" == backend and "</think>" in result_text:
            result_text = result_text.split("</think>", 1)[1].strip()
        return result_text.startswith("1")
    except Exception as e:
        print(f"Error in LLM judge: {e}")
        return check_answer_correctness(model_output, ground_truth)


def _parse_json(text, model_name=None):
    """Parse JSON from LLM response, handling Qwen3 think tags and markdown fences."""
    text = text.strip()

    # If model is Qwen3, extract content after </think> tag
    if model_name and "qwen3" in model_name.lower():
        think_end_tag = "</think>"
        if think_end_tag in text:
            parts = text.split(think_end_tag, 1)
            if len(parts) > 1:
                text = parts[1].strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def generate_reframed_questions(original_question, correct_answer, options, client, model="gpt-5-mini", backend="openai"):
    """
    Use an LLM to generate yes/no and MCQ versions of the question given the correct answer.

    Args:
        original_question: The original open-ended question
        correct_answer: The correct answer from the model
        options: List of options if this is an MCQ question, None otherwise
        client: OpenAI client instance
        model: LLM model to use
        backend: 'openai' or 'vecinf' (vLLM-compatible server)

    Returns:
        dict with 'yes_no' and 'mcq' keys, each containing question and answer info
    """

    # Randomly decide if yes/no question should affirm or negate the answer
    should_affirm = random.choice([True, False])
    if should_affirm:
        yes_no_instruction = f"Convert to a binary yes/no question where the answer is 'yes' (affirming that the answer is '{correct_answer}')."
        expected_yes_no_answer = "yes"
    else:
        yes_no_instruction = f"Convert to a binary yes/no question where the answer is 'no' by replacing the correct answer with a plausible but incorrect alternative (negating that the answer is '{correct_answer}')."
        expected_yes_no_answer = "no"

    # Build prompt based on whether we have options
    if options and len(options) > 0:
        options_text = f"\nAvailable Options: {', '.join(options)}"
        mcq_instruction = f"Create an MCQ version with the same options: {options}. The correct answer should be '{correct_answer}'."
    else:
        options_text = ""
        mcq_instruction = f"Create an MCQ version with 4 easy negative options. The correct answer should be '{correct_answer}', and provide 3 easy negative distractors."

    prompt = f"""Given the following question and its correct answer, create TWO new framings:

Original Question: {original_question}
Correct Answer: {correct_answer}{options_text}

1. Yes/No:
   - {yes_no_instruction}
   - Turn the original question into a yes/no format that tests the same knowledge. For example, if the original question is "What is the person near the garbage bin wearing?" Answer: "a coat", the yes/no question could be "Is the person near the garbage bin wearing a coat?".
   - Must be a binary question (Is/Are/Does/Do/Can/Could/etc.)

2. MCQ (Multiple Choice Question):
   - {mcq_instruction}
   - Should be same open-ended question (What/Which/Where/Who/How/etc.)
   - Provide exactly 4 options as a list

IMPORTANT RULES:
- Yes/No question must test the SAME knowledge as the original
- MCQ must test the SAME knowledge as the original
- Both should be answerable from the same visual information

Output ONLY valid JSON in this exact format:
{{
  "yes_no": {{
    "question": "Is/Are/Does/Do... question",
    "answer": "{expected_yes_no_answer}"
  }},
  "mcq": {{
    "question": "What/Which/Where... question",
    "options": ["option1", "option2", "option3", "option4"],
    "answer_text": "{correct_answer}"
  }}
}}"""

    try:
        create_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": FRAMING_SYSTEM},
                {"role": "user", "content": prompt}
            ],
        )
        # Only use structured JSON output for OpenAI API (not supported by all vLLM servers)
        if backend == "openai":
            create_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**create_kwargs)
        result_text = response.choices[0].message.content.strip()

        if backend == "vecinf":
            result = _parse_json(result_text, model_name=model)
        else:
            result = json.loads(result_text)

        if result is None:
            print(f"Error: Failed to parse JSON from response: {result_text[:200]}")
            return None

        # Ensure the yes/no answer is set as expected
        result["yes_no"]["answer"] = expected_yes_no_answer

        # Shuffle options, then determine the answer letter in the new order
        random.shuffle(result["mcq"]["options"])
        result["mcq"]["answer_letter"] = get_option_letter(
            result["mcq"]["answer_text"],
            result["mcq"]["options"]
        )

        return result
    except Exception as e:
        print(f"Error generating reframed questions: {e}")
        return None


def get_option_letter(answer_text, options):
    """Find which option letter (A, B, C, D) corresponds to the answer text."""
    answer_norm = normalize_answer(answer_text)
    for i, option in enumerate(options):
        option_norm = normalize_answer(option)
        if answer_norm in option_norm or option_norm in answer_norm:
            return chr(65+i)  # Return A, B, C, or D
    # If no match found, return A as default
    return 'A'


def format_mcq_question(question, options):
    """Format MCQ question with options A, B, C, D."""
    formatted = question + "\n"
    for i, option in enumerate(options):
        formatted += f"{chr(65+i)}. {option}\n"
    return formatted.strip()


def extract_mcq_answer(model_output):
    """Extract MCQ answer letter from model output (looks for A/B/C/D)."""
    output_norm = normalize_answer(model_output)

    # Check if output contains A, B, C, or D
    for letter in ['a', 'b', 'c', 'd']:
        if letter in output_norm:
            return letter.upper()

    # If no match found, return the first character if it's A-D
    if output_norm and output_norm[0] in 'abcd':
        return output_norm[0].upper()

    # Default to empty string if no letter found
    return ''


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Framing Consistency Analysis")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model path or local model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gqa", "seedbench"],
        default="gqa",
        help="Dataset to use (gqa or seedbench)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=6000,
        help="Number of samples to test"
    )
    # GQA dataset paths
    parser.add_argument("--gqa-questions-json", type=str, default=None,
                        help="Path to GQA questions JSON")
    parser.add_argument("--gqa-images", type=str, default=None,
                        help="Path to GQA images directory")
    parser.add_argument("--gqa-scene-graphs", type=str, default=None,
                        help="Path to GQA scene graphs JSON (optional)")
    # SeedBench dataset paths
    parser.add_argument("--seedbench-answers-json", type=str, default=None,
                        help="Path to SEED-Bench.json containing questions and answers")
    parser.add_argument("--seedbench-images", type=str, default=None,
                        help="Path to SeedBench images directory")
    # Reframing LLM options
    parser.add_argument(
        "--reframing-backend",
        type=str,
        choices=["vecinf", "openai"],
        default="openai",
        help="Backend for reframing LLM: 'vecinf' (vLLM-compatible server) or 'openai'"
    )
    parser.add_argument(
        "--reframing-base-url",
        type=str,
        default=None,
        help="Base URL for vLLM-compatible server (required if backend is vecinf)"
    )
    parser.add_argument(
        "--reframing-model",
        type=str,
        default=None,
        help="Model name for reframing LLM (e.g. Qwen3-32B for vecinf, gpt-5-mini for openai)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (OpenAI key for openai backend, or 'EMPTY' for vecinf). Falls back to OPENAI_API_KEY env var."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: exp/cross_framing_<dataset>)")

    args = parser.parse_args()

    # Configuration
    MODEL_PATH = args.model_path
    N_SAMPLES = args.n_samples
    SEED = args.seed
    DATASET = args.dataset
    OUTPUT_DIR = args.output_dir or f"exp/cross_framing_{DATASET}"

    # Validate dataset paths
    if DATASET == "gqa":
        if not all([args.gqa_questions_json, args.gqa_images]):
            parser.error("GQA dataset requires --gqa-questions-json and --gqa-images")
    else:
        if not all([args.seedbench_answers_json, args.seedbench_images]):
            parser.error("SeedBench dataset requires --seedbench-answers-json and --seedbench-images")

    print("="*60)
    print("Cross-Framing Consistency Analysis")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize reframing LLM client
    REFRAMING_BACKEND = args.reframing_backend
    if REFRAMING_BACKEND == "vecinf":
        if not args.reframing_base_url:
            print("ERROR: --reframing-base-url is required when using vecinf backend")
            return 1
        api_key = args.api_key or "EMPTY"
        openai_client = OpenAI(base_url=args.reframing_base_url, api_key=api_key)
        REFRAMING_MODEL = args.reframing_model or "Qwen3-32B"
    else:  # openai
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OpenAI API key not found!")
            print("Please set OPENAI_API_KEY environment variable or use --api-key argument")
            return 1
        openai_client = OpenAI(api_key=api_key)
        REFRAMING_MODEL = args.reframing_model or "gpt-5-mini"

    print(f"Reframing backend: {REFRAMING_BACKEND} (model: {REFRAMING_MODEL})")

    # Initialize model components
    print("Loading model...")
    model_manager = ModelManager(MODEL_PATH)
    inference_engine = InferenceEngine(model_manager)

    # Sample data
    print("\n" + "="*60)
    print("Step 1: Sampling Open-Ended Questions")
    print("="*60)

    if DATASET == "gqa":
        sampler = GQASampler(
            questions_path=args.gqa_questions_json,
            image_root=args.gqa_images,
            scene_graphs_path=args.gqa_scene_graphs,
            auto_remove_potential_binary_Q=True,
            seed=SEED
        )
        open_samples = sampler.sample_randomly(n_samples=N_SAMPLES)
    else:  # seedbench
        sampler = SeedBenchSampler(
            answers_json_path=args.seedbench_answers_json,
            image_root=args.seedbench_images,
            seed=SEED
        )
        open_samples = sampler.sample_balanced(n_samples=N_SAMPLES)

    print(f"Sampled {len(open_samples)} open-ended questions")

    # Process samples
    print("\n" + "="*60)
    print("Step 2: Testing Cross-Framing Consistency")
    print("="*60)

    results_path = f"{OUTPUT_DIR}/consistency_results.jsonl"

    # Load existing results if available
    processed_qids = set()
    stats = {
        'total_samples': 0,
        'open_ended_correct': 0,
        'yes_no_consistent': 0,
        'mcq_consistent': 0,
        'both_consistent': 0,
        'gpt_generation_failed': 0
    }

    if Path(results_path).exists():
        print(f"Found existing results file: {results_path}")
        with open(results_path, 'r') as f:
            for line in f:
                result = json.loads(line.strip())
                processed_qids.add(result['question_id'])

                # Rebuild statistics from existing results
                stats['total_samples'] += 1
                if result.get('open_ended_correct', False):
                    stats['open_ended_correct'] += 1
                    if result.get('yes_no_consistent', False):
                        stats['yes_no_consistent'] += 1
                    if result.get('mcq_consistent', False):
                        stats['mcq_consistent'] += 1
                    if result.get('both_consistent', False):
                        stats['both_consistent'] += 1
                if result.get('gpt_generation_failed', False):
                    stats['gpt_generation_failed'] += 1

        print(f"Loaded {len(processed_qids)} existing results")
        print(f"Resuming from: {stats['total_samples']} samples, "
              f"{stats['open_ended_correct']} correct, "
              f"{stats['both_consistent']} both consistent")

    def _postfix(stats):
        n_oe = stats['open_ended_correct']
        n_tot = stats['total_samples']
        oe_acc = f"{n_oe}/{n_tot}"
        if n_oe > 0:
            yn_inc  = f"{100*(n_oe - stats['yes_no_consistent']) / n_oe:.1f}%"
            mcq_inc = f"{100*(n_oe - stats['mcq_consistent'])    / n_oe:.1f}%"
            both_inc= f"{100*(n_oe - stats['both_consistent'])    / n_oe:.1f}%"
        else:
            yn_inc = mcq_inc = both_inc = "n/a"
        return {"OE_corr": oe_acc, "YN_inc": yn_inc, "MCQ_inc": mcq_inc, "Both_inc": both_inc}

    def _print_table(qid, base_q, gt, oe_out,
                     reframed, yn_out, yn_correct,
                     mcq_out, mcq_letter, mcq_correct):
        W = 64
        mcq_gt = f"{reframed['mcq']['answer_letter']} ({reframed['mcq']['answer_text']})"
        lines = [
            f"── [converted] qid={qid} " + "─" * max(0, W - 20 - len(str(qid))),
            f"OE   Q : {base_q}",
            f"     GT: {gt:<18}  Out: {oe_out}",
            f"Y/N  Q : {reframed['yes_no']['question']}",
            f"     GT: {reframed['yes_no']['answer']:<18}  Out: {yn_out}  {'✓' if yn_correct else '✗'}",
            f"MCQ  Q : {reframed['mcq']['question']}",
            "     " + "  ".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(reframed['mcq']['options'])),
            f"     GT: {mcq_gt:<18}  Out: {mcq_out}  {'✓' if mcq_correct else '✗'}",
            "─" * W,
        ]
        return "\n".join(lines)

    # Open results file for appending
    with open(results_path, 'a') as f_out:
        with tqdm(open_samples, desc="Processing samples") as pbar:
            for idx, sample in enumerate(pbar, 1):
                question_id = sample['question_id']

                # Skip if already processed
                if question_id in processed_qids:
                    continue

                stats['total_samples'] += 1

                # Step 1: Get answer for open-ended question
                # GQA has 'text_nopostfix', SeedBench has 'question' (raw question text)
                base_q = sample.get('question') or sample.get('text_nopostfix')
                open_ended_question = base_q + OPEN_ENDED_POSTFIX
                _, _, output_text_open, _ = inference_engine.run_inference_from_path(
                    image_path=sample['image'],
                    question=open_ended_question,
                    max_new_tokens=32,
                )

                # Check if open-ended answer is correct
                ground_truth = sample.get('answer_text', sample['answer'])

                is_open_correct = check_answer_correctness(output_text_open, ground_truth)

                result = {
                    'question_id': question_id,
                    'image': sample['image'],
                    'original_question': base_q,
                    'ground_truth': ground_truth,
                    'open_ended_output': output_text_open,
                    'open_ended_correct': is_open_correct,
                    'yes_no_consistent': False,
                    'mcq_consistent': False,
                    'both_consistent': False,
                    'gpt_generation_failed': False,
                    'category': sample.get('category', 'unknown')
                }

                # Only proceed if open-ended answer is correct
                if is_open_correct:
                    stats['open_ended_correct'] += 1

                    # Step 2: Generate reframed questions using LLM
                    options = sample.get('options', None)
                    reframed = generate_reframed_questions(
                        original_question=base_q,
                        correct_answer=sample.get('answer_text', sample['answer']),
                        options=options,
                        client=openai_client,
                        model=REFRAMING_MODEL,
                        backend=REFRAMING_BACKEND
                    )

                    if reframed is None:
                        stats['gpt_generation_failed'] += 1
                        result['gpt_generation_failed'] = True
                        f_out.write(json.dumps(result) + '\n')
                        f_out.flush()
                        pbar.set_postfix(_postfix(stats))
                        continue

                    # Store reframed questions
                    result['yes_no_question'] = reframed['yes_no']['question']
                    result['yes_no_expected_answer'] = reframed['yes_no']['answer']
                    result['mcq_question'] = reframed['mcq']['question']
                    result['mcq_options'] = reframed['mcq']['options']
                    result['mcq_expected_answer'] = reframed['mcq']['answer_letter']
                    result['mcq_expected_answer_text'] = reframed['mcq']['answer_text']

                    # Step 3a: Test yes/no consistency
                    _, _, output_text_yn, _ = inference_engine.run_inference_from_path(
                        image_path=sample['image'],
                        question=reframed['yes_no']['question'] + YESNO_POSTFIX,
                        max_new_tokens=64,
                    )
                    result['yes_no_output'] = output_text_yn

                    yn_correct = check_answer_correctness(
                        output_text_yn,
                        reframed['yes_no']['answer']
                    )
                    result['yes_no_consistent'] = yn_correct
                    if yn_correct:
                        stats['yes_no_consistent'] += 1

                    # Step 3b: Test MCQ consistency
                    mcq_formatted = format_mcq_question(
                        reframed['mcq']['question'],
                        reframed['mcq']['options']
                    )

                    _, _, output_text_mcq, _ = inference_engine.run_inference_from_path(
                        image_path=sample['image'],
                        question=mcq_formatted + MCQ_POSTFIX,
                        max_new_tokens=16,
                    )
                    result['mcq_output'] = output_text_mcq

                    mcq_answer_letter = extract_mcq_answer(output_text_mcq)
                    result['mcq_predicted_letter'] = mcq_answer_letter

                    mcq_correct = (mcq_answer_letter == reframed['mcq']['answer_letter'])
                    result['mcq_consistent'] = mcq_correct
                    if mcq_correct:
                        stats['mcq_consistent'] += 1

                    if result['yes_no_consistent'] and mcq_correct:
                        stats['both_consistent'] += 1
                        result['both_consistent'] = True

                    tqdm.write(_print_table(
                        question_id, base_q, ground_truth, output_text_open,
                        reframed, output_text_yn, yn_correct,
                        output_text_mcq, mcq_answer_letter, mcq_correct,
                    ))

                else:
                    tqdm.write(f"[✗ OE]  gt={ground_truth!r}  out={output_text_open!r}")

                # Save result
                f_out.write(json.dumps(result) + '\n')
                f_out.flush()

                pbar.set_postfix(_postfix(stats))

    # Compute final statistics
    print("\n" + "="*60)
    print("Step 3: Computing Final Statistics")
    print("="*60)

    final_stats = {
        'dataset': DATASET,
        'model': MODEL_PATH,
        'n_samples': N_SAMPLES,
        'total_processed': stats['total_samples'],
        'open_ended_accuracy': stats['open_ended_correct'] / max(stats['total_samples'], 1),
        'gpt_generation_failures': stats['gpt_generation_failed']
    }

    if stats['open_ended_correct'] > 0:
        final_stats['yes_no_consistency'] = stats['yes_no_consistent'] / stats['open_ended_correct']
        final_stats['mcq_consistency'] = stats['mcq_consistent'] / stats['open_ended_correct']
        final_stats['both_consistency'] = stats['both_consistent'] / stats['open_ended_correct']
        final_stats['yes_no_consistent_count'] = stats['yes_no_consistent']
        final_stats['mcq_consistent_count'] = stats['mcq_consistent']
        final_stats['both_consistent_count'] = stats['both_consistent']
        final_stats['open_ended_correct_count'] = stats['open_ended_correct']
    else:
        final_stats['yes_no_consistency'] = 0.0
        final_stats['mcq_consistency'] = 0.0
        final_stats['both_consistency'] = 0.0

    # Compute category-wise statistics for SeedBench
    category_stats = {}
    if DATASET == 'seedbench':
        all_results = []
        with open(results_path, 'r') as f:
            for line in f:
                all_results.append(json.loads(line.strip()))

        for result in all_results:
            category = result.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {
                    'total': 0,
                    'open_ended_correct': 0,
                    'yes_no_consistent': 0,
                    'mcq_consistent': 0,
                    'both_consistent': 0
                }

            category_stats[category]['total'] += 1
            if result['open_ended_correct']:
                category_stats[category]['open_ended_correct'] += 1
                if result['yes_no_consistent']:
                    category_stats[category]['yes_no_consistent'] += 1
                if result['mcq_consistent']:
                    category_stats[category]['mcq_consistent'] += 1
                if result['both_consistent']:
                    category_stats[category]['both_consistent'] += 1

        for category in category_stats:
            cat = category_stats[category]
            cat['open_ended_accuracy'] = cat['open_ended_correct'] / max(cat['total'], 1)
            if cat['open_ended_correct'] > 0:
                cat['yes_no_consistency'] = cat['yes_no_consistent'] / cat['open_ended_correct']
                cat['mcq_consistency'] = cat['mcq_consistent'] / cat['open_ended_correct']
                cat['both_consistency'] = cat['both_consistent'] / cat['open_ended_correct']
            else:
                cat['yes_no_consistency'] = 0.0
                cat['mcq_consistency'] = 0.0
                cat['both_consistency'] = 0.0

        final_stats['category_breakdown'] = category_stats

    # Save statistics
    stats_path = f"{OUTPUT_DIR}/consistency_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)

    print(f"Saved statistics to: {stats_path}")

    # Print summary
    print("\n" + "="*60)
    print("Cross-Framing Consistency Analysis Complete")
    print("="*60)
    print(f"\nResults:")
    print(f"  - Total samples processed: {stats['total_samples']}")
    print(f"  - Open-ended correct: {stats['open_ended_correct']} ({100*final_stats['open_ended_accuracy']:.1f}%)")
    print(f"  - LLM generation failures: {stats['gpt_generation_failed']}")

    if stats['open_ended_correct'] > 0:
        print(f"\nConsistency Analysis (among {stats['open_ended_correct']} correctly answered open-ended questions):")
        print(f"  - Yes/No consistency: {stats['yes_no_consistent']}/{stats['open_ended_correct']} ({100*final_stats['yes_no_consistency']:.1f}%)")
        print(f"  - MCQ consistency: {stats['mcq_consistent']}/{stats['open_ended_correct']} ({100*final_stats['mcq_consistency']:.1f}%)")
        print(f"  - Both consistent: {stats['both_consistent']}/{stats['open_ended_correct']} ({100*final_stats['both_consistency']:.1f}%)")

    if DATASET == 'seedbench' and category_stats:
        print(f"\nCategory-wise Performance:")
        print(f"{'Category':<25} {'Total':<8} {'Open-Acc':<10} {'YN-Cons':<10} {'MCQ-Cons':<10} {'Both-Cons':<10}")
        print("-" * 80)
        for category in sorted(category_stats.keys()):
            cat = category_stats[category]
            print(f"{category:<25} {cat['total']:<8} "
                  f"{cat['open_ended_accuracy']*100:<10.1f} "
                  f"{cat['yes_no_consistency']*100:<10.1f} "
                  f"{cat['mcq_consistency']*100:<10.1f} "
                  f"{cat['both_consistency']*100:<10.1f}")

    print(f"\nOutput files:")
    print(f"  - Results: {results_path}")
    print(f"  - Statistics: {stats_path}")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
