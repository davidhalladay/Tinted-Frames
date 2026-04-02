"""Generate a CSV table from experiment results for pasting into Google Sheets.

Supports three directory structures:
1. Old format: benchmark_baseline/ and benchmark_intervention/ at top level
2. New format: method/benchmark/ at top level
3. Multi-model format: model/experiment/benchmark/ at top level

Usage:
    python results_to_tsv_batch.py <experiment_folder>

Example:
    python results_to_tsv_batch.py /path/to/finetune_job2_batch_evaluation_mm/
"""

import sys
import json
from pathlib import Path


def calculate_mme_score(jsonl_files):
    """Calculate MME score (Acc + Acc+) per subcategory, then sum.

    For each subcategory:
      Acc  = (# correct) / (# total) * 100
      Acc+ = (# pairs where both correct) / (# pairs) * 100
    Total = sum of (Acc + Acc+) across all subcategories.

    Returns:
        Total score as a float, or None if no data.
    """
    from collections import defaultdict

    # subcategory -> list of (question_id, is_correct)
    subcat_data = defaultdict(list)

    try:
        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    qid = data.get("question_id", "")
                    subcat = qid.split("/")[0]
                    is_correct = data.get("is_correct", False) and data.get("answer", "") != ""
                    subcat_data[subcat].append((qid, is_correct))
    except (json.JSONDecodeError, KeyError):
        return None

    if not subcat_data:
        return None

    total_score = 0.0
    for subcat, entries in subcat_data.items():
        # Acc: individual accuracy
        n_correct = sum(1 for _, c in entries if c)
        n_total = len(entries)
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0

        # Acc+: pair accuracy (group by question_id)
        pairs = defaultdict(list)
        for qid, c in entries:
            pairs[qid].append(c)
        n_pairs = len(pairs)
        n_both_correct = sum(1 for corrections in pairs.values() if all(corrections))
        acc_plus = (n_both_correct / n_pairs) * 100 if n_pairs > 0 else 0

        total_score += acc + acc_plus

    return total_score


def calculate_accuracy_from_jsonl(jsonl_file, benchmark_name=None):
    """Calculate accuracy from results.jsonl file.

    Also handles sharded benchmarks where the directory contains shard_*/results.jsonl
    instead of a single results.jsonl.

    Args:
        jsonl_file: Path to results.jsonl
        benchmark_name: Name of the benchmark (used to detect MME)

    Returns:
        Accuracy as a float, or None if file doesn't exist or is invalid.
        For MME, returns the total score (sum of Acc + Acc+ across subcategories).
    """
    # Collect the files to process
    if not jsonl_file.exists():
        bench_dir = jsonl_file.parent
        shard_files = sorted(bench_dir.glob("shard_*/results.jsonl"))
        if not shard_files:
            return None
        files = shard_files
    else:
        files = [jsonl_file]

    # Use MME scoring if benchmark is mme
    if benchmark_name and benchmark_name.lower() == "mme":
        return calculate_mme_score(files)

    return _calculate_accuracy_from_files(files)


def _calculate_accuracy_from_files(jsonl_files):
    """Calculate accuracy from one or more results.jsonl files."""
    total = 0
    correct = 0

    try:
        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    total += 1

                    # If answer is empty string, treat as incorrect
                    if data.get("answer", "") == "":
                        print("[WARNING] Empty answer found in", jsonl_file)
                        continue

                    # Otherwise use is_correct field
                    if data.get("is_correct", False):
                        correct += 1

        if total == 0:
            return None

        return correct / total

    except (json.JSONDecodeError, KeyError):
        return None


def _format_score(score, is_mme=False):
    """Format a score for display."""
    if score is None:
        return "na"
    if is_mme:
        return f"{score:.1f}"
    return f"{score:.4f}"


def detect_structure(folder):
    """Detect which structure the folder uses."""
    # Check if any subdirectories end with _baseline or _intervention
    for d in folder.iterdir():
        if d.is_dir() and (d.name.endswith("_baseline") or d.name.endswith("_intervention")):
            return "old"

    # Check for multi-model: model/experiment/benchmark/results.jsonl (3 levels deep)
    for model_dir in folder.iterdir():
        if not model_dir.is_dir():
            continue
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            for bench_dir in exp_dir.iterdir():
                if bench_dir.is_dir() and (bench_dir / "results.jsonl").exists():
                    return "multimodel"
            break  # only check first experiment dir
        break  # only check first model dir

    # Otherwise, assume new structure (method/benchmark/)
    return "new"


def process_old_structure(folder):
    """Process the old benchmark_baseline/benchmark_intervention structure."""
    benchmark_set = set()
    for d in folder.iterdir():
        if d.is_dir() and ("_baseline" in d.name or "_intervention" in d.name):
            # Extract benchmark name (everything before _baseline or _intervention)
            if d.name.endswith("_baseline"):
                bench_name = d.name.removesuffix("_baseline")
            elif d.name.endswith("_intervention"):
                bench_name = d.name.removesuffix("_intervention")
            else:
                continue
            benchmark_set.add(bench_name)
    
    benchmarks = sorted(benchmark_set)

    names = []
    baseline_accs = []
    intervention_accs = []

    for bench in benchmarks:
        baseline_file = folder / f"{bench}_baseline" / "results.jsonl"
        intervention_file = folder / f"{bench}_intervention" / "results.jsonl"

        # Read baseline accuracy if available
        b_acc = calculate_accuracy_from_jsonl(baseline_file, benchmark_name=bench)

        # Read intervention accuracy if available
        i_acc = calculate_accuracy_from_jsonl(intervention_file, benchmark_name=bench)

        # Skip if both are missing
        if b_acc is None and i_acc is None:
            continue

        is_mme = bench.lower() == "mme"
        names.append(bench)
        baseline_accs.append(_format_score(b_acc, is_mme))
        intervention_accs.append(_format_score(i_acc, is_mme))

    return names, [baseline_accs, intervention_accs], ["Baseline", "Intervention"]


def process_new_structure(folder):
    """Process the new method/benchmark/ structure."""
    # Discover methods (top-level directories)
    methods = []
    for d in sorted(folder.iterdir()):
        if d.is_dir():
            methods.append(d.name)
    
    if not methods:
        return [], [], []
    
    # Discover benchmarks by looking at subdirectories of the first method
    benchmark_set = set()
    for method in methods:
        method_dir = folder / method
        for bench_dir in method_dir.iterdir():
            if bench_dir.is_dir():
                benchmark_set.add(bench_dir.name)
    
    benchmarks = sorted(benchmark_set)
    
    # Build accuracy matrix: rows are methods, columns are benchmarks
    method_rows = []
    for method in methods:
        accuracies = []
        for bench in benchmarks:
            results_file = folder / method / bench / "results.jsonl"
            acc = calculate_accuracy_from_jsonl(results_file, benchmark_name=bench)
            is_mme = bench.lower() == "mme"
            accuracies.append(_format_score(acc, is_mme))
        method_rows.append(accuracies)
    
    return benchmarks, method_rows, methods


def process_multimodel_structure(folder):
    """Process model/experiment/benchmark/ structure.

    Each row is labeled as 'model/experiment' and columns are benchmarks.
    """
    models = sorted(d.name for d in folder.iterdir() if d.is_dir())

    # Collect all (model, experiment) pairs and all benchmarks
    benchmark_set = set()
    row_keys = []  # list of (model, experiment)
    for model in models:
        model_dir = folder / model
        experiments = sorted(d.name for d in model_dir.iterdir() if d.is_dir())
        for exp in experiments:
            row_keys.append((model, exp))
            exp_dir = model_dir / exp
            for bench_dir in exp_dir.iterdir():
                if bench_dir.is_dir():
                    benchmark_set.add(bench_dir.name)

    benchmarks = sorted(benchmark_set)

    # Build rows
    row_labels = []
    rows = []
    for model, exp in row_keys:
        row_labels.append(f"{model}/{exp}")
        accuracies = []
        for bench in benchmarks:
            results_file = folder / model / exp / bench / "results.jsonl"
            acc = calculate_accuracy_from_jsonl(results_file, benchmark_name=bench)
            is_mme = bench.lower() == "mme"
            accuracies.append(_format_score(acc, is_mme))
        rows.append(accuracies)

    return benchmarks, rows, row_labels


def main():
    folder = Path(sys.argv[1])

    # Detect structure
    structure = detect_structure(folder)

    if structure == "old":
        names, rows, row_labels = process_old_structure(folder)
    elif structure == "multimodel":
        names, rows, row_labels = process_multimodel_structure(folder)
    else:
        names, rows, row_labels = process_new_structure(folder)

    # Build CSV lines
    lines = [",".join([""] + names)]
    for label, row in zip(row_labels, rows):
        lines.append(",".join([label] + row))


    # Print to stdout
    for line in lines:
        print(line)

    # Save as CSV under the input folder
    out_path = folder / "results.csv"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
