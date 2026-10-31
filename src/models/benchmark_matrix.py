"""Comprehensive benchmark comparison matrix runner for Kev decision router models.

Evaluates:
  - 3 Model Sizes: 0.8B, 4B, 9B
  - 2 Variants: Base (Zero-Shot) vs. Fine-Tuned (LoRA + Calibrated)
  - 3 Evaluation Splits:
      * Validation (In-Distribution, 1,215 queries)
      * Test ID (Held-Out In-Distribution, 1,213 queries)
      * Test OOD (tau2-Bench Out-of-Distribution, 278 queries)

Outputs:
  - outputs/model_comparison_matrix.json
  - outputs/model_comparison_table.md
  - Formatted console table
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate_kev import (
    load_dataset_records,
    evaluate_routing_predictions,
    predict_routing,
    DESTINATION_MODELS,
)

MODEL_SIZES = ["0.8b", "4b", "9b"]
SPLITS = ["val", "test_id", "test_ood"]


def run_zero_shot_baseline(model_size: str, split: str) -> Dict[str, Any]:
    """Run or load zero-shot baseline evaluation for a model size and split."""
    out_file = PROJECT_ROOT / "outputs" / f"base_kev_{model_size}_{split}.json"
    data_file = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
    records = load_dataset_records(data_file)

    base_dir = PROJECT_ROOT / "models" / f"kev-{model_size}"
    ckpt_dir = base_dir if base_dir.exists() else None

    predictions, latencies = predict_routing(records, checkpoint_dir=ckpt_dir, rule="argmax")
    results = evaluate_routing_predictions(records, predictions, latencies=latencies)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def run_finetuned_evaluation(model_size: str, split: str) -> Dict[str, Any]:
    """Run fine-tuned model evaluation for a model size and split."""
    out_file = PROJECT_ROOT / "outputs" / f"finetuned_kev_{model_size}_{split}.json"
    data_file = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
    records = load_dataset_records(data_file)

    ckpt_dir = PROJECT_ROOT / "models" / f"kev-{model_size}-finetuned"
    if not ckpt_dir.exists():
        ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / f"kev_{model_size}_finetuned"

    predictions, latencies = predict_routing(records, checkpoint_dir=ckpt_dir, rule="argmax")
    results = evaluate_routing_predictions(records, predictions, latencies=latencies)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def run_optimized_evaluation(model_size: str, split: str, rule: str = "cost_utility", mu: float = 2.5) -> Dict[str, Any]:
    """Run fine-tuned model evaluation with Cost-Aggressive (mu=2.5) strategy."""
    out_file = PROJECT_ROOT / "outputs" / f"optimized_kev_{model_size}_{split}.json"
    data_file = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
    records = load_dataset_records(data_file)

    ckpt_dir = PROJECT_ROOT / "models" / f"kev-{model_size}-finetuned"
    if not ckpt_dir.exists():
        ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / f"kev_{model_size}_optimized"

    predictions, latencies = predict_routing(
        records,
        checkpoint_dir=ckpt_dir,
        rule=rule,
        mu=mu,
    )
    results = evaluate_routing_predictions(records, predictions, latencies=latencies)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format evaluation matrix as a clean GitHub-style Markdown table."""
    headers = [
        "Model Size",
        "Variant",
        "Split",
        "Accuracy (%)",
        "Optimal Match (%)",
        "Avg Cost / Query",
        "Cost / Correct Ans",
        "Latency (ms)",
    ]
    separator = ["| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"]

    table_lines = [
        "| " + " | ".join(headers) + " |",
        separator[0],
    ]

    for r in rows:
        line = (
            f"| {r['model_size'].upper()} "
            f"| {r['variant']} "
            f"| {r['split']} "
            f"| {r['accuracy']:.2f}% "
            f"| {r['optimal_match']:.2f}% "
            f"| ${r['avg_cost']:.5f} "
            f"| ${r['cost_per_correct']:.5f} "
            f"| {r['latency']:.1f} |"
        )
        table_lines.append(line)

    return "\n".join(table_lines)


def run_matrix():
    print("=" * 80)
    print(" KEV ROUTER COMPREHENSIVE BENCHMARK MATRIX RUNNER")
    print("=" * 80)

    matrix_data = []
    summary_rows = []

    # Run for 0.8b (and 4b/9b if available)
    available_sizes = ["0.8b"]

    for size in available_sizes:
        for split in SPLITS:
            # 1. Base Zero-Shot
            print(f"[*] Running Base (Zero-Shot) Kev-{size} on {split}...")
            base_res = run_zero_shot_baseline(size, split)
            row_base = {
                "model_size": size,
                "variant": "Zero-Shot Base",
                "split": split,
                "accuracy": base_res["routing_accuracy"],
                "optimal_match": base_res["optimal_match_rate"],
                "avg_cost": base_res["avg_cost_per_query_usd"],
                "cost_per_correct": base_res["cost_per_correct_usd"],
                "latency": base_res["avg_latency_ms"],
            }
            matrix_data.append(row_base)
            summary_rows.append(row_base)

            # 2. Fine-Tuned (Calibrated Argmax)
            print(f"[*] Running Fine-Tuned Kev-{size} on {split}...")
            ft_res = run_finetuned_evaluation(size, split)
            row_ft = {
                "model_size": size,
                "variant": "Fine-Tuned (Calibrated Argmax)",
                "split": split,
                "accuracy": ft_res["routing_accuracy"],
                "optimal_match": ft_res["optimal_match_rate"],
                "avg_cost": ft_res["avg_cost_per_query_usd"],
                "cost_per_correct": ft_res["cost_per_correct_usd"],
                "latency": ft_res["avg_latency_ms"],
            }
            matrix_data.append(row_ft)
            summary_rows.append(row_ft)

            # 3. Fine-Tuned (Cost-Utility Optimized mu=2.5)
            print(f"[*] Running Cost-Utility Optimized Kev-{size} on {split}...")
            opt_res = run_optimized_evaluation(size, split, rule="cost_utility", mu=2.5)
            row_opt = {
                "model_size": size,
                "variant": "Fine-Tuned (Cost-Utility mu=2.5)",
                "split": split,
                "accuracy": opt_res["routing_accuracy"],
                "optimal_match": opt_res["optimal_match_rate"],
                "avg_cost": opt_res["avg_cost_per_query_usd"],
                "cost_per_correct": opt_res["cost_per_correct_usd"],
                "latency": opt_res["avg_latency_ms"],
            }
            matrix_data.append(row_opt)
            summary_rows.append(row_opt)

    md_table = format_markdown_table(summary_rows)
    print("\n" + "=" * 80)
    print(" BENCHMARK COMPARISON MATRIX:")
    print("=" * 80)
    print(md_table)
    print("=" * 80)

    # Save outputs
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(outputs_dir / "model_comparison_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix_data, f, indent=2)

    with open(outputs_dir / "model_comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# KevRouter Benchmark Comparison Matrix\n\n" + md_table + "\n")

    print(f"\n[+] Saved comparison matrix to {outputs_dir / 'model_comparison_matrix.json'}")
    print(f"[+] Saved markdown table to {outputs_dir / 'model_comparison_table.md'}")


if __name__ == "__main__":
    run_matrix()
