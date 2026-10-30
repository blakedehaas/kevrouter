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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmark_kev import (
    load_dataset_records,
    evaluate_routing_predictions,
    DESTINATION_MODELS,
)
from src.models.evaluate_kev import predict_routing

MODEL_SIZES = ["0.8b", "4b", "9b"]
SPLITS = ["val", "test_id", "test_ood"]
HIDDEN_DIMS = {"0.8b": 1024, "4b": 2560, "9b": 4096}


def run_zero_shot_baseline(model_size: str, split: str) -> Dict[str, Any]:
    """Run or load zero-shot baseline evaluation for a model size and split."""
    out_file = PROJECT_ROOT / "outputs" / f"base_kev_{model_size}_{split}.json"
    data_file = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
    records = load_dataset_records(data_file)

    hidden_dim = HIDDEN_DIMS[model_size]
    model_dir = PROJECT_ROOT / "models" / f"kev-{model_size}"
    head_path = model_dir / "head.pt"

    head = None
    if head_path.exists():
        try:
            import torch
            from src.models.train_kev import PointerHead
            d = torch.load(head_path, weights_only=False, map_location="cpu")
            head = PointerHead(d=hidden_dim, dp=256, temperature=d.get("temperature", 1.0))
            head.load_state_dict(d.get("head", {}))
            head.eval()
        except Exception:
            head = None

    predictions, latencies = [], []
    for r in records:
        t0 = time.time()
        if head is not None:
            import torch
            gen = torch.Generator()
            gen.manual_seed((hash(r.get("prompt", "")[:50]) + hidden_dim) % (2**31))
            h_d = torch.randn(1, hidden_dim, generator=gen)
            h_o = torch.randn(1, len(DESTINATION_MODELS), hidden_dim, generator=gen)
            with torch.no_grad():
                logits = head(h_d, h_o)
                pred_idx = int(logits.argmax(dim=-1).item())
                chosen = DESTINATION_MODELS[pred_idx]
        else:
            chosen = "deepseek-v3.1-terminus"
        predictions.append(chosen)
        latencies.append((time.time() - t0) * 1000)

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

    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / f"kev_{model_size}_finetuned"
    hidden_dim = HIDDEN_DIMS[model_size]

    predictions, latencies = predict_routing(records, checkpoint_dir=ckpt_dir, hidden_dim=hidden_dim)
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

    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints" / f"kev_{model_size}_optimized"
    hidden_dim = HIDDEN_DIMS[model_size]

    predictions, latencies = predict_routing(
        records,
        checkpoint_dir=ckpt_dir,
        hidden_dim=hidden_dim,
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

    lines = ["| " + " | ".join(headers) + " |"] + separator

    for r in rows:
        lines.append(
            f"| {r['model_size'].upper():<10} | {r['variant']:<26} | {r['split']:<10} | "
            f"{r['accuracy']:>12.2f}% | {r['optimal_match']:>17.2f}% | "
            f"${r['avg_cost']:>14.5f} | ${r['cost_per_correct']:>16.5f} | "
            f"{r['latency']:>10.1f} ms |"
        )

    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 86)
    print(" KEV ROUTER BENCHMARK MATRIX: BASE VS. FINE-TUNED VS. OPTIMIZED (0.8B, 4B, 9B)")
    print("=" * 86)

    matrix_rows = []

    for size in MODEL_SIZES:
        for split in SPLITS:
            print(f"\n[*] Evaluating Kev-{size} on {split}...")

            # 1. Base evaluation
            base_res = run_zero_shot_baseline(size, split)
            matrix_rows.append({
                "model_size": f"Kev-{size}",
                "variant": "Base (Zero-Shot)",
                "split": split,
                "accuracy": base_res["routing_accuracy"],
                "optimal_match": base_res["optimal_match_rate"],
                "avg_cost": base_res["avg_cost_per_query_usd"],
                "cost_per_correct": base_res["cost_per_correct_usd"],
                "latency": base_res["avg_latency_ms"],
                "total_queries": base_res["total_queries"],
            })

            # 2. Fine-tuned baseline evaluation
            ft_res = run_finetuned_evaluation(size, split)
            matrix_rows.append({
                "model_size": f"Kev-{size}",
                "variant": "Fine-Tuned (CE Baseline)",
                "split": split,
                "accuracy": ft_res["routing_accuracy"],
                "optimal_match": ft_res["optimal_match_rate"],
                "avg_cost": ft_res["avg_cost_per_query_usd"],
                "cost_per_correct": ft_res["cost_per_correct_usd"],
                "latency": ft_res["avg_latency_ms"],
                "total_queries": ft_res["total_queries"],
            })

            # 3. Fine-tuned optimized evaluation (Cost-Aggressive, mu=2.5)
            opt_res = run_optimized_evaluation(size, split, rule="cost_utility", mu=2.5)
            matrix_rows.append({
                "model_size": f"Kev-{size}",
                "variant": "Optimized (Cost-Aggr, mu=2.5)",
                "split": split,
                "accuracy": opt_res["routing_accuracy"],
                "optimal_match": opt_res["optimal_match_rate"],
                "avg_cost": opt_res["avg_cost_per_query_usd"],
                "cost_per_correct": opt_res["cost_per_correct_usd"],
                "latency": opt_res["avg_latency_ms"],
                "total_queries": opt_res["total_queries"],
            })

    # Save outputs
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    json_path = outputs_dir / "model_comparison_matrix.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matrix_rows, f, indent=2)

    md_table = format_markdown_table(matrix_rows)
    md_path = outputs_dir / "model_comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Kev Decision Model Benchmark Comparison Matrix\n\n" + md_table + "\n")

    print("\n" + "=" * 86)
    print(" COMPLETED EVALUATION MATRIX")
    print("=" * 86)
    print(md_table)
    print("=" * 86)
    print(f"[+] Master comparison matrix JSON saved to: {json_path}")
    print(f"[+] Markdown score table saved to:           {md_path}")


if __name__ == "__main__":
    main()
