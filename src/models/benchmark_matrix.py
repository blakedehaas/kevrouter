"""Comprehensive benchmark comparison matrix runner for Kev decision router models.

Evaluates:
  - Base (Zero-Shot) vs. Fine-Tuned (LoRA + Calibrated)
  - 3 Evaluation Splits:
      * Validation (In-Distribution, 1,215 queries)
      * Test ID (Held-Out In-Distribution, 1,213 queries)
      * Test OOD (tau2-Bench Out-of-Distribution, 278 queries)
  - Decision Rules:
      * ArgMax (Confidence greedy)
      * Cost-Utility Bayes Rule (mu = 2.5)
      * Confidence Cascading (tau = 0.6)

Outputs:
  - outputs/base_kev_0.8b_{split}.json
  - outputs/finetuned_kev_0.8b_{split}.json
  - outputs/optimized_kev_0.8b_{split}.json
  - outputs/cascading_kev_0.8b_{split}.json
  - outputs/model_comparison_matrix.json
  - outputs/model_comparison_table.md
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.decision_rules import (
    DESTINATION_MODELS,
    rule_argmax,
    rule_cost_utility,
    rule_cascading,
)
from src.models.evaluate_kev import (
    load_dataset_records,
    evaluate_routing_predictions,
)

SPLITS = ["val", "test_id", "test_ood"]
EVAL_CONTEXT = {
    "max_state": 2048,
    "max_branch": 2048,
    "max_packed": 4096,
}


def compute_model_probabilities(
    records: List[Dict[str, Any]],
    kev_records: List[Dict[str, Any]],
    predictor: Any,
) -> Tuple[List[List[float]], List[float]]:
    """Compute normalized model probabilities and inference latencies via LocalPredictor."""
    all_probs = []
    latencies = []

    # Warmup pass (exclude from timing)
    if kev_records:
        _ = predictor(kev_records[0])

    t_start = time.time()
    for idx, (r, kev_r) in enumerate(zip(records, kev_records)):
        t0 = time.time()
        res = predictor(kev_r)
        elapsed_ms = (time.time() - t0) * 1000
        raw_probs = res["probabilities"]["destination"]

        # Option 1: Dynamic Candidate Masking for tau2-bench where gpt-5-chat is unevaluated
        is_tau2 = (r.get("dataset") == "tau2-bench")
        p_vec = []
        for m in DESTINATION_MODELS:
            if is_tau2 and m == "gpt-5-chat":
                p_vec.append(0.0)
            else:
                p_vec.append(raw_probs.get(m, 0.0))

        s = sum(p_vec)
        if s > 0:
            p_vec = [p / s for p in p_vec]
        else:
            p_vec = [1.0 / len(DESTINATION_MODELS)] * len(DESTINATION_MODELS)

        all_probs.append(p_vec)
        latencies.append(elapsed_ms)

        if (idx + 1) % 250 == 0 or (idx + 1) == len(records):
            print(f"    Scored {idx + 1}/{len(records)} queries ({(time.time()-t_start)/(idx+1)*1000:.1f} ms/q)...", flush=True)

    return all_probs, latencies


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format evaluation matrix as a clean GitHub-style Markdown table."""
    headers = [
        "Model",
        "Decision Rule",
        "Split",
        "Queries",
        "Accuracy (%)",
        "Optimal Match (%)",
        "Avg Cost / Query",
        "Cost / Correct Ans",
        "Latency (ms)",
    ]
    separator = ["| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]

    table_lines = [
        "| " + " | ".join(headers) + " |",
        separator[0],
    ]

    for r in rows:
        line = (
            f"| {r['model_name']} "
            f"| {r['rule']} "
            f"| {r['split']} "
            f"| {r['queries']:,} "
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

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_dir = PROJECT_ROOT / "models" / "kev-0.8b"
    ft_dir = PROJECT_ROOT / "models" / "kev-0.8b-finetuned"

    if not base_dir.exists():
        raise FileNotFoundError(f"Base Kev model not found at {base_dir}")
    if not ft_dir.exists():
        raise FileNotFoundError(f"Fine-tuned Kev model not found at {ft_dir}")

    from kev.predictors import LocalPredictor
    from kev.data import load_records

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preload split datasets
    datasets = {}
    for split in SPLITS:
        proc_file = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
        kev_file = PROJECT_ROOT / "data" / "kev_finetune" / f"{split}.jsonl"
        proc_records = load_dataset_records(proc_file)
        kev_records = load_records(str(kev_file))
        assert len(proc_records) == len(kev_records), f"Split {split} count mismatch"
        datasets[split] = (proc_records, kev_records)
        print(f"[*] Loaded {len(proc_records):,} records for {split}")

    matrix_data = []

    # =========================================================================
    # PHASE 1: BASE MODEL EVALUATION (Zero-Shot)
    # =========================================================================
    print(f"\n{'='*80}\n PHASE 1: ZERO-SHOT BASE KEV-0.8B EVALUATION\n{'='*80}")
    print(f"[*] Loading LocalPredictor for Base model from {base_dir.name} on {device}...")
    base_predictor = LocalPredictor(str(base_dir), device, context=EVAL_CONTEXT)

    for split in SPLITS:
        proc_recs, kev_recs = datasets[split]
        print(f"\n--- Base Kev-0.8B on {split} ({len(proc_recs)} queries) ---")
        base_probs, base_latencies = compute_model_probabilities(proc_recs, kev_recs, base_predictor)
        base_preds = [rule_argmax(p) for p in base_probs]
        base_results = evaluate_routing_predictions(proc_recs, base_preds, latencies=base_latencies)

        out_base_file = outputs_dir / f"base_kev_0.8b_{split}.json"
        with open(out_base_file, "w", encoding="utf-8") as f:
            json.dump(base_results, f, indent=2)

        row_base = {
            "model_name": "Kev-0.8B (Base)",
            "rule": "ArgMax (Zero-Shot)",
            "split": split,
            "queries": len(proc_recs),
            "accuracy": base_results["routing_accuracy"],
            "optimal_match": base_results["optimal_match_rate"],
            "avg_cost": base_results["avg_cost_per_query_usd"],
            "cost_per_correct": base_results["cost_per_correct_usd"],
            "latency": base_results["avg_latency_ms"],
        }
        matrix_data.append(row_base)
        print(f"    Base Result: Accuracy={row_base['accuracy']:.2f}% | "
              f"Optimal={row_base['optimal_match']:.2f}% | "
              f"Cost/Ans=${row_base['cost_per_correct']:.5f}")

    # Free base predictor memory before loading fine-tuned model
    del base_predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: FINE-TUNED MODEL EVALUATION
    # =========================================================================
    print(f"\n{'='*80}\n PHASE 2: FINE-TUNED KEV-0.8B EVALUATION (T* Calibrated)\n{'='*80}")
    print(f"[*] Loading LocalPredictor for Fine-Tuned model from {ft_dir.name} on {device}...")
    ft_predictor = LocalPredictor(str(ft_dir), device, context=EVAL_CONTEXT)

    for split in SPLITS:
        proc_recs, kev_recs = datasets[split]
        print(f"\n--- Fine-Tuned Kev-0.8B on {split} ({len(proc_recs)} queries) ---")
        ft_probs, ft_latencies = compute_model_probabilities(proc_recs, kev_recs, ft_predictor)

        # Variant A: Calibrated ArgMax
        ft_preds_argmax = [rule_argmax(p) for p in ft_probs]
        ft_results_argmax = evaluate_routing_predictions(proc_recs, ft_preds_argmax, latencies=ft_latencies)
        out_ft_file = outputs_dir / f"finetuned_kev_0.8b_{split}.json"
        with open(out_ft_file, "w", encoding="utf-8") as f:
            json.dump(ft_results_argmax, f, indent=2)

        row_ft_argmax = {
            "model_name": "Kev-0.8B (Fine-Tuned)",
            "rule": "Calibrated ArgMax",
            "split": split,
            "queries": len(proc_recs),
            "accuracy": ft_results_argmax["routing_accuracy"],
            "optimal_match": ft_results_argmax["optimal_match_rate"],
            "avg_cost": ft_results_argmax["avg_cost_per_query_usd"],
            "cost_per_correct": ft_results_argmax["cost_per_correct_usd"],
            "latency": ft_results_argmax["avg_latency_ms"],
        }
        matrix_data.append(row_ft_argmax)
        print(f"    Fine-Tuned (ArgMax): Accuracy={row_ft_argmax['accuracy']:.2f}% | "
              f"Optimal={row_ft_argmax['optimal_match']:.2f}% | "
              f"Cost/Ans=${row_ft_argmax['cost_per_correct']:.5f}")

        # Variant B: Cost-Utility Bayes Rule (mu = 2.5)
        ft_preds_cost = [rule_cost_utility(p, mu=2.5) for p in ft_probs]
        ft_results_cost = evaluate_routing_predictions(proc_recs, ft_preds_cost, latencies=ft_latencies)
        out_cost_file = outputs_dir / f"optimized_kev_0.8b_{split}.json"
        with open(out_cost_file, "w", encoding="utf-8") as f:
            json.dump(ft_results_cost, f, indent=2)

        row_ft_cost = {
            "model_name": "Kev-0.8B (Fine-Tuned)",
            "rule": "Cost-Utility (mu=2.5)",
            "split": split,
            "queries": len(proc_recs),
            "accuracy": ft_results_cost["routing_accuracy"],
            "optimal_match": ft_results_cost["optimal_match_rate"],
            "avg_cost": ft_results_cost["avg_cost_per_query_usd"],
            "cost_per_correct": ft_results_cost["cost_per_correct_usd"],
            "latency": ft_results_cost["avg_latency_ms"],
        }
        matrix_data.append(row_ft_cost)
        print(f"    Fine-Tuned (Cost-Utility): Accuracy={row_ft_cost['accuracy']:.2f}% | "
              f"Optimal={row_ft_cost['optimal_match']:.2f}% | "
              f"Cost/Ans=${row_ft_cost['cost_per_correct']:.5f}")

        # Variant C: Confidence Cascading (tau = 0.6)
        ft_preds_casc = [rule_cascading(p, tau=0.6) for p in ft_probs]
        ft_results_casc = evaluate_routing_predictions(proc_recs, ft_preds_casc, latencies=ft_latencies)
        out_casc_file = outputs_dir / f"cascading_kev_0.8b_{split}.json"
        with open(out_casc_file, "w", encoding="utf-8") as f:
            json.dump(ft_results_casc, f, indent=2)

        row_ft_casc = {
            "model_name": "Kev-0.8B (Fine-Tuned)",
            "rule": "Cascading (tau=0.6)",
            "split": split,
            "queries": len(proc_recs),
            "accuracy": ft_results_casc["routing_accuracy"],
            "optimal_match": ft_results_casc["optimal_match_rate"],
            "avg_cost": ft_results_casc["avg_cost_per_query_usd"],
            "cost_per_correct": ft_results_casc["cost_per_correct_usd"],
            "latency": ft_results_casc["avg_latency_ms"],
        }
        matrix_data.append(row_ft_casc)
        print(f"    Fine-Tuned (Cascading): Accuracy={row_ft_casc['accuracy']:.2f}% | "
              f"Optimal={row_ft_casc['optimal_match']:.2f}% | "
              f"Cost/Ans=${row_ft_casc['cost_per_correct']:.5f}")

    md_table = format_markdown_table(matrix_data)
    print("\n" + "=" * 90)
    print(" FINAL KEVROUTER BENCHMARK COMPARISON MATRIX:")
    print("=" * 90)
    print(md_table)
    print("=" * 90)

    with open(outputs_dir / "model_comparison_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix_data, f, indent=2)

    with open(outputs_dir / "model_comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# KevRouter Benchmark Comparison Matrix\n\n" + md_table + "\n")

    print(f"\n[+] Saved complete comparison matrix to {outputs_dir / 'model_comparison_matrix.json'}")
    print(f"[+] Saved markdown table to {outputs_dir / 'model_comparison_table.md'}")


main = run_matrix

if __name__ == "__main__":
    main()

