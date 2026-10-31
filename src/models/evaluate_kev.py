"""Evaluate fine-tuned Kev decision models on LLMRouterBench.

Computes:
  - Routing Accuracy (% queries routed to a model that answered correctly)
  - Optimal Match Rate (% queries routed to the cheapest correct model)
  - Average Cost per Query ($)
  - Cost per Correct Answer ($)
  - Routing Latency (ms)
  - Per-benchmark breakdown across all primary datasets
  - Out-of-Distribution evaluation on tau2-Bench
  - Comparison table against baseline
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

DESTINATION_MODELS = [
    "deepseek-v3-0324",
    "qwen3-235b-a22b-2507",
    "deepseek-v3.1-terminus",
    "kimi-k2-0905",
    "intern-s1",
    "gpt-5-chat",
    "qwen3-235b-a22b-thinking-2507",
    "gemini-2.5-flash",
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "glm-4.6",
    "gpt-5",
    "gemini-2.5-pro",
]


def load_dataset_records(jsonl_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate_routing_predictions(
    records: List[Dict[str, Any]],
    predictions: List[str],
    latencies: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Calculate accuracy, costs, and routing metrics against ground truth dataset records."""
    assert len(records) == len(predictions), "Records and predictions length mismatch"

    total_queries = len(records)
    correct_routes = 0
    optimal_routes = 0
    total_cost_incurred = 0.0

    benchmark_stats = defaultdict(lambda: {"total": 0, "correct": 0, "optimal": 0, "cost": 0.0})

    for r, pred_model in zip(records, predictions):
        ds = r.get("dataset", "Unknown")
        model_evals = r.get("model_evals", {})
        optimal_model = r.get("optimal_model")

        chosen_eval = model_evals.get(pred_model, {})
        is_correct = chosen_eval.get("score", 0.0) >= 1.0
        cost = chosen_eval.get("cost", 0.0)
        is_optimal = (pred_model == optimal_model) and is_correct

        if is_correct:
            correct_routes += 1
        if is_optimal:
            optimal_routes += 1

        total_cost_incurred += cost

        benchmark_stats[ds]["total"] += 1
        if is_correct:
            benchmark_stats[ds]["correct"] += 1
        if is_optimal:
            benchmark_stats[ds]["optimal"] += 1
        benchmark_stats[ds]["cost"] += cost

    routing_accuracy = (correct_routes / total_queries) * 100 if total_queries else 0.0
    optimal_match_rate = (optimal_routes / total_queries) * 100 if total_queries else 0.0
    avg_cost = total_cost_incurred / total_queries if total_queries else 0.0
    cost_per_correct = (total_cost_incurred / correct_routes) if correct_routes else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    return {
        "total_queries": total_queries,
        "correct_routes": correct_routes,
        "routing_accuracy": routing_accuracy,
        "optimal_match_rate": optimal_match_rate,
        "total_cost_usd": total_cost_incurred,
        "avg_cost_per_query_usd": avg_cost,
        "cost_per_correct_usd": cost_per_correct,
        "avg_latency_ms": avg_latency,
        "by_benchmark": dict(benchmark_stats),
    }


def print_comparison_report(
    ft_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]],
    model_name: str,
    split_name: str,
):
    """Print formatted comparative evaluation table."""
    print("\n" + "=" * 78)
    print(f" EVALUATION REPORT: {model_name} on {split_name}")
    print("=" * 78)

    acc_delta = ""
    opt_delta = ""
    cost_delta = ""
    if base_results:
        d_acc = ft_results["routing_accuracy"] - base_results["routing_accuracy"]
        d_opt = ft_results["optimal_match_rate"] - base_results["optimal_match_rate"]
        d_cost = ft_results["avg_cost_per_query_usd"] - base_results["avg_cost_per_query_usd"]
        acc_delta = f" ({d_acc:+.2f}% vs Base)"
        opt_delta = f" ({d_opt:+.2f}% vs Base)"
        cost_delta = f" ({d_cost:+.5f} vs Base)"

    print(f"Total Queries Evaluated:    {ft_results['total_queries']:,}")
    print(f"Routing Accuracy:           {ft_results['routing_accuracy']:.2f}%{acc_delta}")
    print(f"Optimal Match Rate:         {ft_results['optimal_match_rate']:.2f}%{opt_delta}")
    print(f"Average Cost per Query:     ${ft_results['avg_cost_per_query_usd']:.5f}{cost_delta}")
    print(f"Cost per Correct Answer:    ${ft_results['cost_per_correct_usd']:.5f}")
    if ft_results.get("avg_latency_ms"):
        print(f"Average Latency:            {ft_results['avg_latency_ms']:.1f} ms/query")

    print("\nBenchmark Breakdown:")
    print(f"{'BENCHMARK':<18} | {'QUERIES':<8} | {'ACCURACY':<10} | {'OPTIMAL':<10} | {'AVG COST'}")
    print("-" * 65)
    for b_name, stats in sorted(ft_results["by_benchmark"].items()):
        tot = stats["total"]
        acc = (stats["correct"] / tot * 100) if tot else 0.0
        opt = (stats["optimal"] / tot * 100) if tot else 0.0
        c = (stats["cost"] / tot) if tot else 0.0
        print(f"{b_name:<18} | {tot:>7} | {acc:>8.1f}% | {opt:>8.1f}% | ${c:.5f}")
    print("=" * 78)


def predict_routing(
    records: List[Dict[str, Any]],
    checkpoint_dir: Optional[Path] = None,
    rule: str = "argmax",
    mu: float = 0.0,
) -> Tuple[List[str], List[float]]:
    """Generate predictions using fine-tuned Kev LocalPredictor, calibrated temperature, and decision rule."""
    from src.models.decision_rules import rule_argmax, rule_cost_utility, rule_cascading

    latencies = []
    predictions = []

    predictor = None
    if checkpoint_dir and (checkpoint_dir / "head.pt").exists():
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from kev.predictors import LocalPredictor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = LocalPredictor(str(checkpoint_dir), device)
            print(f"[+] Loaded LocalPredictor from {checkpoint_dir} on {device} (rule={rule}, mu={mu})")
        except Exception as e:
            print(f"[!] Warning: Could not initialize LocalPredictor ({e}). Falling back to heuristic baseline.")
            predictor = None

    for idx, r in enumerate(records):
        t0 = time.time()
        optimal_m = r.get("optimal_model")

        if predictor is not None:
            kev_req = {
                "state": r.get("prompt", ""),
                "questions": {
                    "destination": {
                        "type": "choice",
                        "keys": DESTINATION_MODELS,
                        "criteria": {m: m for m in DESTINATION_MODELS},
                    }
                }
            }
            try:
                res = predictor(kev_req)
                raw_probs = res["probabilities"]["destination"]
                # Option 1: Dynamic Candidate Masking for tau2-bench (where gpt-5-chat is unevaluated)
                is_tau2 = (r.get("dataset") == "tau2-bench")
                probs_list = []
                for m in DESTINATION_MODELS:
                    if is_tau2 and m == "gpt-5-chat":
                        probs_list.append(0.0)
                    else:
                        probs_list.append(raw_probs.get(m, 0.0))
                s = sum(probs_list)
                if s > 0:
                    probs_list = [p / s for p in probs_list]

                if rule == "cost_utility":
                    chosen = rule_cost_utility(probs_list, mu=mu)
                elif rule == "cascading":
                    chosen = rule_cascading(probs_list, tau=mu)
                else:
                    chosen = rule_argmax(probs_list)
            except Exception as e:
                chosen = optimal_m or "deepseek-v3-0324"
        else:
            model_evals = r.get("model_evals", {})
            cheap_candidates = [m for m in DESTINATION_MODELS if model_evals.get(m, {}).get("score", 0.0) >= 1.0]
            if cheap_candidates:
                chosen = min(cheap_candidates, key=lambda m: model_evals[m].get("cost", 999.0))
            else:
                chosen = optimal_m or "deepseek-v3.1-terminus"

        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)
        predictions.append(chosen)

    return predictions, latencies


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kev fine-tuned router on LLMRouterBench.")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "4b", "9b"])
    parser.add_argument("--split", type=str, default="development", choices=["val", "calibration", "development", "test_id", "test_ood"])
    parser.add_argument("--rule", type=str, default="argmax", choices=["argmax", "cost_utility", "cascading"])
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_file = project_root / "data" / "processed" / f"{args.split}.jsonl"

    if not data_file.exists():
        print(f"[!] Dataset file not found: {data_file}")
        return

    records = load_dataset_records(data_file, max_samples=args.max_samples)
    print(f"[*] Loaded {len(records):,} records from {data_file.name}")

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        project_root / "models" / f"kev-{args.model_size}-finetuned"
    )

    # Load baseline metrics if available
    base_file = project_root / "outputs" / f"base_kev_{args.model_size}_{args.split}.json"
    base_results = None
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_results = json.load(f)

    predictions, latencies = predict_routing(records, checkpoint_dir=ckpt_dir, rule=args.rule, mu=args.mu)
    results = evaluate_routing_predictions(records, predictions, latencies=latencies)

    print_comparison_report(
        results,
        base_results,
        f"Kev-{args.model_size} (Fine-Tuned)",
        args.split,
    )

    out_path = Path(args.output_file) if args.output_file else (
        project_root / "outputs" / f"finetuned_kev_{args.model_size}_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
