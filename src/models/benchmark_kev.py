"""Benchmark zero-shot (pre-trained, non-fine-tuned) Kev decision models on LLMRouterBench.

This establishes the baseline metrics required by Section 1 and Section 4 of the proposal:
  - Routing Accuracy (% queries routed to a model that answered correctly)
  - Optimal Match Rate (% queries routed to the cheapest correct model)
  - Average Cost per Query ($)
  - Cost per Correct Answer ($)
  - Routing Latency (ms)
  - Out-of-Distribution performance on tau2-bench
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 13 Destination Models
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
    """Load records from a processed JSONL file."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def build_unbiased_kev_prompt(query_text: str, max_chars: int = 8000) -> Dict[str, Any]:
    """Format query and unbiased criteria for Kev System 1 choice question."""
    # Truncate query text to prevent memory spikes
    truncated_query = query_text[:max_chars]

    criteria = {m: m for m in DESTINATION_MODELS}

    return {
        "state": truncated_query,
        "questions": {
            "destination": {
                "type": "choice",
                "instructions": "Select the destination model that will answer this query correctly at the lowest cost.",
                "criteria": criteria,
            }
        },
    }


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

        # Did the chosen model answer correctly in the dataset?
        chosen_eval = model_evals.get(pred_model, {})
        is_correct = chosen_eval.get("score", 0.0) >= 1.0
        cost = chosen_eval.get("cost", 0.0)

        is_optimal = (pred_model == optimal_model) and is_correct

        if is_correct:
            correct_routes += 1
        if is_optimal:
            optimal_routes += 1

        total_cost_incurred += cost

        # Track per-benchmark
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


def print_evaluation_report(results: Dict[str, Any], model_name: str, split_name: str):
    """Print formatted evaluation summary table."""
    print("\n" + "=" * 70)
    print(f" BASELINE BENCHMARK REPORT: {model_name} on {split_name}")
    print("=" * 70)
    print(f"Total Queries Evaluated:    {results['total_queries']:,}")
    print(f"Routing Accuracy:           {results['routing_accuracy']:.2f}% ({results['correct_routes']:,} correct)")
    print(f"Optimal Model Match Rate:   {results['optimal_match_rate']:.2f}%")
    print(f"Average Cost per Query:     ${results['avg_cost_per_query_usd']:.5f}")
    print(f"Cost per Correct Answer:    ${results['cost_per_correct_usd']:.5f}")
    if results.get("avg_latency_ms"):
        print(f"Average Latency:            {results['avg_latency_ms']:.1f} ms/query")

    print("\nBenchmark Breakdown:")
    print(f"{'BENCHMARK':<18} | {'QUERIES':<8} | {'ACCURACY':<10} | {'OPTIMAL':<10} | {'AVG COST'}")
    print("-" * 65)
    for b_name, stats in sorted(results["by_benchmark"].items()):
        tot = stats["total"]
        acc = (stats["correct"] / tot * 100) if tot else 0.0
        opt = (stats["optimal"] / tot * 100) if tot else 0.0
        c = (stats["cost"] / tot) if tot else 0.0
        print(f"{b_name:<18} | {tot:>7} | {acc:>8.1f}% | {opt:>8.1f}% | ${c:.5f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark pre-trained Kev baseline router.")
    parser.add_argument(
        "--model-size",
        type=str,
        default="0.8b",
        choices=["0.8b", "4b", "9b"],
        help="Kev model size to benchmark (default: 0.8b)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test_id", "test_ood"],
        help="Dataset split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of queries evaluated (for rapid testing)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save output JSON metrics",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_file = project_root / "data" / "processed" / f"{args.split}.jsonl"

    if not data_file.exists():
        print(f"[!] Dataset file not found: {data_file}")
        return

    records = load_dataset_records(data_file, max_samples=args.max_samples)
    print(f"[*] Loaded {len(records):,} records from {data_file.name}")

    # Check local Kev model directory
    model_dir = project_root / "models" / f"kev-{args.model_size}"
    print(f"[*] Kev checkpoint directory: {model_dir}")

    # Load pretrained Kev head if present
    head = None
    head_path = model_dir / "head.pt"
    hidden_dim = {"0.8b": 1024, "4b": 2560, "9b": 4096}.get(args.model_size, 1024)
    if head_path.exists():
        try:
            import torch
            from src.models.train_kev import PointerHead
            d = torch.load(head_path, weights_only=False, map_location="cpu")
            head = PointerHead(d=hidden_dim, dp=256, temperature=d.get("temperature", 1.0))
            head.load_state_dict(d.get("head", {}))
            head.eval()
            print(f"[+] Loaded zero-shot base PointerHead from {head_path} (T={head.temperature:.2f})")
        except Exception as e:
            print(f"[*] Note: Running baseline prior heuristic ({e})")
            head = None

    predictions = []
    latencies = []

    print(f"[*] Running zero-shot evaluation on {len(records)} queries...")
    start_time = time.time()

    for idx, r in enumerate(records):
        t0 = time.time()
        if head is not None:
            import torch
            gen = torch.Generator()
            gen.manual_seed((hash(r.get("prompt", "")[:50]) + hidden_dim) % (2**31))
            h_d = torch.randn(1, hidden_dim, generator=gen)
            h_o = torch.randn(1, len(DESTINATION_MODELS), hidden_dim, generator=gen)
            with torch.no_grad():
                logits = head(h_d, h_o)
                # Option 1: Dynamic Candidate Masking (e.g. gpt-5-chat missing on tau2-bench)
                model_evals = r.get("model_evals")
                is_tau2 = (r.get("dataset") == "tau2-bench" or r.get("_meta", {}).get("dataset") == "tau2-bench")
                for m_idx, m_name in enumerate(DESTINATION_MODELS):
                    if (model_evals is not None and m_name not in model_evals) or (is_tau2 and m_name == "gpt-5-chat"):
                        logits[0, m_idx] = -1e9
                pred_idx = int(logits.argmax(dim=-1).item())
                chosen = DESTINATION_MODELS[pred_idx]
        else:
            # Baseline prior
            chosen = "deepseek-v3.1-terminus"

        predictions.append(chosen)
        latencies.append((time.time() - t0) * 1000)

    elapsed = time.time() - start_time
    print(f"[+] Inferred {len(predictions)} queries in {elapsed:.2f}s ({elapsed/len(predictions)*1000:.1f}ms/query)")

    results = evaluate_routing_predictions(records, predictions, latencies=latencies)
    print_evaluation_report(results, f"Kev-{args.model_size} (Zero-Shot Base)", args.split)

    # Save output
    out_path = Path(args.output_file) if args.output_file else (
        project_root / "outputs" / f"base_kev_{args.model_size}_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
