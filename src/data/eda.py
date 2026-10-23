"""Exploratory Data Analysis (EDA) for KevRouter.

Analyzes processed or raw benchmark data to answer:
  1. Dataset distributions and record counts across the 9 primary benchmarks + tau2 OOD.
  2. Prompt length & token distributions (verifying the ~1,024 tokens/query assumption).
  3. Individual destination model performance (accuracy) and cost per query across benchmarks.
  4. Cost-sensitive optimal routing target distributions (class balance across 13 destination models).
  5. Routing potential: Cost savings of Oracle Routing vs Best Single Model vs Cheapest Model.
  6. Failure rates: Queries where no destination model answered correctly.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def analyze_token_lengths(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate character, word, and token statistics."""
    char_lengths = []
    word_lengths = []
    prompt_tokens = []

    for r in records:
        text = r.get("prompt") or r.get("origin_query") or ""
        char_lengths.append(len(text))
        word_lengths.append(len(text.split()))

        # Check if any model logged prompt_tokens
        evals = r.get("model_evals", {})
        for m_data in evals.values():
            pt = m_data.get("prompt_tokens")
            if pt is not None:
                prompt_tokens.append(int(pt))
                break

    def stats(arr):
        if not arr:
            return {"mean": 0, "min": 0, "max": 0, "p50": 0}
        arr_sorted = sorted(arr)
        return {
            "mean": sum(arr_sorted) / len(arr_sorted),
            "min": arr_sorted[0],
            "max": arr_sorted[-1],
            "p50": arr_sorted[len(arr_sorted) // 2],
            "p95": arr_sorted[int(len(arr_sorted) * 0.95)],
        }

    return {
        "chars": stats(char_lengths),
        "words": stats(word_lengths),
        "tokens": stats(prompt_tokens) if prompt_tokens else stats(word_lengths),
    }


def analyze_destination_models(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute aggregate accuracy, average cost, and win rate for each model."""
    model_scores = defaultdict(list)
    model_costs = defaultdict(list)
    model_wins = Counter()
    total_queries = len(records)

    for r in records:
        opt_model = r.get("optimal_model")
        if opt_model:
            model_wins[opt_model] += 1

        for m, data in r.get("model_evals", {}).items():
            model_scores[m].append(data.get("score", 0.0))
            model_costs[m].append(data.get("cost", 0.0))

    summary = {}
    for m in model_scores:
        scores = model_scores[m]
        costs = model_costs[m]
        summary[m] = {
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "avg_cost_per_query": sum(costs) / len(costs) if costs else 0.0,
            "total_cost": sum(costs),
            "optimal_routing_count": model_wins[m],
            "optimal_routing_share": (model_wins[m] / total_queries) * 100 if total_queries else 0.0,
        }

    return summary


def run_eda(data_dir: Path, output_report: Optional[Path] = None) -> Dict[str, Any]:
    """Execute EDA across all processed splits."""
    report: Dict[str, Any] = {}

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    test_id_file = data_dir / "test_id.jsonl"
    test_ood_file = data_dir / "test_ood.jsonl"

    if not train_file.exists():
        print(f"[!] train.jsonl not found in {data_dir}. Run build_dataset.py first.")
        return report

    train_recs = load_jsonl(train_file)
    val_recs = load_jsonl(val_file)
    test_id_recs = load_jsonl(test_id_file)
    test_ood_recs = load_jsonl(test_ood_file) if test_ood_file.exists() else []

    all_primary = train_recs + val_recs + test_id_recs

    # 1. Split & Dataset Counts
    print_section("1. Dataset & Split Distributions")
    print(f"Primary Training Set:         {len(train_recs):>6,} queries")
    print(f"Primary Validation Set:       {len(val_recs):>6,} queries")
    print(f"Primary Test Set (In-Dist):   {len(test_id_recs):>6,} queries")
    print(f"Total Primary Data Pool:      {len(all_primary):>6,} queries (Target: 12,168)")
    print(f"Held-out OOD Set (tau2):      {len(test_ood_recs):>6,} queries (Target: 278)")

    # Breakdown by benchmark
    dataset_counts = Counter(r.get("dataset", "Unknown") for r in all_primary)
    print("\nPrimary Breakdown by Benchmark:")
    for ds, cnt in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        print(f"  - {ds:<18}: {cnt:>5,} queries ({cnt / len(all_primary) * 100:>5.1f}%)")

    # 2. Token Lengths
    print_section("2. Query Token Length Analysis (VRAM / Memory Check)")
    length_stats = analyze_token_lengths(all_primary)
    tok_s = length_stats["tokens"]
    print(f"Token Estimates (Proposal assumed ~1,024 avg tokens/query):")
    print(f"  - Mean Tokens: {tok_s['mean']:>7.1f}")
    print(f"  - Median (P50): {tok_s['p50']:>6.1f}")
    print(f"  - 95th Percentile: {tok_s['p95']:>6.1f}")
    print(f"  - Max Tokens: {tok_s['max']:>7.1f}")

    # 3. Destination Models Performance & Optimal Routing Share
    print_section("3. Destination Models & Optimal Routing Target Distribution")
    model_stats = analyze_destination_models(all_primary)

    print(
        f"{'MODEL NAME':<28} | {'ACCURACY':<9} | {'AVG COST/Q':<12} | "
        f"{'OPTIMAL WINS':<12} | {'ROUTER SHARE'}"
    )
    print("-" * 75)

    # Sort by accuracy descending
    sorted_models = sorted(model_stats.items(), key=lambda x: -x[1]["accuracy"])
    for m, stats in sorted_models:
        acc_pct = stats["accuracy"] * 100
        cost_str = f"${stats['avg_cost_per_query']:.5f}"
        wins = stats["optimal_routing_count"]
        share_pct = stats["optimal_routing_share"]
        print(
            f"{m:<28} | {acc_pct:>7.2f}% | {cost_str:>12} | {wins:>12,} | {share_pct:>6.1f}%"
        )

    # 4. Solvability & Routing Ceiling
    print_section("4. Routing Ceiling & Cost Savings Potential")
    correctness_counts = Counter()
    total_oracle_cost = 0.0
    for r in all_primary:
        c = sum(1 for m in r.get("model_evals", {}).values() if m.get("score", 0.0) >= 1.0)
        correctness_counts[c] += 1
        total_oracle_cost += r.get("optimal_cost", 0.0)

    no_correct = correctness_counts[0]
    multi_correct = sum(v for k, v in correctness_counts.items() if k > 1)
    exactly_one = correctness_counts[1]

    print(f"Queries where NO destination model succeeded:    {no_correct:>5,} ({no_correct/len(all_primary)*100:>5.1f}%)")
    print(f"Queries where EXACTLY ONE model succeeded:      {exactly_one:>5,} ({exactly_one/len(all_primary)*100:>5.1f}%)")
    print(f"Queries with MULTIPLE correct models (Choices):  {multi_correct:>5,} ({multi_correct/len(all_primary)*100:>5.1f}%)")

    # Oracle accuracy vs Best single model
    oracle_acc = ((len(all_primary) - no_correct) / len(all_primary)) * 100
    best_single = sorted_models[0]
    best_single_name = best_single[0]
    best_single_acc = best_single[1]["accuracy"] * 100
    best_single_cost = best_single[1]["total_cost"]

    oracle_avg_cost = total_oracle_cost / len(all_primary)
    best_single_avg_cost = best_single[1]["avg_cost_per_query"]
    cost_reduction = (
        (best_single_cost - total_oracle_cost) / best_single_cost * 100
        if best_single_cost > 0
        else 0.0
    )

    print(f"\nOracle vs Best Single Model ({best_single_name}):")
    print(f"  - Best Single Model Accuracy: {best_single_acc:.2f}% (Avg Cost: ${best_single_avg_cost:.5f}/query)")
    print(f"  - Oracle Router Accuracy:     {oracle_acc:.2f}% (Avg Cost: ${oracle_avg_cost:.5f}/query)")
    print(f"  - Accuracy Gain:             +{oracle_acc - best_single_acc:.2f}%")
    print(f"  - Theoretical Cost Savings:   {cost_reduction:.1f}% reduction vs {best_single_name}")

    report = {
        "dataset_counts": dict(dataset_counts),
        "split_counts": {
            "train": len(train_recs),
            "val": len(val_recs),
            "test_id": len(test_id_recs),
            "test_ood": len(test_ood_recs),
        },
        "token_stats": length_stats,
        "model_performance": model_stats,
        "routing_potential": {
            "oracle_accuracy": oracle_acc,
            "best_single_accuracy": best_single_acc,
            "best_single_model": best_single_name,
            "oracle_avg_cost": oracle_avg_cost,
            "best_single_avg_cost": best_single_avg_cost,
            "cost_reduction_pct": cost_reduction,
        },
    }

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Full report saved to {output_report}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on KevRouter data.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing processed JSONL files (default: data/processed)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/processed/eda_report.json",
        help="Path to save output JSON report (default: data/processed/eda_report.json)",
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = (project_root / args.data_dir).resolve()
    report_file = (project_root / args.output_report).resolve()

    run_eda(data_dir, output_report=report_file)


if __name__ == "__main__":
    main()
