"""Align, deduplicate, label, and split LLMRouterBench data for KevRouter.

This script implements the exact data specification in the project proposal:
  - 13 Flagship Destination Models:
      1. claude-sonnet-4
      2. deepseek-r1-0528
      3. deepseek-v3-0324
      4. deepseek-v3.1-terminus
      5. gemini-2.5-flash
      6. gemini-2.5-pro
      7. glm-4.6
      8. gpt-5
      9. gpt-5-chat
     10. intern-s1
     11. kimi-k2-0905
     12. qwen3-235b-a22b-2507
     13. qwen3-235b-a22b-thinking-2507

  - 9 Primary Datasets (12,168 total queries):
      - AIME (60)
      - LiveMathBench (121)
      - LiveCodeBench (1,055)
      - SWE-Bench (500)
      - GPQA (198)
      - HLE (2,158)
      - MMLU-Pro (3,000)
      - SimpleQA (4,326)
      - ArenaHard (750)

  - 1 Held-Out OOD Dataset:
      - tau2-bench (278 queries)

  - Splits:
      - Deterministic 80/10/10 split stratified by dataset & optimal routing label:
        * 9,734 training queries
        * 1,217 validation queries
        * 1,217 in-distribution test queries
        * 278 out-of-distribution queries (tau2-bench)
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 13 Destination Models specified in LLMRouterBench & project proposal
DESTINATION_MODELS_13 = [
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    "deepseek-v3.1-terminus",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "glm-4.6",
    "gpt-5",
    "gpt-5-chat",
    "intern-s1",
    "kimi-k2-0905",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
]

# Primary benchmarks and their canonical target splits
PRIMARY_BENCHMARKS = {
    "aime": {"split": "hybrid", "expected_count": 60, "name": "AIME"},
    "livemathbench": {"split": "test", "expected_count": 121, "name": "LiveMathBench"},
    "livecodebench": {"split": "test", "expected_count": 1055, "name": "LiveCodeBench"},
    "swe-bench": {"split": "verified", "expected_count": 500, "name": "SWE-Bench"},
    "gpqa": {"split": "test", "expected_count": 198, "name": "GPQA"},
    "hle": {"split": "test", "expected_count": 2158, "name": "HLE"},
    "mmlupro": {"split": "test_3000", "expected_count": 3000, "name": "MMLU-Pro"},
    "simpleqa": {"split": "test", "expected_count": 4326, "name": "SimpleQA"},
    "arenahard": {"split": "test", "expected_count": 750, "name": "ArenaHard"},
}

OOD_BENCHMARK = {
    "tau2": {"split": "test", "expected_count": 278, "name": "tau2-bench"}
}


def normalize_query_text(text: str) -> str:
    """Normalize text by stripping whitespace and standardizing spacing for hashing."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def compute_query_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized query text."""
    norm = normalize_query_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def load_dataset_model_records(
    dataset_dir: Path,
    target_split: str,
    target_models: List[str],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load JSON records for target models in a benchmark directory.

    Returns:
        Dict mapping model_name -> {record_index: record_dict}
    """
    model_records: Dict[str, Dict[int, Dict[str, Any]]] = {}

    json_files = list(dataset_dir.rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            m_name = data.get("model_name")
            sp = data.get("split")

            if m_name not in target_models:
                continue

            # Check split match
            if target_split and sp != target_split:
                continue

            records = data.get("records", [])
            indexed = {}
            for idx, r in enumerate(records):
                rec_idx = r.get("index", r.get("record_index", idx))
                indexed[rec_idx] = r

            model_records[m_name] = indexed
        except Exception as e:
            print(f"[!] Error reading {jf}: {e}")

    return model_records


def align_query_instances(
    model_records: Dict[str, Dict[int, Dict[str, Any]]],
    dataset_name: str,
    required_models: List[str],
) -> List[Dict[str, Any]]:
    """Align query records across all required models, dropping incomplete instances."""
    # Find available models that have records
    available_models = [m for m in required_models if m in model_records and model_records[m]]
    if not available_models:
        return []

    # Find common indices
    common_indices = set(model_records[available_models[0]].keys())
    for m in available_models[1:]:
        common_indices &= set(model_records[m].keys())

    aligned = []
    for idx in sorted(common_indices):
        base_rec = model_records[available_models[0]][idx]
        query_text = (
            base_rec.get("origin_query")
            or base_rec.get("prompt")
            or base_rec.get("question")
            or ""
        )
        prompt_text = base_rec.get("prompt") or query_text

        model_evals = {}
        for m in available_models:
            m_rec = model_records[m][idx]
            model_evals[m] = {
                "score": float(m_rec.get("score", 0.0)),
                "cost": float(m_rec.get("cost", 0.0)),
                "prompt_tokens": m_rec.get("prompt_tokens"),
                "completion_tokens": m_rec.get("completion_tokens"),
                "prediction": m_rec.get("prediction"),
            }

        # Determine optimal routing target (cheapest model that answered correctly)
        correct_models = [
            (m, data["cost"])
            for m, data in model_evals.items()
            if data["score"] >= 1.0
        ]

        if correct_models:
            # Sort by cost ascending, tie-break by name
            correct_models.sort(key=lambda x: (x[1], x[0]))
            optimal_model, optimal_cost = correct_models[0]
            any_correct = True
        else:
            # Fallback to cheapest overall
            cheapest = min(model_evals.items(), key=lambda x: (x[1]["cost"], x[0]))
            optimal_model, optimal_cost = cheapest[0], cheapest[1]["cost"]
            any_correct = False

        aligned.append({
            "index": idx,
            "dataset": dataset_name,
            "origin_query": query_text,
            "prompt": prompt_text,
            "query_hash": compute_query_hash(query_text),
            "optimal_model": optimal_model,
            "optimal_cost": optimal_cost,
            "any_model_correct": any_correct,
            "model_evals": model_evals,
        })

    return aligned


def stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic stratified 80/10/10 split by optimal routing target."""
    import random
    rng = random.Random(seed)

    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        strata[r["optimal_model"]].append(r)

    train_set, val_set, test_set = [], [], []

    for label, group in sorted(strata.items()):
        shuffled = list(group)
        # Stable sort by query hash then shuffle with seed
        shuffled.sort(key=lambda x: x["query_hash"])
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure we don't exceed length
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        train_set.extend(shuffled[:n_train])
        val_set.extend(shuffled[n_train : n_train + n_val])
        test_set.extend(shuffled[n_train + n_val :])

    return train_set, val_set, test_set


def save_jsonl(records: List[Dict[str, Any]], output_path: Path):
    """Save records to a JSON Lines file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] Saved {len(records):>5,} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build aligned, deduplicated, and stratified datasets for KevRouter."
    )
    parser.add_argument(
        "--raw-bench-dir",
        type=str,
        default="data/raw/bench-release",
        help="Path to extracted benchmark release directory (default: data/raw/bench-release)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to save processed splits (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting (default: 42)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    bench_dir = (project_root / args.raw_bench_dir).resolve()
    out_dir = (project_root / args.output_dir).resolve()

    if not bench_dir.exists():
        print(f"[!] Directory {bench_dir} does not exist.")
        return

    print("=== KevRouter: Dataset Processing, Alignment & Stratification ===")
    print(f"Reading Benchmarks from: {bench_dir}")
    print(f"Saving Processed Data to: {out_dir}")

    primary_records_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    seen_hashes: Set[str] = set()
    duplicates_dropped = 0

    print("\n--- 1. Processing 9 Primary Benchmarks ---")
    for ds_key, cfg in PRIMARY_BENCHMARKS.items():
        ds_dir = bench_dir / ds_key
        if not ds_dir.exists():
            print(f"[!] Missing benchmark directory: {ds_dir}")
            continue

        model_recs = load_dataset_model_records(
            dataset_dir=ds_dir,
            target_split=cfg["split"],
            target_models=DESTINATION_MODELS_13,
        )

        aligned = align_query_instances(
            model_records=model_recs,
            dataset_name=cfg["name"],
            required_models=list(model_recs.keys()),
        )

        # Deduplication
        deduped = []
        for r in aligned:
            q_hash = r["query_hash"]
            if q_hash in seen_hashes:
                duplicates_dropped += 1
                continue
            seen_hashes.add(q_hash)
            deduped.append(r)

        primary_records_by_dataset[cfg["name"]] = deduped
        print(
            f"  {cfg['name']:<18} | Target: {cfg['expected_count']:>5} | "
            f"Aligned: {len(aligned):>5} | Cleaned: {len(deduped):>5}"
        )

    print("\n--- 2. Processing Held-Out OOD Benchmark (tau2-bench) ---")
    tau2_dir = bench_dir / "tau2"
    tau2_records = []
    if tau2_dir.exists():
        model_recs = load_dataset_model_records(
            dataset_dir=tau2_dir,
            target_split="test",
            target_models=DESTINATION_MODELS_13,
        )
        tau2_aligned = align_query_instances(
            model_records=model_recs,
            dataset_name="tau2-bench",
            required_models=list(model_recs.keys()),
        )
        for r in tau2_aligned:
            tau2_records.append(r)
        print(f"  tau2-bench         | Target:   278 | Aligned: {len(tau2_records):>5}")

    # 3. Stratified 80/10/10 Splits
    all_train, all_val, all_test = [], [], []

    print("\n--- 3. Stratified 80/10/10 Splitting by Benchmark ---")
    print(f"{'BENCHMARK':<18} | {'TOTAL':<7} | {'TRAIN (80%)':<12} | {'VAL (10%)':<10} | {'TEST (10%)'}")
    print("-" * 65)

    for b_name, recs in sorted(primary_records_by_dataset.items()):
        tr, va, te = stratified_split(recs, train_ratio=0.8, val_ratio=0.1, seed=args.seed)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        print(f"{b_name:<18} | {len(recs):>5} | {len(tr):>10} | {len(va):>8} | {len(te):>8}")

    print("=" * 65)
    total_primary = len(all_train) + len(all_val) + len(all_test)
    print(f"Total Primary Queries:    {total_primary:>6,} (Target: 12,168)")
    print(f"  - Training Set:         {len(all_train):>6,} (Target: 9,734)")
    print(f"  - Validation Set:       {len(all_val):>6,} (Target: 1,217)")
    print(f"  - In-Distribution Test: {len(all_test):>6,} (Target: 1,217)")
    print(f"Held-Out OOD (tau2):      {len(tau2_records):>6,} (Target: 278)")
    print(f"Duplicates Dropped:       {duplicates_dropped:>6,}")

    # 4. Save splits to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_train, out_dir / "train.jsonl")
    save_jsonl(all_val, out_dir / "val.jsonl")
    save_jsonl(all_test, out_dir / "test_id.jsonl")
    save_jsonl(tau2_records, out_dir / "test_ood.jsonl")

    # 5. Metadata Summary
    summary = {
        "seed": args.seed,
        "primary_benchmarks": list(PRIMARY_BENCHMARKS.keys()),
        "destination_models": DESTINATION_MODELS_13,
        "counts": {
            "train": len(all_train),
            "val": len(all_val),
            "test_id": len(all_test),
            "test_ood": len(tau2_records),
            "total_primary": total_primary,
            "duplicates_dropped": duplicates_dropped,
        },
        "target_model_distribution_train": dict(
            Counter(r["optimal_model"] for r in all_train)
        ),
    }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Dataset summary saved to {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
