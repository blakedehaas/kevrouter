"""Parse, align, deduplicate, and split LLMRouterBench data for KevRouter.

This script implements the exact data specification in the project proposal:
  1. Filters to the 9 primary benchmarks: AIME, LiveMathBench, LiveCodeBench,
     SWE-Bench, GPQA, HLE, MMLU-Pro, SimpleQA, and ArenaHard.
  2. Holds out tau2-Bench (278 queries) exclusively for Out-Of-Distribution (OOD) evaluation.
  3. Verifies complete model results for all destination models per query.
  4. Deduplicates queries via normalization and cryptographic hashing.
  5. Computes the optimal cost-sensitive routing label per query:
     - The least expensive destination model that answered correctly (score = 1.0).
  6. Creates deterministic 80/10/10 stratified splits within each benchmark.
  7. Outputs train, validation, in-distribution test, and OOD test splits to data/processed/.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# Benchmark name aliases to match directory names in LLMRouterBench
PRIMARY_DATASET_MAP = {
    "aime": "AIME",
    "livemathbench": "LiveMathBench",
    "livecodebench": "LiveCodeBench",
    "swebench": "SWE-Bench",
    "swe-bench": "SWE-Bench",
    "gpqa": "GPQA",
    "hle": "HLE",
    "mmlupro": "MMLU-Pro",
    "mmlu-pro": "MMLU-Pro",
    "simpleqa": "SimpleQA",
    "arenahard": "ArenaHard",
    "arena-hard": "ArenaHard",
}

OOD_DATASET_ALIASES = ["tau2", "tau2bench", "tau2-bench", "tau_2"]


def normalize_query_text(text: str) -> str:
    """Normalize text by stripping whitespace and standardizing spacing for hashing."""
    if not text:
        return ""
    # Standardize whitespace and trim
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized


def compute_query_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized query text."""
    norm = normalize_query_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def load_raw_dataset_records(dataset_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load all model JSON files for a given dataset directory.

    Returns:
        Dict mapping model_name -> {record_index: record_dict}
    """
    model_data: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for json_file in dataset_dir.glob("*.json"):
        model_name = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            if isinstance(records, list):
                # Map by record_index or list position
                indexed_records = {}
                for idx, r in enumerate(records):
                    rec_idx = r.get("record_index", idx)
                    indexed_records[rec_idx] = r
                model_data[model_name] = indexed_records
            elif isinstance(records, dict):
                # Dict keyed by index or id
                indexed_records = {}
                for k, v in records.items():
                    try:
                        rec_idx = int(k)
                    except ValueError:
                        rec_idx = v.get("record_index", len(indexed_records))
                    indexed_records[rec_idx] = v
                model_data[model_name] = indexed_records
        except Exception as e:
            print(f"[!] Warning: failed to load {json_file}: {e}")

    return model_data


def align_and_filter_records(
    model_data: Dict[str, Dict[int, Dict[str, Any]]],
    target_models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Align query records across all destination models and retain only complete queries."""
    if not model_data:
        return []

    available_models = list(model_data.keys())
    if target_models:
        eval_models = [m for m in target_models if m in model_data]
    else:
        eval_models = available_models

    if not eval_models:
        return []

    # Find common record indices that exist across all required models
    candidate_indices = set(model_data[eval_models[0]].keys())
    for m in eval_models[1:]:
        candidate_indices &= set(model_data[m].keys())

    aligned_records = []

    for idx in sorted(candidate_indices):
        # Extract prompt / origin_query from first model
        base_record = model_data[eval_models[0]][idx]
        query_text = (
            base_record.get("origin_query")
            or base_record.get("prompt")
            or base_record.get("question")
            or ""
        )
        prompt_text = base_record.get("prompt") or query_text

        # Collect model results for this query
        models_eval = {}
        is_complete = True

        for m in eval_models:
            m_rec = model_data[m][idx]
            score = m_rec.get("score")
            cost = m_rec.get("cost")

            if score is None or cost is None:
                is_complete = False
                break

            models_eval[m] = {
                "score": float(score),
                "cost": float(cost),
                "prompt_tokens": m_rec.get("prompt_tokens"),
                "completion_tokens": m_rec.get("completion_tokens"),
                "prediction": m_rec.get("prediction"),
            }

        if not is_complete:
            continue

        aligned_records.append({
            "record_index": idx,
            "origin_query": query_text,
            "prompt": prompt_text,
            "query_hash": compute_query_hash(query_text),
            "model_evals": models_eval,
        })

    return aligned_records


def determine_optimal_routing_label(
    model_evals: Dict[str, Dict[str, float]],
) -> Tuple[Optional[str], float, bool]:
    """Find the cheapest destination model that answered correctly (score == 1.0).

    Returns:
        (best_model_name, best_model_cost, any_correct_flag)
    """
    correct_models = [
        (m, data["cost"])
        for m, data in model_evals.items()
        if data.get("score", 0.0) >= 1.0
    ]

    if not correct_models:
        # None answered correctly; fallback to lowest cost overall
        cheapest_model = min(model_evals.items(), key=lambda x: x[1]["cost"])
        return cheapest_model[0], cheapest_model[1]["cost"], False

    # Sort by cost ascending, tie-break alphabetically for determinism
    correct_models.sort(key=lambda x: (x[1], x[0]))
    return correct_models[0][0], correct_models[0][1], True


def stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic stratified 80/10/10 split based on the optimal routing label."""
    import random

    rng = random.Random(seed)

    # Group by routing target label
    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        label = r.get("optimal_model") or "NONE_CORRECT"
        strata[label].append(r)

    train_set, val_set, test_set = [], [], []

    for label, group in sorted(strata.items()):
        # Deterministic shuffle within group
        shuffled = list(group)
        shuffled.sort(key=lambda x: x["query_hash"])  # stable sort first
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

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
    print(f"[+] Saved {len(records):,} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build aligned, deduplicated, and stratified datasets for KevRouter."
    )
    parser.add_argument(
        "--raw-bench-dir",
        type=str,
        default="data/raw/bench",
        help="Path to extracted benchmark directory (default: data/raw/bench)",
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
        print(f"[!] Benchmark directory not found at {bench_dir}.")
        print("    Please run the download script first: python scripts/download_data.py")
        return

    print("=== KevRouter Dataset Alignment & Splitting ===")
    print(f"Reading from: {bench_dir}")
    print(f"Saving to:    {out_dir}")

    # Discover dataset subdirectories
    subdirs = [d for d in bench_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} benchmark subdirectories.")

    primary_records_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    ood_records: List[Dict[str, Any]] = []

    seen_hashes: Set[str] = set()
    duplicates_dropped = 0

    for d in subdirs:
        d_name_lower = d.name.lower()

        # Check if primary or OOD
        canonical_name = None
        for key, val in PRIMARY_DATASET_MAP.items():
            if key in d_name_lower:
                canonical_name = val
                break

        is_ood = any(ood in d_name_lower for ood in OOD_DATASET_ALIASES)

        if not canonical_name and not is_ood:
            continue

        print(f"[*] Processing {d.name} ({'OOD' if is_ood else canonical_name})...")
        model_data = load_raw_dataset_records(d)
        aligned = align_and_filter_records(model_data)

        # Deduplicate and compute optimal model labels
        clean_dataset_records = []
        for r in aligned:
            q_hash = r["query_hash"]
            if q_hash in seen_hashes:
                duplicates_dropped += 1
                continue
            seen_hashes.add(q_hash)

            best_model, best_cost, any_correct = determine_optimal_routing_label(
                r["model_evals"]
            )
            r["dataset"] = canonical_name if canonical_name else "tau2-bench"
            r["optimal_model"] = best_model
            r["optimal_cost"] = best_cost
            r["any_model_correct"] = any_correct

            clean_dataset_records.append(r)

        if is_ood:
            ood_records.extend(clean_dataset_records)
        else:
            primary_records_by_dataset[canonical_name] = clean_dataset_records

    # Perform stratified 80/10/10 split per primary dataset
    all_train, all_val, all_test = [], [], []

    print("\n" + "=" * 65)
    print(f"{'DATASET':<20} | {'TOTAL':<8} | {'TRAIN (80%)':<12} | {'VAL (10%)':<10} | {'TEST (10%)'}")
    print("-" * 65)

    for d_name, records in sorted(primary_records_by_dataset.items()):
        tr, va, te = stratified_split(records, train_ratio=0.8, val_ratio=0.1, seed=args.seed)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        print(f"{d_name:<20} | {len(records):<8} | {len(tr):<12} | {len(va):<10} | {len(te)}")

    print("=" * 65)
    print(f"Total Primary Queries:    {len(all_train) + len(all_val) + len(all_test):,}")
    print(f"  - Training Set:         {len(all_train):,} (Target in proposal: 9,734)")
    print(f"  - Validation Set:       {len(all_val):,} (Target in proposal: 1,217)")
    print(f"  - In-Distribution Test: {len(all_test):,} (Target in proposal: 1,217)")
    print(f"Held-out OOD (tau2-bench): {len(ood_records):,} (Target in proposal: 278)")
    print(f"Duplicate Queries Dropped: {duplicates_dropped:,}")

    # Save to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_train, out_dir / "train.jsonl")
    save_jsonl(all_val, out_dir / "val.jsonl")
    save_jsonl(all_test, out_dir / "test_id.jsonl")
    save_jsonl(ood_records, out_dir / "test_ood.jsonl")

    # Save metadata summary
    metadata = {
        "seed": args.seed,
        "counts": {
            "train": len(all_train),
            "val": len(all_val),
            "test_id": len(all_test),
            "test_ood": len(ood_records),
            "duplicates_dropped": duplicates_dropped,
        },
        "target_model_distribution_train": dict(
            Counter(r["optimal_model"] for r in all_train)
        ),
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Summary written to {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
