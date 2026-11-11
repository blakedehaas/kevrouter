#!/usr/bin/env python3
"""Prepare reproducible Kev records for the 4B/9B experiments."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.convert_to_kev import convert_record


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def calibration_indices(records, count, seed):
    """Select a reproducible, benchmark-stratified calibration subset."""
    if count <= 0 or count >= len(records):
        raise ValueError("Calibration size must leave nonempty calibration and development sets")
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record["dataset"]].append(i)
    quotas = {key: count * len(group) / len(records) for key, group in groups.items()}
    allocations = {key: int(quota) for key, quota in quotas.items()}
    remaining = count - sum(allocations.values())
    for key in sorted(groups, key=lambda k: (-(quotas[k] - allocations[k]), k))[:remaining]:
        allocations[key] += 1
    rng = random.Random(seed)
    selected = set()
    for key in sorted(groups):
        selected.update(rng.sample(groups[key], allocations[key]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-size", type=int, default=500)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = {name: read_jsonl(ROOT / "data" / "processed" / (name + ".jsonl"))
           for name in ("train", "val", "test_id", "test_ood")}
    cal_indices = calibration_indices(raw["val"], args.calibration_size, args.seed)
    manifest = {"seed": args.seed, "calibration_size": args.calibration_size,
                "max_tokens_cl100k_base": 1024, "splits": {}}
    for name, records in raw.items():
        converted = []
        for i, record in enumerate(records):
            item = convert_record(record, max_tokens=1024)
            if item is None:
                raise ValueError(f"Empty prompt in {name}, row {i}")
            item["_meta"]["processed_row"] = i
            item["_meta"]["processed_split"] = name
            item["_meta"]["query_hash"] = record["query_hash"]
            converted.append(item)
        variants = {name: converted}
        if name == "val":
            variants.update(calibration=[r for i, r in enumerate(converted) if i in cal_indices],
                            development=[r for i, r in enumerate(converted) if i not in cal_indices])
        for part, rows in variants.items():
            dest = out / (part + ".jsonl")
            with dest.open("w", encoding="utf-8") as stream:
                for item in rows:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            manifest["splits"][part] = {"records": len(rows), "sha256": digest,
                                        "datasets": dict(Counter(r["_meta"]["dataset"] for r in rows))}
            print(part, len(rows), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
