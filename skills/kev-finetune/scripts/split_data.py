#!/usr/bin/env python3
"""Check a labelled Kev JSONL file and split it into train / calibration / development partitions.

    python3 scripts/split_data.py data/input.jsonl --out data/partitions
    python3 scripts/split_data.py data/input.jsonl            # check only, no files written
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255
STATE_CHARS_WARN = 1400   # ~384 tokens of English; longer states are dropped by the trainer


def render_length(value):
    """Rough character length of a state or option after Kev renders it."""
    if value is None:
        return 0
    if isinstance(value, (str, int, float, bool)):
        return len(str(value))
    return len(json.dumps(value, ensure_ascii=False))


def normalized_state(state):
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return " ".join(text.casefold().split())


def state_key(state):
    return hashlib.sha256(normalized_state(state).encode()).hexdigest()


def label_key(q):
    """The label as the key Kev reports probabilities under (option name / "true"|"false" / level index as a string)."""
    if q["type"] == "noul":
        return str(bool(q["label"])).lower()
    return str(q["label"])


def check_question(qid, q):
    """Return a list of problems with one labelled question (empty when it is valid)."""
    problems = []
    if not isinstance(q, dict):
        return [f"question {qid!r} is not an object"]
    t = q.get("type")
    if t not in TYPES:
        return [f"question {qid!r}: type must be one of {TYPES}, got {t!r}"]
    if "instructions" not in q or q["instructions"] in (None, ""):
        problems.append(f"question {qid!r}: instructions are missing")
    if "label" not in q:
        return problems + [f"question {qid!r}: no label"]
    label, criteria = q["label"], q.get("criteria")
    if t == "noul":
        if not isinstance(label, bool):
            problems.append(f"question {qid!r}: noul label must be true or false, got {label!r}")
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
            problems.append(f"question {qid!r}: noul criteria may only describe 'true' and 'false'")
    elif t == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            problems.append(f"question {qid!r}: choice criteria must be an object with 1..{MAX_OPTIONS} options")
        elif label not in criteria:
            problems.append(f"question {qid!r}: label {label!r} is not one of the criteria {list(criteria)}")
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_OPTIONS:
            problems.append(f"question {qid!r}: score criteria must be a list of 2..{MAX_OPTIONS} ordered levels")
        elif isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(criteria):
            problems.append(f"question {qid!r}: score label must be a level index 0..{len(criteria) - 1}, got {label!r}")
    target = q.get("target")
    if target is not None:
        keys = ["false", "true"] if t == "noul" else (list(criteria) if t == "choice" else [str(i) for i in range(len(criteria or []))])
        if not isinstance(target, dict) or set(target) - set(keys) or not all(isinstance(v, (int, float)) and v >= 0 for v in target.values()) or sum(target.values()) <= 0:
            problems.append(f"question {qid!r}: target must map option keys to nonnegative weights with positive total")
    return problems


def check_record(record):
    problems = []
    if not isinstance(record, dict) or "state" not in record:
        return ["record needs a state"]
    if record["state"] in (None, ""):
        problems.append("state is empty")
    qs = record.get("questions")
    if not isinstance(qs, dict) or not qs:
        return problems + ["record needs a non-empty questions object"]
    for qid, q in qs.items():
        problems += check_question(qid, q)
    return problems


def read_records(path):
    """Read and validate records from JSONL file."""
    valid, errors = [], []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                errors.append(f"Line {idx}: JSON parse error ({e})")
                continue
            errs = check_record(rec)
            if errs:
                errors.append(f"Line {idx}: " + "; ".join(errs))
            else:
                valid.append(rec)
    return valid, errors


def split_records(records, dev_frac=0.10, cal_frac=0.10, seed=42):
    """Split records by distinct state to prevent data leakage."""
    rng = random.Random(seed)
    by_state = defaultdict(list)
    for r in records:
        by_state[state_key(r["state"])].append(r)

    keys = list(by_state.keys())
    rng.shuffle(keys)

    n_total = len(records)
    n_dev = int(n_total * dev_frac)
    n_cal = int(n_total * cal_frac)

    train, cal, dev = [], [], []
    for k in keys:
        group = by_state[k]
        if len(dev) < n_dev:
            dev.extend(group)
        elif len(cal) < n_cal:
            cal.extend(group)
        else:
            train.extend(group)

    return train, cal, dev


def main():
    ap = argparse.ArgumentParser(description="Validate and split Kev JSONL records.")
    ap.add_argument("data", help="Input JSONL file")
    ap.add_argument("--out", default=None, help="Output directory to write train/calibration/development.jsonl")
    ap.add_argument("--dev-frac", type=float, default=0.10, help="Fraction for development partition (default: 0.10)")
    ap.add_argument("--cal-frac", type=float, default=0.10, help="Fraction for calibration partition (default: 0.10)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    records, errors = read_records(args.data)
    print(f"[*] Read {len(records):,} valid records from {args.data}")
    if errors:
        print(f"[!] Encountered {len(errors):,} record errors. First 5:")
        for err in errors[:5]:
            print("    -", err)

    if not args.out:
        print("[+] Check only mode: records validated successfully.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, cal, dev = split_records(records, dev_frac=args.dev_frac, cal_frac=args.cal_frac, seed=args.seed)

    def write_jsonl(p, rows):
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "calibration.jsonl", cal)
    write_jsonl(out_dir / "development.jsonl", dev)

    summary = {
        "total_records": len(records),
        "train": len(train),
        "calibration": len(cal),
        "development": len(dev),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Successfully wrote partitions to {out_dir}:")
    print(f"    - train.jsonl:       {len(train):,} records")
    print(f"    - calibration.jsonl: {len(cal):,} records")
    print(f"    - development.jsonl: {len(dev):,} records")


if __name__ == "__main__":
    main()
