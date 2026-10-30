#!/usr/bin/env python3
"""Turn a CSV / TSV / JSONL export of already-labelled examples into Kev training records.

    python3 scripts/convert_data.py workload.json tickets.csv --state body --label department=team --out data/support.jsonl
"""
import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import check_record, check_question  # noqa: E402


def rows_from(path):
    p = Path(path)
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            yield from (data if isinstance(data, list) else data.get("rows") or data.get("data") or [])
    else:
        with open(p, "r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, delimiter="\t" if p.suffix.lower() == ".tsv" else ",")


def get(row, column):
    val = row
    for part in column.split("."):
        if not isinstance(val, dict) or part not in val:
            return None
        val = val[part]
    return val


def main():
    ap = argparse.ArgumentParser(description="Convert CSV/JSONL to Kev training format.")
    ap.add_argument("workload", help="Path to workload.json")
    ap.add_argument("input", help="Source file (.csv, .tsv, .jsonl)")
    ap.add_argument("--state", required=True, help="Column name(s) for input state (comma-separated)")
    ap.add_argument("--label", action="append", required=True, help="Mapping question_id=column_name")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    with open(args.workload, "r", encoding="utf-8") as f:
        workload = json.load(f)

    label_map = {}
    for mapping in args.label:
        qid, col = mapping.split("=", 1)
        label_map[qid.strip()] = col.strip()

    state_cols = [c.strip() for c in args.state.split(",")]
    out_records = []

    for row in rows_from(args.input):
        state_parts = [str(get(row, c)) for c in state_cols if get(row, c) is not None]
        if not state_parts:
            continue
        state = "\n".join(state_parts)

        questions = {}
        for qid, q_spec in workload.get("questions", {}).items():
            col = label_map.get(qid)
            if not col:
                continue
            lbl = get(row, col)
            if lbl is None:
                continue
            q_copy = dict(q_spec)
            q_copy["label"] = lbl
            questions[qid] = q_copy

        if not questions:
            continue

        rec = {"state": state, "questions": questions}
        if not check_record(rec):
            out_records.append(rec)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[+] Converted {len(out_records):,} records to {out_p}")


if __name__ == "__main__":
    main()
