#!/usr/bin/env python3
"""How many records to generate so the fine-tune vs baseline comparison can be statistically significant.

    python3 scripts/plan_size.py workload.json                              # default: detect +5 accuracy points at 80% power
    python3 scripts/plan_size.py workload.json --min-gain 0.03 --baseline-acc 0.85
"""
import argparse
import json
import math
import sys
from pathlib import Path

Z = {0.8: 0.8416, 0.9: 1.2816, 0.95: 1.6449}
Z_ALPHA = 1.96   # two-sided 95%


def paired_questions(gain, regressions, power):
    """McNemar sample size: n = (z_a sqrt(p_d) + z_b sqrt(p_d - gain^2))^2 / gain^2, p_d = share of discordant pairs."""
    p_d = gain + 2 * regressions
    return math.ceil((Z_ALPHA * math.sqrt(p_d) + Z[power] * math.sqrt(max(p_d - gain ** 2, 1e-9))) ** 2 / gain ** 2)


def unpaired_questions(p1, gain, power):
    p2 = min(p1 + gain, 0.999)
    return math.ceil((Z_ALPHA + Z[power]) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / gain ** 2)


def plan(questions_per_record, baseline_acc, gain, regressions, power, development, calibration):
    q_paired = paired_questions(gain, regressions, power)
    q_unpaired = unpaired_questions(baseline_acc, gain, power)
    dev_records = math.ceil(q_paired / questions_per_record)
    cal_records = math.ceil(max(100 / questions_per_record, dev_records * calibration / development))   # >= 100 questions for a stable temperature fit
    total = math.ceil(max(dev_records / development, cal_records / calibration))
    return {
        "development_questions": q_paired,
        "development_questions_unpaired_bound": q_unpaired,
        "development_records": dev_records,
        "calibration_records": cal_records,
        "train_records": total - dev_records - cal_records,
        "total_records": total,
        "assumptions": {
            "baseline_acc": baseline_acc,
            "min_gain": gain,
            "regressions": regressions,
            "power": power,
            "alpha": 0.05,
            "questions_per_record": questions_per_record,
            "development": development,
            "calibration": calibration,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Calculate required dataset size for Kev fine-tuning.")
    ap.add_argument("workload", nargs="?", default="workload.json", help="Path to workload.json")
    ap.add_argument("--min-gain", type=float, default=0.05, help="Minimum accuracy gain to detect (default: 0.05)")
    ap.add_argument("--baseline-acc", type=float, default=0.50, help="Expected baseline accuracy (default: 0.50)")
    ap.add_argument("--regressions", type=float, default=0.03, help="Discordant regression fraction (default: 0.03)")
    ap.add_argument("--power", type=float, default=0.80, choices=[0.8, 0.9, 0.95], help="Statistical power (default: 0.8)")
    ap.add_argument("--development", type=float, default=0.10, help="Development fraction (default: 0.10)")
    ap.add_argument("--calibration", type=float, default=0.10, help="Calibration fraction (default: 0.10)")
    args = ap.parse_args()

    q_per_rec = 1
    if Path(args.workload).exists():
        with open(args.workload, encoding="utf-8") as f:
            w = json.load(f)
            q_per_rec = max(1, len(w.get("questions", {})))

    result = plan(q_per_rec, args.baseline_acc, args.min_gain, args.regressions, args.power, args.development, args.calibration)
    print("=== Dataset Sizing Plan ===")
    print(f"Target Accuracy Gain:     +{args.min_gain*100:.1f}% (Power {int(args.power*100)}%)")
    print(f"Questions per Record:     {q_per_rec}")
    print(f"Required Dev Questions:   {result['development_questions']:,} (paired McNemar)")
    print(f"Development Records:      {result['development_records']:,}")
    print(f"Calibration Records:      {result['calibration_records']:,}")
    print(f"Training Records:         {result['train_records']:,}")
    print(f"Total Recommended Sizing: {result['total_records']:,} records")


if __name__ == "__main__":
    main()
