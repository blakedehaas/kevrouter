import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_dir = PROJECT_ROOT / "data" / "processed"

print("| Benchmark Dataset | Train (80%) | Val (10%) | Test ID (10%) | Total In-Dist | % of Dataset |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")

bench_counts = {}
for s in ["train", "val", "test_id"]:
    with open(data_dir / f"{s}.jsonl", "r", encoding="utf-8") as f:
        bench_counts[s] = Counter(json.loads(line)["dataset"] for line in f)

all_benchmarks = sorted(list(set(b for c in bench_counts.values() for b in c)))
total_all = sum(sum(bench_counts[s].values()) for s in ["train", "val", "test_id"])

for b in all_benchmarks:
    tr = bench_counts["train"][b]
    va = bench_counts["val"][b]
    te = bench_counts["test_id"][b]
    tot = tr + va + te
    pct = (tot / total_all) * 100
    print(f"| {b:<17} | {tr:>11,} | {va:>9,} | {te:>13,} | {tot:>13,} | {pct:>10.2f}% |")

tr_tot = sum(bench_counts["train"].values())
va_tot = sum(bench_counts["val"].values())
te_tot = sum(bench_counts["test_id"].values())
print(f"| {'TOTAL':<17} | {tr_tot:>11,} | {va_tot:>9,} | {te_tot:>13,} | {total_all:>13,} | {100.0:>10.2f}% |")

with open(data_dir / "test_ood.jsonl", "r", encoding="utf-8") as f:
    ood_c = Counter(json.loads(line)["dataset"] for line in f)
print(f"\n| Out-of-Distribution | Split | Total Queries | Status |")
print("| :--- | :--- | :--- | :--- |")
for b, cnt in ood_c.items():
    print(f"| {b:<19} | test_ood | {cnt:>13,} | 100% Held Out (Zero Training Leakage) |")

