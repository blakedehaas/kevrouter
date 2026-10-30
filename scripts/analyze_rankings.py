import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "outputs" / "model_comparison_matrix.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

for split in ["test_id", "test_ood", "val"]:
    print(f"\n{'='*30} {split.upper()} RANKING (Acc -> Opt -> Cost/Corr) {'='*30}")
    s_rows = [r for r in rows if r["split"] == split]
    s_rows.sort(key=lambda r: (-r["accuracy"], -r["optimal_match"], r["cost_per_correct"]))
    for r in s_rows:
        print(f"{r['model_size']:<9} | {r['variant']:<30} | Acc: {r['accuracy']:>6.2f}% | Opt: {r['optimal_match']:>6.2f}% | $/Corr: ${r['cost_per_correct']:>8.5f} | AvgCost: ${r['avg_cost']:>8.5f}")
