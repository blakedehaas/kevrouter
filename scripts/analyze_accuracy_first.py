import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "outputs" / "optimization_results.json", "r", encoding="utf-8") as f:
    d = json.load(f)

sweep = d["sweep_results"]
sweep.sort(key=lambda x: (-x["routing_accuracy"], -x["optimal_match_rate"], x["cost_per_correct_usd"]))

print("=" * 85)
print(" TOP CONFIGURATIONS IN VALIDATION SWEEP (Accuracy -> Optimal Match -> Cost/Correct)")
print("=" * 85)
print(f"{'LOSS FORMULATION':<28} | {'RULE':<12} | {'PARAM':<6} | {'ACCURACY':<10} | {'OPTIMAL':<10} | {'COST/CORRECT'}")
print("-" * 85)
for r in sweep[:20]:
    print(
        f"{r['loss_name']:<28} | {r['rule']:<12} | {r['param']:<6.1f} | "
        f"{r['routing_accuracy']:>8.2f}% | {r['optimal_match_rate']:>8.2f}% | "
        f"${r['cost_per_correct_usd']:>10.5f}"
    )
print("=" * 85)
