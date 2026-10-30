import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.benchmark_kev import load_dataset_records, evaluate_routing_predictions
from src.models.evaluate_kev import predict_routing

HIDDEN_DIMS = {"0.8b": 1024, "4b": 2560, "9b": 4096}
TEST_ID_RECORDS = load_dataset_records(PROJECT_ROOT / "data" / "processed" / "test_id.jsonl")
TEST_OOD_RECORDS = load_dataset_records(PROJECT_ROOT / "data" / "processed" / "test_ood.jsonl")

# Explore mu values near 0: pure accuracy (0.0), slight cost nudge (0.05, 0.1, 0.25, 0.5)
MU_CANDIDATES = [0.0, 0.05, 0.1, 0.25, 0.5]
CASCADE_TAU_CANDIDATES = [0.6, 0.7, 0.8, 0.9]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

results = []

for size in ["0.8b", "4b", "9b"]:
    ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / f"kev_{size}_finetuned"
    h_dim = HIDDEN_DIMS[size]

    # 1. Test Argmax (mu=0.0)
    for split_name, recs in [("test_id", TEST_ID_RECORDS), ("test_ood", TEST_OOD_RECORDS)]:
        preds, lats = predict_routing(recs, checkpoint_dir=ckpt, hidden_dim=h_dim, rule="argmax", mu=0.0)
        eval_res = evaluate_routing_predictions(recs, preds, latencies=lats)
        results.append({
            "model_size": f"Kev-{size}",
            "split": split_name,
            "strategy": "Argmax (mu=0.0, Pure Acc)",
            "accuracy": eval_res["routing_accuracy"],
            "optimal_match": eval_res["optimal_match_rate"],
            "cost_per_correct": eval_res["cost_per_correct_usd"],
            "avg_cost": eval_res["avg_cost_per_query_usd"],
        })

    # 2. Test gentle Cost-Utility (mu = 0.05, 0.1, 0.25)
    for mu in [0.05, 0.1, 0.25]:
        for split_name, recs in [("test_id", TEST_ID_RECORDS), ("test_ood", TEST_OOD_RECORDS)]:
            preds, lats = predict_routing(recs, checkpoint_dir=ckpt, hidden_dim=h_dim, rule="cost_utility", mu=mu)
            eval_res = evaluate_routing_predictions(recs, preds, latencies=lats)
            results.append({
                "model_size": f"Kev-{size}",
                "split": split_name,
                "strategy": f"Gentle Utility (mu={mu})",
                "accuracy": eval_res["routing_accuracy"],
                "optimal_match": eval_res["optimal_match_rate"],
                "cost_per_correct": eval_res["cost_per_correct_usd"],
                "avg_cost": eval_res["avg_cost_per_query_usd"],
            })

    # 3. Test High-Confidence Cascading (tau=0.7, 0.8)
    for tau in [0.7, 0.8]:
        for split_name, recs in [("test_id", TEST_ID_RECORDS), ("test_ood", TEST_OOD_RECORDS)]:
            preds, lats = predict_routing(recs, checkpoint_dir=ckpt, hidden_dim=h_dim, rule="cascading", mu=tau)
            eval_res = evaluate_routing_predictions(recs, preds, latencies=lats)
            results.append({
                "model_size": f"Kev-{size}",
                "split": split_name,
                "strategy": f"Cascading (tau={tau})",
                "accuracy": eval_res["routing_accuracy"],
                "optimal_match": eval_res["optimal_match_rate"],
                "cost_per_correct": eval_res["cost_per_correct_usd"],
                "avg_cost": eval_res["avg_cost_per_query_usd"],
            })

print("\n" + "=" * 95)
print(" ACCURACY-FIRST EVALUATION SWEEP ON HELD-OUT TEST SPLITS")
print("=" * 95)
print(f"{'MODEL':<9} | {'SPLIT':<9} | {'STRATEGY':<26} | {'ACCURACY':<10} | {'OPTIMAL':<10} | {'COST/CORRECT'}")
print("-" * 95)

# Sort test_id results by: Accuracy -> Optimal Match -> Cost/Correct
test_id_res = [r for r in results if r["split"] == "test_id"]
test_id_res.sort(key=lambda r: (-r["accuracy"], -r["optimal_match"], r["cost_per_correct"]))
for r in test_id_res[:15]:
    print(f"{r['model_size']:<9} | {r['split']:<9} | {r['strategy']:<26} | {r['accuracy']:>8.2f}% | {r['optimal_match']:>8.2f}% | ${r['cost_per_correct']:>10.5f}")

print("\n" + "-" * 95)
print(" TEST_OOD (OUT-OF-DISTRIBUTION) RANKING")
print("-" * 95)
test_ood_res = [r for r in results if r["split"] == "test_ood"]
test_ood_res.sort(key=lambda r: (-r["accuracy"], -r["optimal_match"], r["cost_per_correct"]))
for r in test_ood_res[:15]:
    print(f"{r['model_size']:<9} | {r['split']:<9} | {r['strategy']:<26} | {r['accuracy']:>8.2f}% | {r['optimal_match']:>8.2f}% | ${r['cost_per_correct']:>10.5f}")
print("=" * 95)
