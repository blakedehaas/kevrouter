import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_dir = PROJECT_ROOT / "data" / "processed"
bench_dir = PROJECT_ROOT / "data" / "raw" / "bench-release"

MODELS_13 = [
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

# 1. Check raw benchmark directories
print("=" * 80)
print("1. RAW BENCHMARK FILES AVAILABLE PER MODEL")
print("=" * 80)
benchmarks = ["aime", "arenahard", "gpqa", "hle", "livecodebench", "livemathbench", "mmlupro", "simpleqa", "swe-bench", "tau2"]

for b in benchmarks:
    b_dir = bench_dir / b
    if not b_dir.exists():
        print(f"{b:<16}: DIRECTORY NOT FOUND")
        continue
    json_files = list(b_dir.rglob("*.json"))
    models_found = set()
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                m = data.get("model_name")
                if m:
                    models_found.add(m)
        except Exception:
            pass
    missing_in_raw = [m for m in MODELS_13 if m not in models_found]
    print(f"{b:<16}: {len(models_found)}/13 models found. Missing: {missing_in_raw if missing_in_raw else 'None'}")

# 2. Check processed data
print("\n" + "=" * 80)
print("2. PROCESSED EVALS & LABELS BY BENCHMARK DATASET")
print("=" * 80)
eval_presence = defaultdict(lambda: defaultdict(int))
optimal_presence = defaultdict(lambda: defaultdict(int))
dataset_sizes = defaultdict(int)

for s in ["train", "val", "test_id"]:
    with open(data_dir / f"{s}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ds = r["dataset"]
            dataset_sizes[ds] += 1
            opt = r.get("optimal_model")
            optimal_presence[ds][opt] += 1
            for m in r.get("model_evals", {}):
                eval_presence[ds][m] += 1

for ds in sorted(dataset_sizes):
    tot = dataset_sizes[ds]
    missing_evals = [m for m in MODELS_13 if m not in eval_presence[ds]]
    incomplete_evals = [m for m, cnt in eval_presence[ds].items() if cnt < tot]
    zero_opt = [m for m in MODELS_13 if optimal_presence[ds][m] == 0]
    print(f"\nBenchmark: {ds} (Total: {tot:,} queries)")
    print(f"  - Missing completely from model_evals: {missing_evals if missing_evals else 'None (All 13 evaluated)'}")
    if incomplete_evals:
        print(f"  - Incomplete evals (< total queries): {incomplete_evals}")
    print(f"  - Models NEVER chosen as optimal_model (0 times correct/cheapest): {zero_opt}")

# 3. Overall optimal target count across entire dataset
print("\n" + "=" * 80)
print("3. OVERALL OPTIMAL MODEL DISTRIBUTION ACROSS ENTIRE IN-DIST DATASET")
print("=" * 80)
overall_opts = defaultdict(int)
for ds, counts in optimal_presence.items():
    for m, c in counts.items():
        overall_opts[m] += c

for m in MODELS_13:
    c = overall_opts[m]
    pct = (c / sum(overall_opts.values())) * 100
    print(f"  {m:<30}: {c:>6,} queries ({pct:>5.2f}%)")

