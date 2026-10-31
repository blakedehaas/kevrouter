import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    
    splits = {
        "train": data_dir / "train.jsonl",
        "val": data_dir / "val.jsonl",
        "test_id": data_dir / "test_id.jsonl",
        "test_ood": data_dir / "test_ood.jsonl",
    }
    
    all_rows = []
    split_rows = {}
    for s_name, path in splits.items():
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_split"] = s_name
                    rows.append(r)
        split_rows[s_name] = rows
        all_rows.extend(rows)
        
    id_rows = split_rows["train"] + split_rows["val"] + split_rows["test_id"]
    ood_rows = split_rows["test_ood"]
    
    print("=" * 95)
    print("TABLE 1: BENCHMARK SUITE OVERVIEW AND PARTITIONS")
    print("=" * 95)
    bench_groups = defaultdict(list)
    for r in id_rows:
        bench_groups[r["dataset"]].append(r)
        
    hdr = f"{'Benchmark':<16} | {'Domain / Task Description':<32} | {'Train':>6} | {'Val':>5} | {'Test':>5} | {'Total':>6} | {'Solv%':>6}"
    print(hdr)
    print("-" * len(hdr))
    
    domain_map = {
        "SimpleQA": "Factuality & Short-Form QA",
        "MMLU-Pro": "Multi-discipline College Knowledge",
        "HLE": "Frontier Multimodal Reasoning",
        "LiveCodeBench": "Algorithmic & Software Coding",
        "ArenaHard": "Complex Multi-Turn Instructions",
        "SWE-Bench": "Software Issue Resolution",
        "GPQA": "Expert Biology, Physics & Chem",
        "LiveMathBench": "Fresh Mathematical Competitions",
        "AIME": "Elite Olympiad Mathematics",
    }
    
    for ds in sorted(bench_groups.keys()):
        rows = bench_groups[ds]
        n_tr = sum(1 for r in rows if r["_split"] == "train")
        n_va = sum(1 for r in rows if r["_split"] == "val")
        n_te = sum(1 for r in rows if r["_split"] == "test_id")
        n_tot = len(rows)
        solv = sum(1 for r in rows if r.get("any_model_correct", False)) / n_tot * 100
        desc = domain_map.get(ds, "General Reasoning")
        print(f"{ds:<16} | {desc:<32} | {n_tr:>6} | {n_va:>5} | {n_te:>5} | {n_tot:>6} | {solv:>5.1f}%")
        
    print("-" * len(hdr))
    n_tr_id = len(split_rows["train"])
    n_va_id = len(split_rows["val"])
    n_te_id = len(split_rows["test_id"])
    n_tot_id = len(id_rows)
    tot_solv_id = sum(1 for r in id_rows if r.get("any_model_correct", False)) / n_tot_id * 100
    print(f"{'In-Dist Total':<16} | {'8 In-Distribution Benchmarks':<32} | {n_tr_id:>6} | {n_va_id:>5} | {n_te_id:>5} | {n_tot_id:>6} | {tot_solv_id:>5.1f}%")
    
    # OOD
    print("\n" + "=" * 95)
    print("HELD-OUT OUT-OF-DISTRIBUTION (OOD) BENCHMARK")
    print("=" * 95)
    ood_tot = len(ood_rows)
    ood_solv = sum(1 for r in ood_rows if r.get("any_model_correct", False)) / ood_tot * 100
    print(f"{'tau2-bench':<16} | {'Agentic Multi-Turn Tool Use':<32} | {'-':>6} | {'-':>5} | {ood_tot:>5} | {ood_tot:>6} | {ood_solv:>5.1f}%")

    print("\n" + "=" * 95)
    print("TABLE 2: CANDIDATE MODEL POOL (ACTION SPACE, COSTS & INDIVIDUAL ACCURACIES)")
    print("=" * 95)
    
    # Compute accuracy and cost for each model over in-distribution test set and train set
    model_stats = defaultdict(lambda: {"correct_id": 0, "total_id": 0, "total_cost_id": 0.0, "optimal_count": 0})
    
    # Fixed pricing from destination criteria
    model_pricing = {
        "deepseek-v3-0324": 0.002,
        "gpt-5-chat": 0.002,
        "deepseek-v3.1-terminus": 0.004,
        "qwen3-235b-a22b-2507": 0.007,
        "intern-s1": 0.010,
        "kimi-k2-0905": 0.012,
        "claude-sonnet-4": 0.016,
        "gemini-2.5-flash": 0.020,
        "qwen3-235b-a22b-thinking-2507": 0.021,
        "deepseek-r1-0528": 0.050,
        "glm-4.6": 0.055,
        "gpt-5": 0.150,
        "gemini-2.5-pro": 0.250,
    }
    
    for r in id_rows:
        opt_m = r.get("optimal_model")
        if opt_m in model_stats:
            model_stats[opt_m]["optimal_count"] += 1
            
        evals = r.get("model_evals", {})
        for m in model_pricing.keys():
            model_stats[m]["total_id"] += 1
            ev = evals.get(m, {})
            score = ev.get("score", 0.0)
            cost = ev.get("cost", model_pricing[m])
            if score >= 1.0:
                model_stats[m]["correct_id"] += 1
            model_stats[m]["total_cost_id"] += cost
            
    m_hdr = f"{'Candidate Model':<28} | {'Vendor/Family':<16} | {'Price/Query':>11} | {'ID Acc (%)':>10} | {'Opt Share %':>11}"
    print(m_hdr)
    print("-" * len(m_hdr))
    
    vendor_map = {
        "deepseek-v3-0324": "DeepSeek",
        "gpt-5-chat": "OpenAI",
        "deepseek-v3.1-terminus": "DeepSeek",
        "qwen3-235b-a22b-2507": "Alibaba",
        "intern-s1": "OpenGVLab",
        "kimi-k2-0905": "Moonshot",
        "claude-sonnet-4": "Anthropic",
        "gemini-2.5-flash": "Google",
        "qwen3-235b-a22b-thinking-2507": "Alibaba (Think)",
        "deepseek-r1-0528": "DeepSeek (Reason)",
        "glm-4.6": "Zhipu AI",
        "gpt-5": "OpenAI",
        "gemini-2.5-pro": "Google",
    }
    
    sorted_models = sorted(model_pricing.keys(), key=lambda m: model_pricing[m])
    for m in sorted_models:
        st = model_stats[m]
        acc = st["correct_id"] / st["total_id"] * 100 if st["total_id"] else 0.0
        opt_share = st["optimal_count"] / len(id_rows) * 100
        pr = f"${model_pricing[m]:.3f}"
        vend = vendor_map.get(m, "Open")
        print(f"{m:<28} | {vend:<16} | {pr:>11} | {acc:>9.2f}% | {opt_share:>10.2f}%")

if __name__ == "__main__":
    main()

