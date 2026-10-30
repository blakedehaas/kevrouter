import json
from pathlib import Path
import tiktoken
import numpy as np

def main():
    enc = tiktoken.get_encoding("cl100k_base")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"

    splits = ["train.jsonl", "val.jsonl", "test_id.jsonl", "test_ood.jsonl"]
    dataset_tokens = {}
    dataset_chars = {}
    all_id_tokens = []
    all_id_chars = []
    all_tokens = []

    for s in splits:
        path = data_dir / s
        if not path.exists():
            continue
        is_ood = (s == "test_ood.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ds = r.get("dataset", "unknown")
                p = r.get("prompt") or r.get("origin_query") or ""
                c_len = len(p)
                t_len = len(enc.encode(p, disallowed_special=()))
                
                if ds not in dataset_tokens:
                    dataset_tokens[ds] = []
                    dataset_chars[ds] = []
                dataset_tokens[ds].append(t_len)
                dataset_chars[ds].append(c_len)
                all_tokens.append(t_len)
                if not is_ood:
                    all_id_tokens.append(t_len)
                    all_id_chars.append(c_len)

    print("=" * 80)
    print("PER-BENCHMARK TOKEN LENGTH PERCENTILES (cl100k_base BPE)")
    print("=" * 80)
    hdr = f"{'Dataset':<16} | {'Count':>6} | {'p50':>6} | {'p75':>6} | {'p90':>6} | {'p95':>6} | {'p99':>7} | {'Max':>8}"
    print(hdr)
    print("-" * len(hdr))
    for ds in sorted(dataset_tokens.keys()):
        arr = np.array(dataset_tokens[ds])
        p50, p75, p90, p95, p99 = np.percentile(arr, [50, 75, 90, 95, 99])
        print(f"{ds:<16} | {len(arr):>6} | {int(p50):>6} | {int(p75):>6} | {int(p90):>6} | {int(p95):>6} | {int(p99):>7} | {int(arr.max()):>8}")

    print("\n" + "=" * 80)
    print("AGGREGATE TOKEN DISTRIBUTIONS")
    print("=" * 80)
    groups = [
        ("All In-Distribution (including SWE-Bench)", all_id_tokens),
        ("In-Distribution EXCLUDING SWE-Bench", [t for ds in dataset_tokens if ds not in ['swe_bench', 'tau2', 'hle'] for t in dataset_tokens[ds]]),
        ("Entire Corpus (In-Distribution + OOD)", all_tokens),
    ]
    for name, arr_list in groups:
        arr = np.array(arr_list)
        p50, p75, p90, p95, p98, p99 = np.percentile(arr, [50, 75, 90, 95, 98, 99])
        print(f"\n{name} (N = {len(arr):,}):")
        print(f"  Mean:  {int(arr.mean()):>6} tokens")
        print(f"  p50:   {int(p50):>6} tokens")
        print(f"  p75:   {int(p75):>6} tokens")
        print(f"  p90:   {int(p90):>6} tokens")
        print(f"  p95:   {int(p95):>6} tokens")
        print(f"  p98:   {int(p98):>6} tokens")
        print(f"  p99:   {int(p99):>6} tokens")
        print(f"  Max:   {int(arr.max()):>6} tokens")

    print("\n" + "=" * 80)
    print("CHARACTER LENGTH PERCENTILES (All In-Distribution)")
    print("=" * 80)
    id_chars_arr = np.array(all_id_chars)
    p50c, p75c, p90c, p95c, p98c, p99c = np.percentile(id_chars_arr, [50, 75, 90, 95, 98, 99])
    print(f"  Mean chars: {int(id_chars_arr.mean()):,}")
    print(f"  p50 chars:  {int(p50c):,}")
    print(f"  p75 chars:  {int(p75c):,}")
    print(f"  p90 chars:  {int(p90c):,}")
    print(f"  p95 chars:  {int(p95c):,}")
    print(f"  p98 chars:  {int(p98c):,}")
    print(f"  p99 chars:  {int(p99c):,}")
    print(f"  Max chars:  {int(id_chars_arr.max()):,}")

    # Check how many prompts are truncated under various limits
    print("\n" + "=" * 80)
    print("TRUNCATION IMPACT ON IN-DISTRIBUTION PROMPTS")
    print("=" * 80)
    token_limits = [256, 384, 512, 768, 1024, 1536, 2048]
    id_tokens_arr = np.array(all_id_tokens)
    for limit in token_limits:
        pct_covered = (id_tokens_arr <= limit).mean() * 100
        truncated_count = (id_tokens_arr > limit).sum()
        print(f"  Limit = {limit:>4} tokens: {pct_covered:>6.2f}% fully retained ({truncated_count:>5,} truncated out of {len(id_tokens_arr):,})")

    # Also check character limits
    char_limits = [1400, 2000, 3000, 4000, 5000, 6000, 8000]
    for clim in char_limits:
        pct_c = (id_chars_arr <= clim).mean() * 100
        trunc_c = (id_chars_arr > clim).sum()
        print(f"  Limit = {clim:>5} chars:  {pct_c:>6.2f}% fully retained ({trunc_c:>5,} truncated out of {len(id_chars_arr):,})")

if __name__ == "__main__":
    main()
