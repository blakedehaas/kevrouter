import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_dir = PROJECT_ROOT / "data" / "processed"

lens = defaultdict(list)
char_lens = defaultdict(list)

with open(data_dir / "train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        ds = r["dataset"]
        p = r.get("prompt") or r.get("origin_query") or ""
        lens[ds].append(len(p.split()))
        char_lens[ds].append(len(p))

print("| Dataset | Avg Words (~Tokens) | Median Words | 90th %ile Words | Max Words | Max Chars |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")
for ds, l in sorted(lens.items()):
    l.sort()
    avg_l = sum(l) / len(l)
    med_l = l[len(l) // 2]
    p90 = l[int(len(l) * 0.90)]
    max_c = max(char_lens[ds])
    print(f"| {ds:<16} | {avg_l:>19.1f} | {med_l:>12} | {p90:>15} | {l[-1]:>9} | {max_c:>9,} |")

