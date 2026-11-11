#!/usr/bin/env python3
"""Prepare the Qwen-admitted subset used by the original 0.8B-style run."""
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/experiments/kev_4b_9b_v1"
DEST = ROOT / "data/experiments/kev_4b_9b_v3"
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        if digest(SOURCE / f"{split}.jsonl") != info["sha256"]:
            raise ValueError(f"Original data changed: {split}")
    tokenizer_files = [ROOT / f"models/kev-{s}/tokenizer.json" for s in ("0.8b", "4b", "9b")]
    tokenizer_hashes = {digest(p) for p in tokenizer_files}
    if len(tokenizer_hashes) != 1:
        raise ValueError("0.8B, 4B and 9B tokenizers differ")
    tokenizer = Tokenizer.from_file(str(tokenizer_files[0]))
    def count(text):
        return len(tokenizer.encode(SPECIAL.sub(r"<¦\1¦>", text), add_special_tokens=False).ids)
    DEST.mkdir(parents=True, exist_ok=False)
    kept, dropped, pilot = Counter(), Counter(), Counter()
    with (SOURCE / "train.jsonl").open(encoding="utf-8") as src, \
            (DEST / "train.jsonl").open("w", encoding="utf-8") as out, \
            (DEST / "pilot_train.jsonl").open("w", encoding="utf-8") as small:
        for line in src:
            row = json.loads(line)
            q = row["questions"]["destination"]
            state = count(row["state"]) + 1
            branch = 1 + count(q["instructions"]) + sum(2 + count(k + ": " + v) for k, v in q["criteria"].items()) + 1
            dataset = row["_meta"]["dataset"]
            if state > 1024 or state + branch > 1664 or state + branch > 2688:
                dropped[dataset] += 1
                continue
            kept[dataset] += 1
            out.write(line)
            if pilot[dataset] < 12:
                pilot[dataset] += 1
                small.write(line)
    if sum(kept.values()) != 9193 or sum(dropped.values()) != 545 or dropped["SWE-Bench"] != 398:
        raise ValueError(f"Unexpected admission counts: kept={kept}, dropped={dropped}")
    for name in ("val", "calibration", "development", "test_id", "test_ood"):
        shutil.copyfile(SOURCE / f"{name}.jsonl", DEST / f"{name}.jsonl")
    manifest["version"] = 3
    manifest["training_admission"] = {"tokenizer_sha256": next(iter(tokenizer_hashes)),
        "max_state": 1024, "max_branch": 1664, "max_packed": 2688,
        "original_records": 9738, "kept_by_dataset": dict(kept), "dropped_by_dataset": dict(dropped),
        "note": "Matches the input and Qwen-token admission used by the v1 0.8B-style run."}
    manifest["splits"]["train"] = {"records": sum(kept.values()), "sha256": digest(DEST / "train.jsonl"), "datasets": dict(kept)}
    manifest["splits"]["pilot_train"] = {"records": sum(pilot.values()), "sha256": digest(DEST / "pilot_train.jsonl"), "datasets": dict(pilot)}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Kept:", dict(kept), "total", sum(kept.values()), flush=True)
    print("Dropped:", dict(dropped), "total", sum(dropped.values()), flush=True)
    print("Pilot:", dict(pilot), flush=True)


if __name__ == "__main__":
    main()
