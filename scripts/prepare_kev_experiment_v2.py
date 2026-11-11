#!/usr/bin/env python3
"""Prepare a separate training set whose Qwen token lengths fit Kev's context."""
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/experiments/kev_4b_9b_v1"
DEST = ROOT / "data/experiments/kev_4b_9b_v2"
TOKENIZER = ROOT / "models/kev-4b/tokenizer.json"
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def qwen_text(text):
    return SPECIAL.sub(r"<¦\1¦>", text)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, meta in manifest["splits"].items():
        path = SOURCE / f"{name}.jsonl"
        if digest(path) != meta["sha256"]:
            raise ValueError(f"Input file has changed: {path}")
    other = ROOT / "models/kev-9b/tokenizer.json"
    if digest(TOKENIZER) != digest(other):
        raise ValueError("4B and 9B tokenizers differ; prepare separate training sets")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    DEST.mkdir(parents=True, exist_ok=False)
    changed = Counter()
    pilot_counts = Counter()
    train_path = DEST / "train.jsonl"
    pilot_path = DEST / "pilot_train.jsonl"
    with (SOURCE / "train.jsonl").open(encoding="utf-8") as src, \
            train_path.open("w", encoding="utf-8") as train, \
            pilot_path.open("w", encoding="utf-8") as pilot:
        for line in src:
            row = json.loads(line)
            text = qwen_text(row["state"])
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) > 1023:
                keep = 1023
                while True:
                    shortened = tokenizer.decode(ids[:keep], skip_special_tokens=False)
                    if len(tokenizer.encode(qwen_text(shortened), add_special_tokens=False).ids) <= 1023:
                        break
                    keep -= 1
                row["state"] = shortened
                changed[row["_meta"]["dataset"]] += 1
            actual = len(tokenizer.encode(qwen_text(row["state"]), add_special_tokens=False).ids) + 1
            if actual > 1024:
                raise ValueError(f"Qwen state exceeds limit: {actual}")
            text = json.dumps(row, ensure_ascii=False) + "\n"
            train.write(text)
            dataset = row["_meta"]["dataset"]
            if pilot_counts[dataset] < 12:
                pilot.write(text)
                pilot_counts[dataset] += 1
    for name in ("val", "calibration", "development", "test_id", "test_ood"):
        shutil.copyfile(SOURCE / f"{name}.jsonl", DEST / f"{name}.jsonl")
    manifest["version"] = 2
    manifest["training_tokenizer"] = {"name": "Qwen3.5", "sha256": digest(TOKENIZER),
                                      "max_state_tokens_including_delimiter": 1024,
                                      "truncated_by_dataset": dict(changed)}
    manifest["splits"]["train"]["sha256"] = digest(train_path)
    manifest["splits"]["pilot_train"] = {"records": sum(pilot_counts.values()),
                                         "sha256": digest(pilot_path), "datasets": dict(pilot_counts)}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Qwen-truncated training records:", dict(changed), flush=True)
    print("Pilot training records:", dict(pilot_counts), flush=True)
    print("Prepared:", DEST, flush=True)


if __name__ == "__main__":
    main()
