#!/usr/bin/env python3
"""Train or evaluate one Kev size without substituting ground-truth answers."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "experiments" / "kev_4b_9b_v1"
MODELS = {"4b": ("4B", "1001bb4d826a52d1f399e183466143f4da7b741b"),
          "9b": ("9B", "68c46c4b3498877f3ef123c856ecfde50c39f404")}
CONTEXT = {"max_state": 2048, "max_branch": 2048, "max_packed": 4096}


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_split(predictor, name, output):
    from kev.data import load_records

    records = load_records(DATA / f"{name}.jsonl")
    raw_name = "val" if name == "development" else name
    raw = load_jsonl(ROOT / "data" / "processed" / f"{raw_name}.jsonl")
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "optimal": 0, "cost": 0.0})
    total_cost, correct, optimal, latencies = 0.0, 0, 0, []
    path = output / f"{name}_predictions.jsonl"
    with path.open("x", encoding="utf-8") as predictions:
        for i, record in enumerate(records):
            meta = record["_meta"]
            source = raw[meta["processed_row"]]
            if (source["query_hash"], source["dataset"], source["index"]) != (
                    meta["query_hash"], meta["dataset"], meta["index"]):
                raise ValueError(f"Processed and Kev rows differ in {name} at {i}")
            result = predictor(record)
            probabilities = result["probabilities"]["destination"]
            available = source["model_evals"]
            if not available or any(m not in probabilities for m in available):
                raise ValueError(f"Missing evaluated candidate probabilities in {name} at {i}")
            candidates = {m: probabilities[m] for m in available}
            if any(not math.isfinite(float(p)) or p < 0 for p in candidates.values()) or sum(candidates.values()) <= 0:
                raise ValueError(f"Invalid model probabilities in {name} at {i}")
            chosen = max(candidates, key=candidates.get)
            evaluation = available[chosen]
            cost = float(evaluation["cost"])
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid cost in {name} at {i}")
            is_correct = float(evaluation["score"]) >= 1.0
            is_optimal = is_correct and chosen == source["optimal_model"]
            latency = float(result["latency_ms"])
            correct += is_correct
            optimal += is_optimal
            total_cost += cost
            latencies.append(latency)
            group = stats[source["dataset"]]
            group["total"] += 1
            group["correct"] += is_correct
            group["optimal"] += is_optimal
            group["cost"] += cost
            predictions.write(json.dumps({"dataset": source["dataset"], "index": source["index"],
                                          "processed_row": meta["processed_row"], "chosen": chosen,
                                          "optimal": source["optimal_model"], "correct": is_correct,
                                          "cost": cost, "latency_ms": latency}) + "\n")
            if (i + 1) % 50 == 0:
                predictions.flush()
                print(f"{name}: scored {i + 1}/{len(records)}", flush=True)
    n = len(records)
    summary = {"total_queries": n, "correct_routes": correct,
               "routing_accuracy": 100 * correct / n,
               "optimal_match_rate": 100 * optimal / n,
               "avg_cost_per_query_usd": total_cost / n,
               "cost_per_correct_usd": total_cost / correct if correct else None,
               "avg_latency_ms": sum(latencies) / n, "by_benchmark": dict(stats)}
    (output / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: accuracy={summary['routing_accuracy']:.2f}% avg_cost=${summary['avg_cost_per_query_usd']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(MODELS))
    parser.add_argument("--variant", required=True, choices=["base", "finetuned"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not (DATA / "manifest.json").is_file():
        raise FileNotFoundError(f"Prepared data manifest missing: {DATA}")
    import torch
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.data import load_records
    from kev.benchmark import evaluate_records
    from kev.metrics import fit_temperature

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment")
    checkpoint = ROOT / "models" / f"kev-{args.size}"
    upper_size, revision = MODELS[args.size]
    meta = {"size": args.size, "variant": args.variant, "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "base_revision": revision,
            "data_manifest": json.loads((DATA / "manifest.json").read_text()),
            "training_epochs": 1 if args.variant == "finetuned" else 0,
            "eval_dtype": "bfloat16", "context": CONTEXT}
    if args.variant == "finetuned":
        trained = out / "checkpoint"
        command = [sys.executable, "-m", "kev.train", "--base", f"Qwen/Qwen3.5-{upper_size}-Base",
                   "--base_revision", revision, "--data", str(DATA / "train.jsonl"),
                   "--init_from", str(checkpoint), "--epochs", "1", "--lr", "2e-5",
                   "--lora", "16", "--batch", "1", "--accum", "8", "--max_state", "1024",
                   "--device", "cuda", "--dtype", "bf16", "--weights_dtype", "bf16",
                   "--checkpointing", "1", "--out", str(trained)]
        meta["training_command"] = command
        print("Training command:", " ".join(command), flush=True)
        started = time.time()
        subprocess.run(command, cwd=ROOT, check=True)
        meta["training_seconds"] = time.time() - started
        checkpoint = trained
    opts = LoadOptions(dtype=torch.bfloat16, temperature=1.0 if args.variant == "finetuned" else None)
    predictor = LocalPredictor(str(checkpoint), "cuda", opts=opts, context=CONTEXT)
    meta["predictor_environment"] = predictor.environment
    if args.variant == "finetuned":
        cal = load_records(DATA / "calibration.jsonl")
        _, rows = evaluate_records(cal, predictor, out / "calibration", skip_overlong=False)
        temperature = float(fit_temperature(rows, aggregation="micro"))
        predictor.model.head.temperature = temperature
        predictor.temperature = temperature
        meta["calibration_temperature"] = temperature
        print(f"Calibration temperature: {temperature}", flush=True)
    else:
        meta["calibration_temperature"] = predictor.temperature
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    for split in ("development", "test_id", "test_ood"):
        score_split(predictor, split, out)
    (out / "COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
