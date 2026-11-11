#!/usr/bin/env python3
"""Isolated 4B/9B Kev experiment with pilot, calibration, and policy outputs."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("KEV_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data/experiments/kev_4b_9b_v2"
MODELS = {"4b": ("4B", "1001bb4d826a52d1f399e183466143f4da7b741b"),
          "9b": ("9B", "68c46c4b3498877f3ef123c856ecfde50c39f404")}
CONTEXT = {"max_state": 2048, "max_branch": 2048, "max_packed": 4096}
COSTS = {"deepseek-v3-0324": .0020, "gpt-5-chat": .0022,
         "deepseek-v3.1-terminus": .0038, "qwen3-235b-a22b-2507": .0071,
         "intern-s1": .0095, "kimi-k2-0905": .0125, "claude-sonnet-4": .0158,
         "gemini-2.5-flash": .0195, "qwen3-235b-a22b-thinking-2507": .0210,
         "deepseek-r1-0528": .0485, "glm-4.6": .0540, "gpt-5": .1520,
         "gemini-2.5-pro": .2450}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_data():
    run_root = os.environ.get("KEV_RUN_ROOT")
    if run_root:
        pinned = json.loads((Path(run_root) / "code/input_hashes.json").read_text())
        for relative, expected in pinned.items():
            path = ROOT / relative
            if digest(path) != expected:
                raise ValueError(f"Experiment input changed since submission: {path}")
    manifest = json.loads((DATA / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        path = DATA / f"{split}.jsonl"
        if digest(path) != info["sha256"]:
            raise ValueError(f"Prepared data hash mismatch: {path}")
    return manifest


def fit_soft_temperature(rows, records):
    """Fit positive temperature to the paper's soft targets using L-BFGS."""
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    if not rows or len(rows) != len(records):
        raise ValueError("Calibration prediction count does not match records")
    logits, targets = [], []
    for row, record in zip(rows, records):
        if row["id"] != record["_meta"]["id"] or row.get("inference_temperature") != 1.0:
            raise ValueError("Calibration rows are misaligned or not raw logits")
        target = record["questions"]["destination"]["target"]
        vector = np.array([target.get(k, 0.0) for k in row["keys"]], dtype=np.float64)
        if vector.sum() <= 0:
            raise ValueError("Calibration target has no probability mass")
        logits.append(row["logits"])
        targets.append(vector / vector.sum())
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite calibration logits")

    def objective(x):
        scaled = logits / np.exp(x[0])
        return float(np.mean(logsumexp(scaled, axis=1) - np.sum(targets * scaled, axis=1)))

    fit = minimize(objective, [0.0], method="L-BFGS-B", bounds=[(-5.0, 5.0)])
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Temperature fit failed: {fit.message}")
    return float(np.exp(fit.x[0])), {"loss": float(fit.fun), "method": "soft-target NLL / L-BFGS-B",
                                    "log_temperature_bounds": [-5.0, 5.0]}


def policy_choices(probabilities, available):
    probs = {m: float(probabilities[m]) for m in available}
    if any(not math.isfinite(p) or p < 0 for p in probs.values()) or sum(probs.values()) <= 0:
        raise ValueError("Invalid candidate probabilities")
    total = sum(probs.values())
    probs = {m: p / total for m, p in probs.items()}
    if set(probs) - COSTS.keys():
        raise ValueError("Cost estimates missing for an eligible destination")
    best = max(probs, key=probs.get)
    utility = max(probs, key=lambda m: probs[m] - 2.5 * COSTS[m])
    cheap = [m for m in probs if COSTS[m] <= .010]
    flagship = [m for m in probs if COSTS[m] > .010]
    best_cheap = max(cheap, key=probs.get) if cheap else None
    if best_cheap is not None and (probs[best_cheap] >= .6 or not flagship):
        cascade = best_cheap
    else:
        cascade = max(flagship, key=probs.get)
    return {"argmax": best, "cost_utility_mu_2.5": utility, "paper_tier_gate_tau_0.6": cascade}, probs


def score_split(predictor, name, output, limit=None):
    from kev.data import load_records
    records = load_records(DATA / f"{name}.jsonl")
    if limit:
        records = records[:limit]
    raw_name = "val" if name == "development" else name
    with (ROOT / "data/processed" / f"{raw_name}.jsonl").open(encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]
    stats = {p: defaultdict(lambda: {"total": 0, "correct": 0, "optimal": 0, "cost": 0.0})
             for p in ("argmax", "cost_utility_mu_2.5", "paper_tier_gate_tau_0.6")}
    forward, end_to_end = [], []
    with (output / f"{name}_predictions.jsonl").open("x", encoding="utf-8") as predictions:
        for i, record in enumerate(records):
            meta = record["_meta"]
            source = raw[meta["processed_row"]]
            if (source["query_hash"], source["dataset"], source["index"]) != (
                    meta["query_hash"], meta["dataset"], meta["index"]):
                raise ValueError(f"Processed and Kev rows differ in {name}, row {i}")
            start = time.perf_counter()
            result = predictor(record)
            elapsed = (time.perf_counter() - start) * 1000
            available = source["model_evals"]
            probs = result["probabilities"]["destination"]
            if not available or any(m not in probs for m in available):
                raise ValueError(f"Missing evaluated model probabilities in {name}, row {i}")
            choices, masked = policy_choices(probs, available)
            entry = {"dataset": source["dataset"], "index": source["index"],
                     "processed_row": meta["processed_row"], "eligible_probabilities": masked,
                     "logits": result["logits"]["destination"], "temperature": predictor.temperature,
                     "forward_ms": float(result["latency_ms"]), "predictor_call_ms": elapsed,
                     "policies": {}}
            for policy, model in choices.items():
                ev = available[model]
                cost = float(ev["cost"])
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("Invalid recorded destination cost")
                correct = float(ev["score"]) >= 1.0
                optimal = correct and model == source["optimal_model"]
                group = stats[policy][source["dataset"]]
                group["total"] += 1; group["correct"] += correct
                group["optimal"] += optimal; group["cost"] += cost
                entry["policies"][policy] = {"chosen": model, "correct": correct, "cost": cost, "optimal": optimal}
            predictions.write(json.dumps(entry, allow_nan=False) + "\n")
            forward.append(float(result["latency_ms"])); end_to_end.append(elapsed)
            if (i + 1) % 50 == 0:
                predictions.flush(); print(f"{name}: {i+1}/{len(records)}", flush=True)
    n = len(records)
    summaries = {}
    for policy, groups in stats.items():
        correct = sum(g["correct"] for g in groups.values())
        optimal = sum(g["optimal"] for g in groups.values())
        cost = sum(g["cost"] for g in groups.values())
        summaries[policy] = {"total_queries": n, "routing_accuracy": 100*correct/n,
                             "optimal_match_rate": 100*optimal/n, "avg_cost_per_query_usd": cost/n,
                             "cost_per_correct_usd": cost/correct if correct else None,
                             "by_benchmark": dict(groups)}
    summary = {"policies": summaries, "latency": {"mean_forward_ms": sum(forward)/n,
               "mean_predictor_call_ms": sum(end_to_end)/n},
               "policy_note": "Tier gate follows paper's flagship-only fallback; v1 used overall argmax."}
    (output / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{name}: {n} records; argmax accuracy={summaries['argmax']['routing_accuracy']:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(MODELS))
    parser.add_argument("--variant", required=True, choices=["base", "finetuned"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    if args.pilot and args.variant != "finetuned":
        raise ValueError("Pilot must exercise fine-tuning")
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = verify_data()
    import torch
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.data import load_records
    from kev.benchmark import evaluate_records
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    upper, revision = MODELS[args.size]
    checkpoint = ROOT / "models" / f"kev-{args.size}"
    meta = {"size": args.size, "variant": args.variant, "pilot": args.pilot,
            "job_id": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "base_revision": revision, "runner_sha256": digest(Path(__file__)),
            "data_manifest": manifest, "eval_dtype": "bfloat16", "context": CONTEXT}
    if args.variant == "finetuned":
        trained = out / "checkpoint"
        train_file = DATA / ("pilot_train.jsonl" if args.pilot else "train.jsonl")
        command = [sys.executable, "-m", "kev.train", "--base", f"Qwen/Qwen3.5-{upper}-Base",
                   "--base_revision", revision, "--data", str(train_file), "--init_from", str(checkpoint),
                   "--epochs", str(args.epochs), "--lr", str(args.lr), "--lora", "16", "--batch", "1",
                   "--accum", "8", "--max_state", "1024", "--device", "cuda", "--dtype", "bf16",
                   "--weights_dtype", "bf16", "--checkpointing", "1", "--out", str(trained)]
        if args.pilot: command += ["--max_steps", "5"]
        meta["training_command"] = command
        (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        start = time.time(); print("Training:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        meta["training_seconds"] = time.time() - start; checkpoint = trained
    opts = LoadOptions(dtype=torch.bfloat16, temperature=1.0 if args.variant == "finetuned" else None)
    predictor = LocalPredictor(str(checkpoint), "cuda", opts=opts, context=CONTEXT)
    meta["predictor_environment"] = predictor.environment
    if args.variant == "finetuned":
        cal = load_records(DATA / "calibration.jsonl")
        if args.pilot: cal = cal[:5]
        _, rows = evaluate_records(cal, predictor, out / "calibration", skip_overlong=False)
        temperature, fit = fit_soft_temperature(rows, cal)
        head_file = checkpoint / "head.pt"
        head = torch.load(head_file, map_location="cpu", weights_only=False)
        head["temperature"] = temperature
        tmp = checkpoint / "head.pt.tmp"; torch.save(head, tmp); os.replace(tmp, head_file)
        predictor.model.head.temperature = temperature; predictor.temperature = temperature
        meta["calibration_temperature"] = temperature; meta["calibration_fit"] = fit
        meta["calibrated_head_sha256"] = digest(head_file)
        print(f"Calibrated checkpoint temperature: {temperature:.6f}", flush=True)
    else:
        meta["calibration_temperature"] = predictor.temperature
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    for split in ("development", "test_id", "test_ood"):
        score_split(predictor, split, out, limit=5 if args.pilot else None)
    if args.pilot:
        training = json.loads((checkpoint / "training_metrics.json").read_text())
        seen = float(training["records_seen"])
        if seen <= 0:
            raise ValueError("Pilot did not report any training records")
        estimated = float(training["wall_seconds"]) / seen * manifest["splits"]["train"]["records"] * args.epochs
        pilot = {"records_seen": seen, "pilot_training_seconds": training["wall_seconds"],
                 "peak_device_bytes": training["peak_device_bytes"],
                 "estimated_full_training_hours": estimated / 3600,
                 "note": "Rough estimate from five optimizer steps; evaluation and setup are additional."}
        (out / "pilot_estimate.json").write_text(json.dumps(pilot, indent=2) + "\n")
        print("Pilot estimate:", json.dumps(pilot), flush=True)
        if estimated > 20 * 3600:
            raise RuntimeError("Pilot projects over 20 hours of training; full jobs need review")
    (out / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
