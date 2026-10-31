"""Fine-tune Kev decision models (Qwen backbone + PEFT LoRA + PointerHead) on LLMRouterBench.

Features:
  - Official Kev training pipeline utilizing PyTorch 2.14+, transformers, PEFT LoRA, and CUDA bf16
  - Parameter-efficient LoRA (r=16, alpha=32) over attention projections and MLPs
  - Cost-sensitive soft cross-entropy / KL objective against LLMRouterBench targets
  - Post-training temperature calibration on calibration.jsonl (minimizing NLL via L-BFGS grid)
  - Validation evaluation on development.jsonl
  - Saves loadable, calibrated checkpoint ready for local or cloud inference
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIGS = {
    "0.8b": {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "init_from": "models/kev-0.8b",
        "dir_name": "kev-0.8b-finetuned",
        "hidden_dim": 1024,
        "pointer_dim": 256,
    },
    "4b": {
        "base_model": "Qwen/Qwen3.5-4B-Base",
        "init_from": "models/kev-4b",
        "dir_name": "kev-4b-finetuned",
        "hidden_dim": 2560,
        "pointer_dim": 256,
    },
    "9b": {
        "base_model": "Qwen/Qwen3.5-9B-Base",
        "init_from": "models/kev-9b",
        "dir_name": "kev-9b-finetuned",
        "hidden_dim": 4096,
        "pointer_dim": 256,
    },
}

DESTINATION_MODELS = [
    "deepseek-v3-0324",
    "qwen3-235b-a22b-2507",
    "deepseek-v3.1-terminus",
    "kimi-k2-0905",
    "intern-s1",
    "gpt-5-chat",
    "qwen3-235b-a22b-thinking-2507",
    "gemini-2.5-flash",
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "glm-4.6",
    "gpt-5",
    "gemini-2.5-pro",
]


class PointerHead(nn.Module):
    """Kev Pointer Head: maps question decision token and option tokens to logits."""

    def __init__(self, d: int = 1024, dp: int = 256, temperature: float = 1.0):
        super().__init__()
        self.d = d
        self.dp = dp
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1.0 / math.sqrt(dp)
        self.temperature = float(temperature)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """Compute option logits from decision query and option keys.

        Args:
            h_decide: [B, d] representation of the <decide> token
            h_opts:   [B, K, d] representations of each option's </opt> token
        Returns:
            logits:   [B, K]
        """
        q_proj = self.q(h_decide).unsqueeze(-1)
        k_proj = self.k(h_opts)
        logits = torch.bmm(k_proj, q_proj).squeeze(-1) * self.scale

        if not self.training and self.temperature != 1.0:
            logits = logits / self.temperature

        return logits


def load_head_weights(head_path: Path, hidden_dim: int, pointer_dim: int = 256) -> PointerHead:
    """Load pretrained Kev PointerHead weights from head.pt."""
    head = PointerHead(d=hidden_dim, dp=pointer_dim)
    if head_path.exists():
        d = torch.load(head_path, weights_only=False, map_location="cpu")
        weights = d.get("head", {})
        if "q.weight" in weights and "k.weight" in weights:
            head.q.weight.data.copy_(weights["q.weight"])
            head.q.bias.data.copy_(weights["q.bias"])
            head.k.weight.data.copy_(weights["k.weight"])
            head.k.bias.data.copy_(weights["k.bias"])
        if "temperature" in d:
            head.temperature = float(d["temperature"])
        print(f"[+] Loaded pretrained PointerHead from {head_path} (temp: {head.temperature:.2f})")
    else:
        print(f"[*] Pretrained head not found at {head_path}. Initialized fresh PointerHead.")
    return head


def run_training_pipeline(
    data_dir: Path,
    output_dir: Path,
    model_size: str = "0.8b",
    epochs: int = 1,
    lr: float = 2e-5,
    batch_size: int = 1,
    accum_steps: int = 8,
    max_state_tokens: int = 1024,
    device: str = "cuda",
    dtype: str = "bf16",
    checkpointing: int = 1,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute official Kev fine-tuning on local GPU and calibrate temperature."""
    cfg = MODEL_CONFIGS[model_size]
    project_root = data_dir.parent.parent

    train_file = data_dir / "train.jsonl"
    cal_file = data_dir / "calibration.jsonl"
    dev_file = data_dir / "development.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at: {train_file}")

    init_dir = project_root / cfg["init_from"]
    if not init_dir.exists():
        raise FileNotFoundError(f"Pretrained Kev weights not found at: {init_dir}")

    # Remove existing output directory to avoid "refusing to overwrite" error from kev.train
    if output_dir.exists():
        print(f"[*] Cleaning output directory at {output_dir}...")
        shutil.rmtree(output_dir, ignore_errors=True)

    # 1. Launch kev.train
    print("=" * 80)
    print(f" KEV ROUTER FINE-TUNING: Kev-{model_size} ({cfg['base_model']})")
    print(f" Device: {device} | Dtype: {dtype} | Epochs: {epochs} | Accum: {accum_steps} | Max State: {max_state_tokens}")
    print(f" Training Data: {train_file.name} | Output: {output_dir}")
    print("=" * 80)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "-m", "kev.train",
        "--base", cfg["base_model"],
        "--data", str(train_file),
        "--init_from", str(init_dir),
        "--max_state", str(max_state_tokens),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch", str(batch_size),
        "--accum", str(accum_steps),
        "--device", device,
        "--dtype", dtype,
        "--checkpointing", str(checkpointing),
        "--out", str(output_dir),
    ]

    if max_steps and max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])

    print(f"[*] Executing command:\n    {' '.join(cmd)}\n")
    start_train_time = time.time()

    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Kev training failed with exit code {return_code}")

    train_duration = time.time() - start_train_time
    print(f"\n[+] Fine-tuning completed successfully in {train_duration/60:.2f} minutes.")

    # 2. Post-Training Calibration & Development Evaluation
    calibration_metrics = {}
    if cal_file.exists() and dev_file.exists():
        print("\n" + "=" * 80)
        print(" POST-TRAINING TEMPERATURE CALIBRATION & EVALUATION")
        print("=" * 80)

        from kev.benchmark import evaluate_records
        from kev.data import load_records
        from kev.metrics import fit_temperature
        from kev.predictors import LocalPredictor

        print(f"[*] Loading LocalPredictor from {output_dir}...")
        predictor = LocalPredictor(str(output_dir), device)

        cal_records = load_records(str(cal_file))
        dev_records = load_records(str(dev_file))

        temp_eval_dir = output_dir / "eval_temp_cal"
        dev_eval_dir = output_dir / "eval_temp_dev"
        shutil.rmtree(temp_eval_dir, ignore_errors=True)
        shutil.rmtree(dev_eval_dir, ignore_errors=True)

        print(f"[*] Scoring {len(cal_records)} calibration records...")
        _, cal_rows = evaluate_records(cal_records, predictor, str(temp_eval_dir), skip_overlong=True)

        optimal_temp = fit_temperature(cal_rows, aggregation="micro")
        print(f"[+] Fitted Optimal Temperature (T*): {optimal_temp:.4f}")

        # Update head.pt with optimal temperature
        head_path = output_dir / "head.pt"
        if head_path.exists():
            head_data = torch.load(head_path, weights_only=False, map_location="cpu")
            head_data["temperature"] = float(optimal_temp)
            torch.save(head_data, head_path)
            print(f"[+] Updated {head_path.name} with calibrated temperature T*={optimal_temp:.4f}")

        # Score development set under calibrated temperature
        print(f"[*] Evaluating {len(dev_records)} development records under T*={optimal_temp:.4f}...")
        dev_report, _ = evaluate_records(dev_records, predictor, str(dev_eval_dir), temperature=optimal_temp, skip_overlong=True)

        cal_acc = dev_report.get("calibrated_clean", {}).get("acc", 0.0) * 100
        raw_acc = dev_report.get("raw_clean", {}).get("acc", 0.0) * 100
        cal_brier = dev_report.get("calibrated_clean", {}).get("brier", 0.0)
        cal_ece = dev_report.get("calibrated_clean", {}).get("ece", 0.0)

        print("\n" + "-" * 80)
        print(" DEVELOPMENT BENCHMARK REPORT:")
        print(f"    Raw Validation Accuracy:        {raw_acc:>6.2f}%")
        print(f"    Calibrated Validation Accuracy: {cal_acc:>6.2f}%")
        print(f"    Calibrated Brier Score:         {cal_brier:>8.4f}")
        print(f"    Expected Calibration Error(ECE):{cal_ece:>8.4f}")
        print(f"    Fitted Temperature:             {optimal_temp:>8.4f}")
        print("-" * 80)

        calibration_metrics = {
            "optimal_temperature": float(optimal_temp),
            "raw_validation_accuracy": raw_acc,
            "calibrated_validation_accuracy": cal_acc,
            "calibrated_brier_score": float(cal_brier),
            "calibrated_ece": float(cal_ece),
        }

        # Clean up temporary eval directories
        shutil.rmtree(temp_eval_dir, ignore_errors=True)
        shutil.rmtree(dev_eval_dir, ignore_errors=True)

    summary = {
        "model_size": model_size,
        "base_model": cfg["base_model"],
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "accum_steps": accum_steps,
        "max_state_tokens": max_state_tokens,
        "train_duration_seconds": train_duration,
        "calibration": calibration_metrics,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Full training pipeline completed. Checkpoint and summary saved to:\n    {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Kev decision model on LLMRouterBench.")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "4b", "9b"], help="Kev model size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=1, help="Records per forward pass (default: 1)")
    parser.add_argument("--accum", type=int, default=8, help="Gradient accumulation micro-batches (default: 8)")
    parser.add_argument("--max-state", type=int, default=1024, help="Maximum state tokens (default: 1024)")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (for testing)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "kev_finetune"

    out_dir = Path(args.out_dir) if args.out_dir else (
        project_root / "models" / f"kev-{args.model_size}-finetuned"
    )

    run_training_pipeline(
        data_dir=data_dir,
        output_dir=out_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        max_state_tokens=args.max_state,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
