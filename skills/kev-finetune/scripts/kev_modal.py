"""Modal fine-tuning, calibration, evaluation, and serving runner for Kev models.

Official Modal recipe from jaredpalmer/kev (skills/kev-finetune):
    modal run skills/kev-finetune/scripts/kev_modal.py::train --data data/kev_finetune --name kevrouter-v1 --init-from jaredpalmer/kev-0.8b
    modal run skills/kev-finetune/scripts/kev_modal.py::evaluate --run kevrouter-v1 --data data/kev_finetune
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

SETTINGS = {
    "KEV_APP_NAME": "kev-finetune-llmrouter",
    "KEV_REF": "fc4a17e194bca35a48c57211392ec49ea54a94ac",
    "KEV_SERVE_RUN": "jaredpalmer/kev-0.8b",
}
SETTINGS = {k: os.environ.get(k, v) for k, v in SETTINGS.items()}
APP_NAME, KEV_REF, SERVE_RUN = SETTINGS["KEV_APP_NAME"], SETTINGS["KEV_REF"], SETTINGS["KEV_SERVE_RUN"]
KEV_REPO = "https://github.com/jaredpalmer/kev.git"
KEV_ROOT = "/kev"
RUNS, HF = "/runs", "/hf"
GPU = os.environ.get("KEV_GPU", "H100")
SERVE_GPU = os.environ.get("KEV_SERVE_GPU", "L4")

if HAS_MODAL:
    app = modal.App(APP_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .run_commands(f"git clone {KEV_REPO} {KEV_ROOT} && git -C {KEV_ROOT} checkout --quiet {KEV_REF}")
        .uv_pip_install(f"kev[serve] @ file://{KEV_ROOT}")
        .uv_pip_install("peft", "transformers", "accelerate", "torch")
        .env({"HF_HOME": HF, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    )
    runs = modal.Volume.from_name("kev-finetune-runs", create_if_missing=True)
    hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
    VOLUMES = {RUNS: runs, HF: hf_cache}

    @app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=7200)
    def train(data_dir: str, name: str, init_from: str = "jaredpalmer/kev-0.8b", epochs: int = 2, lr: float = 2e-5):
        """Remote fine-tuning on Modal GPU."""
        out_dir = Path(RUNS) / name / "checkpoint"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python", "-m", "kev.train",
            "--data", f"{data_dir}/train.jsonl",
            "--init_from", init_from,
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--batch", "1",
            "--accum", "8",
            "--dtype", "bf16",
            "--checkpointing", "1",
            "--device", "cuda",
            "--out", str(out_dir),
        ]
        print(f"[*] Running remote command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        runs.commit()
        print(f"[+] Remote training completed. Checkpoint committed to volume: {out_dir}")

    @app.function(image=image, gpu=SERVE_GPU, volumes=VOLUMES)
    def evaluate(run_name: str, data_dir: str):
        """Remote evaluation and calibration on Modal."""
        from kev.benchmark import evaluate_records
        from kev.data import load_records
        from kev.metrics import fit_temperature
        from kev.predictors import LocalPredictor

        ckpt_path = Path(RUNS) / run_name / "checkpoint"
        predictor = LocalPredictor(str(ckpt_path), "cuda")

        cal_records = load_records(f"{data_dir}/calibration.jsonl")
        dev_records = load_records(f"{data_dir}/development.jsonl")

        print(f"[*] Fitting temperature on {len(cal_records)} calibration records...")
        _, cal_rows = evaluate_records(cal_records, predictor)
        temperature = fit_temperature(cal_rows, aggregation="micro")
        print(f"[+] Fitted temperature: {temperature:.3f}")

        print(f"[*] Evaluating on {len(dev_records)} development records...")
        report, _ = evaluate_records(dev_records, predictor, temperature=temperature)
        print(f"[+] Calibrated Accuracy: {report['calibrated_clean']['acc']:.3f}")
        return report
else:
    def train(*args, **kwargs):
        raise ImportError("Modal is not installed. Install via 'pip install modal' or run locally via train_local.py")
