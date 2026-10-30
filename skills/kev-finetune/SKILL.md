---
name: kev-finetune
description: Fine-tune a Kev decision model (open Jev-style System One model) on a user's own workload and serve it with calibrated probabilities. Interviews the user, finds existing Jev/TypeSafe questions or labelled data in their codebase, generates enough synthetic training data to measure a gain, trains and calibrates on Modal, scores against the released checkpoint, deploys a TypeSafe-compatible endpoint, and tears everything down. Use when someone wants noul/choice/score questions answered on their own domain (routing, triage, moderation, classification, scoring), wants calibrated confidence for automation thresholds, mentions Jev, TypeSafe, System One or Kev, or asks to fine-tune, evaluate, deploy or clean up Kev.
license: Apache-2.0
compatibility: Requires Python 3.10+, uv, and a Modal account (`uvx modal setup`). Training uses one H100 (about $1-3 per run). Local GPU execution supported with PyTorch/CUDA.
metadata:
  author: jaredpalmer
  version: "1.1"
  repository: https://github.com/jaredpalmer/kev
---

# Fine-tune Kev on the user's workload

Jev (TypeSafe's System One model) answers typed questions about a text without generating tokens, but it is a fixed
hosted model: on the user's data it is out of distribution and its probabilities cannot be recalibrated. Kev is the open
reconstruction; because it can be trained, you can fine-tune it on a few hundred to a few thousand labelled examples of
the user's exact questions and fit its temperature on a held-out slice, so the probabilities it serves are calibrated for
*their* data. This skill takes a user from "I have this decision to automate" to a deployed, measured endpoint, then
cleans up.

Scripts (`scripts/`) are short, standard-library Python plus Modal / local training apps:

| Script | Purpose |
| --- | --- |
| `extract_workload.py` | scan a codebase for Jev / TypeSafe / System One calls and labelled data; draft `workload.json` |
| `convert_data.py` | CSV / JSONL of existing labelled examples -> Kev records (column mapping) |
| `generate_data.py` | workload spec -> balanced labelled records via any OpenAI-compatible model; `--dry-run` prints the prompt |
| `plan_size.py` | how many records make the fine-tune vs baseline comparison statistically meaningful |
| `split_data.py` | validate records, split by state into train / calibration / development |
| `kev_modal.py` | Modal: `validate`, `train`, `evaluate`, `compare`, `pull`, `publish`, `teardown`, and the `Serve` endpoint |
| `train_local.py` | Local / cluster GPU runner: fine-tune Kev backbone + pointer head using PyTorch & PEFT |

## Workflow Phases

### Phase 0: Workload Definition
Describe the decision in `workload.json`: the domain, the input `state`, and the typed `questions` (`choice`, `noul`, or `score`).
For KevRouter, the question is `destination`: selecting among the 13 destination LLMs.

### Phase 1: Data Conversion & Partitioning
Convert source data records into Kev's JSONL format (`{"state": "...", "questions": {"...": {"type": "...", "instructions": "...", "criteria": {...}, "label": "..."}}}`).
Run `split_data.py` to ensure zero state leakage between partitions:
- `train.jsonl`
- `calibration.jsonl` (for temperature calibration)
- `development.jsonl` (for validation & hyperparameter comparison)

### Phase 2: Sizing & Power Analysis
Run `plan_size.py workload.json` to calculate the statistical sample size needed to detect significant gains over the base model.

### Phase 3: Fine-Tuning
Execute fine-tuning on Modal (`modal run scripts/kev_modal.py::train ...`) or locally via GPU (`python scripts/train_local.py ...`).

### Phase 4: Temperature Calibration & Evaluation
Fit temperature $T$ on `calibration.jsonl` by minimizing NLL, then evaluate calibrated accuracy, Brier score, ECE, and routing metrics on `development.jsonl` and held-out test sets.
