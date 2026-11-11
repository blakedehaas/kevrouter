#!/usr/bin/env bash
#SBATCH --job-name=kev-4b-9b
#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=78G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=outputs/experiments/slurm-%A_%a.out

set -euo pipefail
cd /projects/abdu6273/kevrouter
: "${KEV_RUN_ROOT:?KEV_RUN_ROOT must be set at submission}"
export HF_HOME="$PWD/models/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TIKTOKEN_CACHE_DIR="$PWD/data/.cache/tiktoken"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

case "$SLURM_ARRAY_TASK_ID" in
  0) size=4b; variant=base ;;
  1) size=4b; variant=finetuned ;;
  2) size=9b; variant=base ;;
  3) size=9b; variant=finetuned ;;
  *) echo "Unexpected array task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "Job $SLURM_JOB_ID: Kev $size $variant on $(hostname)"
nvidia-smi
exec "$PWD/kevrouter_venv/bin/python" -u scripts/run_kev_experiment.py \
  --size "$size" --variant "$variant" --output-dir "$KEV_RUN_ROOT/kev-$size-$variant"
