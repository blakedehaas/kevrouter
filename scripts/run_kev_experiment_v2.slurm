#!/usr/bin/env bash
#SBATCH --job-name=kev-4b-9b-v2
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
: "${KEV_RUN_ROOT:?}" "${KEV_STAGE:?}" "${KEV_EPOCHS:?}" "${KEV_LR:?}"
export KEV_PROJECT_ROOT="$PWD"
export HF_HOME="$PWD/models/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TIKTOKEN_CACHE_DIR="$PWD/data/.cache/tiktoken"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

extra=()
if [[ "$KEV_STAGE" == pilot ]]; then
  case "$SLURM_ARRAY_TASK_ID" in
    0) size=4b ;;
    1) size=9b ;;
    *) exit 1 ;;
  esac
  variant=finetuned
  extra=(--pilot)
elif [[ "$KEV_STAGE" == full ]]; then
  case "$SLURM_ARRAY_TASK_ID" in
    0) size=4b; variant=base ;;
    1) size=4b; variant=finetuned ;;
    2) size=9b; variant=base ;;
    3) size=9b; variant=finetuned ;;
    *) exit 1 ;;
  esac
else
  echo "Unexpected stage: $KEV_STAGE" >&2; exit 1
fi

echo "Job $SLURM_JOB_ID: $KEV_STAGE Kev $size $variant on $(hostname)"
nvidia-smi
exec "$PWD/kevrouter_venv/bin/python" -u "$KEV_RUN_ROOT/code/run_kev_experiment_v2.py" \
  --size "$size" --variant "$variant" --epochs "$KEV_EPOCHS" --lr "$KEV_LR" \
  --output-dir "$KEV_RUN_ROOT/$KEV_STAGE/kev-$size-$variant" "${extra[@]}"
