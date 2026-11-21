#!/bin/bash
#SBATCH --job-name=UNet_SynEval
#SBATCH --partition=GPU_Compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/unet/UNet_SynEval_%A_%a.out
#SBATCH --error=logs/unet/UNet_SynEval_%A_%a.err

set -euo pipefail

: "${PROJECT:?Export PROJECT before submitting}"
: "${CONFIG:?Export CONFIG before submitting}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array}"

module load EasyBuild/V1
module load Anaconda3/2024.02-1
module load CUDA/11.7.0

PY="/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/.conda/envs/samenv/bin/python"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$PROJECT"

TASK="$SLURM_ARRAY_TASK_ID"
if (( TASK < 0 || TASK > 14 )); then
    echo "Invalid task index: $TASK"
    exit 1
fi

SEEDS=(42 43 44)
SEED="${SEEDS[$((TASK / 5))]}"
FOLD="$((TASK % 5 + 1))"

echo "Evaluation only: seed=$SEED fold=$FOLD"
echo "Configuration: $CONFIG"

"$PY" -u unet/src/train.py \
    --protocol synthetic_only \
    --seed "$SEED" \
    --fold "$FOLD" \
    --config "$CONFIG"

echo "Synthetic-only evaluation finished: seed=$SEED fold=$FOLD"
