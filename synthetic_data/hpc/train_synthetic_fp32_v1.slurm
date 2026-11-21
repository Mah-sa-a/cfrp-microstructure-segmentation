#!/bin/bash
#SBATCH --job-name=UNet_Synth
#SBATCH --partition=GPU_Compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/unet/UNet_Synth_%A_%a.out
#SBATCH --error=logs/unet/UNet_Synth_%A_%a.err

set -euo pipefail

: "${PROJECT:?Export PROJECT before submitting}"
: "${CONFIG:?Export CONFIG before submitting}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a job array}"

module load EasyBuild/V1
module load Anaconda3/2024.02-1
module load CUDA/11.7.0

PY="/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/.conda/envs/samenv/bin/python"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$PROJECT"

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "Job: ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
echo "Seed: $SEED"
echo "Configuration: $CONFIG"

"$PY" -u unet/src/train.py \
    --protocol synthetic_pretrain \
    --seed "$SEED" \
    --fold 1 \
    --config "$CONFIG"
