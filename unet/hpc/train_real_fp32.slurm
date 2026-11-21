#!/bin/bash -l

#SBATCH --job-name=UNet_FP32
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/unet/%x_%A_%a.out
#SBATCH --error=logs/unet/%x_%A_%a.err

set -eo pipefail

module purge
module load EasyBuild/V1
module load Anaconda3/2024.02-1
module load CUDA/11.7.0

: "${PROJECT:?Set PROJECT to the repository root}"
: "${UNET_ENV:?Set UNET_ENV to the Conda environment directory}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$UNET_ENV"

set -u

PYTHON="$UNET_ENV/bin/python"
CONFIG="${CONFIG:-unet/configs/reviewed_multimag_mixedclass_fp32_20261111T182044Z_e01af2.json}"
TASK="${SLURM_ARRAY_TASK_ID:?Submit this script as a job array}"

cd "$PROJECT"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Array: ${SLURM_ARRAY_JOB_ID}"
echo "Task: $TASK"
echo "Configuration: $CONFIG"

"$PYTHON" -u - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU available.")

print("GPU:", torch.cuda.get_device_name(0))
PY

PARAMETERS=$(
    "$PYTHON" - "$CONFIG" "$TASK" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
task = int(sys.argv[2])
seeds = cfg["seeds"]
fold_count = 5

if cfg.get("precision") != "float32":
    raise ValueError("This submission requires precision=float32.")

if not 0 <= task < len(seeds) * fold_count:
    raise ValueError(f"Invalid array task: {task}")

seed = seeds[task // fold_count]
fold = task % fold_count + 1
print(seed, fold)
PY
)

read -r SEED FOLD <<< "$PARAMETERS"

echo "Seed: $SEED"
echo "Fold: $FOLD"

"$PYTHON" -u unet/src/train.py \
    --config "$CONFIG" \
    --protocol real_only \
    --seed "$SEED" \
    --fold "$FOLD"

echo "Completed successfully: $(date)"
