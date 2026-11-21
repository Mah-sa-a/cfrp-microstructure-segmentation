#!/bin/bash -l

#SBATCH --job-name=prepare_real_cv
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mahsa.arabkoohichetabi@parisnanterre.fr
#SBATCH --output=slurm_prepare_%j.out
#SBATCH --error=slurm_prepare_%j.err

set -eo pipefail

module purge
module load EasyBuild/V1
module load Anaconda3/2024.02-1
module load CUDA/11.7.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate samenv

PROJECT="/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/Computations/CFRP-microstructure-segmentation"
cd "$PROJECT"

SCRIPT="unet/src/prepare_real_dataset.py"
CONFIG="unet/configs/dataset_preparation.json"
METADATA="data/metadata/sample_groups.csv"

for FILE in "$SCRIPT" "$CONFIG" "$METADATA"; do
    if [[ ! -f "$FILE" ]]; then
        echo "ERROR: missing required file: $PROJECT/$FILE" >&2
        exit 1
    fi
done

# Default: exploratory preparation while annotation review is unfinished.
# For reviewed-only preparation, submit with:
# sbatch --export=ALL,REVIEW_MODE=reviewed unet/hpc/prepare_real_dataset.slurm
REVIEW_MODE="${REVIEW_MODE:-exploratory}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

case "$REVIEW_MODE" in
    exploratory)
        RUN_NAME="exploratory_5fold_${RUN_ID}"
        FLAGS=(--allow-unreviewed)
        ;;
    reviewed)
        RUN_NAME="real_5fold_${RUN_ID}"
        FLAGS=()
        ;;
    *)
        echo "ERROR: REVIEW_MODE must be exploratory or reviewed" >&2
        exit 1
        ;;
esac

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Project: $PROJECT"
echo "Review mode: $REVIEW_MODE"
echo "Run name: $RUN_NAME"
python --version

python -u "$SCRIPT" --config "$CONFIG" --run-name "$RUN_NAME" "${FLAGS[@]}"

echo "Completed: $(date)"
echo "Manifests: $PROJECT/data/manifests/$RUN_NAME"
echo "Splits: $PROJECT/data/splits/$RUN_NAME"
