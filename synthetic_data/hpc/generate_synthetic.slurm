#!/bin/bash
#SBATCH --job-name=SynGenerate
#SBATCH --partition=GPU_Compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/unet/SynGenerate_%j.out
#SBATCH --error=logs/unet/SynGenerate_%j.err
set -euo pipefail
: "${PROJECT:?Export PROJECT before sbatch}"
: "${SYNTH_OUT:?Export SYNTH_OUT before sbatch}"
PY="/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/.conda/envs/samenv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT"
"$PY" -u synthetic_data/src/generate_synthetic.py \
  --config synthetic_data/configs/synthetic_generation_v1.json \
  --out "$SYNTH_OUT" --workers 2
