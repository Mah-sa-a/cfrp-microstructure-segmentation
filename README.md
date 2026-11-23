# CFRP Microstructure Segmentation

### Synthetic-to-real deep learning for three-phase segmentation of CFRP optical micrographs

Research software for segmenting carbon fiber-reinforced polymer (CFRP)
optical micrographs into matrix, fiber, and void regions using a
PyTorch-based U-Net.

This repository combines parameterized synthetic-image generation,
SAM-assisted experimental annotation, and a controlled comparison of
synthetic and experimental training strategies.

## Overview

Pixel-wise annotation of experimental micrographs is labor-intensive.
Synthetic microstructures provide automatically generated labels, but
differences between synthetic and experimental images can affect model
transfer.

This study investigates two related questions:

- How effectively does a model trained on synthetic images transfer
  directly to experimental CFRP micrographs?
- Does synthetic pretraining improve segmentation relative to training
  on experimental images alone?

Three training strategies are evaluated using the same U-Net architecture
and identical held-out experimental samples:

| Strategy | Training procedure | Protocol |
|---|---|---|
| Experimental-only | Train from scratch on experimental image–mask pairs | `real_only` |
| Synthetic-only | Train on synthetic pairs, then evaluate directly on experimental images | `synthetic_pretrain`, then `synthetic_only` |
| Synthetic pretraining + experimental fine-tuning | Initialize from synthetic-trained weights and adapt using experimental pairs | `synthetic_pretrain`, then `synthetic_to_real` |

## Workflow

1. Generate synthetic microstructures and their corresponding phase masks.
2. Prepare experimental reference masks using SAM-assisted proposals,
   intensity-based classification, morphological criteria, and visual
   quality screening.
3. Prepare image–mask patches and sample-wise experimental partitions.
4. Train and evaluate the three strategies.
5. Compare class-wise overlap metrics and inspect localized segmentation
   differences.

### Semantic classes

| Class ID | Phase |
|---|---|
| 0 | Matrix |
| 1 | Fiber |
| 2 | Void |

Network inputs are 512 × 512-pixel grayscale patches. The network produces
three output-logit channels at the same spatial resolution.

## Evaluation design

Experimental performance is assessed using five-fold leave-one-sample-out
evaluation across five physical samples and three random seeds:
42, 43, and 44.

Within each experimental fold:

- Three samples provide training data.
- One separate sample provides validation data for checkpoint selection.
- One sample is held out for testing.

All patches from a physical sample remain within the same partition.
Experimental-only training and experimental fine-tuning use the same
training, validation, and test assignments.

Fine-tuning uses the full experimental training partition available
within each fold. It does not use validation or test samples for
weight updates.

Synthetic pretraining and synthetic-only evaluation are separate stages.
The experimental evaluation partition must not be used to select the
synthetic pretrained checkpoint.

## Principal results

The following IoUs are percentages, averaged equally across the five
sample-wise folds and three random seeds.

| Training strategy | Matrix IoU | Fiber IoU | Void IoU | Macro IoU |
|---|---:|---:|---:|---:|
| Synthetic-only | 79.27 | 86.10 | 85.90 | 83.75 |
| Synthetic pretraining + experimental fine-tuning | 92.65 | 95.42 | 96.26 | 94.78 |
| Experimental-only | 94.28 | 96.44 | 97.31 | 96.01 |

Experimental fine-tuning improved reference agreement in every matched
seed–fold comparison, with a mean macro-IoU increase of 11.02 percentage
points over synthetic-only training. The largest class-wise improvement
occurred for matrix.

Experimental-only training retained the highest reference agreement.
The fine-tuned model remained below the experimental-only model by a
mean of 1.23 percentage points in matched comparisons.

These results demonstrate the benefit of experimental adaptation for
synthetic-trained models, but do not establish an advantage of synthetic
pretraining over experimental-only training under the investigated
conditions.

### Interpretation

The reported metrics quantify agreement with visually screened
SAM-assisted reference annotations.

Selected qualitative comparisons highlight differences in small-void
assignment and fiber-boundary delineation. These examples do not
independently establish which prediction is physically more accurate
in disputed regions or how frequently such discrepancies occur.

The current comparison does not establish reduced annotation
requirements or validate the predicted geometry for mechanical
simulation.

## Repository structure

```text
.
├── sam_segmentation/
│   ├── src/                  # SAM-assisted annotation and tile preparation
│   └── hpc/                  # Annotation submission scripts
├── synthetic_data/
│   ├── configs/              # Synthetic-generation settings
│   ├── src/                  # Synthetic-image generation
│   └── hpc/                  # Generation, pretraining, evaluation, fine-tuning
├── unet/
│   ├── configs/              # Dataset and training configurations
│   ├── environments/         # Additional environment specifications
│   ├── hpc/                  # Experimental preparation/training scripts
│   ├── src/                  # Model, data loading, training, and metrics
│   ├── requirements.txt
│   └── requirements-samenv-snapshot.txt
├── samenv.yml                # SAM environment specification
├── CITATION.cff
├── LICENSE
└── README.md
```

Additional analysis and visualization utilities may be added as the
manuscript-associated release is finalized.

## Software requirements

The training implementation uses Python and PyTorch. Dependencies listed
in `unet/requirements.txt` include:

- PyTorch
- NumPy
- Pillow
- SciPy
- Matplotlib

The complete workflow also requires the dependencies used by the
annotation and data-preparation scripts. SAM-assisted annotation
requires a compatible Segment Anything Model installation and checkpoint.

The requirements snapshot and environment specifications document
available environment information; platform-specific adaptation may
still be necessary.

### Basic installation

Clone the repository:

```bash
git clone [https://github.com/Mah-sa-a/cfrp-microstructure-segmentation.git](https://github.com/Mah-sa-a/cfrp-microstructure-segmentation.git)
cd cfrp-microstructure-segmentation
```

After preparing an appropriate Python environment and a PyTorch
installation compatible with your system, install the listed U-Net
dependencies:

```bash
python -m pip install -r unet/requirements.txt
```

Installing this requirements file alone does not install the complete
SAM annotation workflow.

The current training entry point requires a CUDA-capable GPU.

## Training configuration

| Parameter | Setting |
|---|---|
| Input | 512 × 512 pixels; one grayscale channel |
| Output | Three phase-logit channels |
| Loss | 0.5 multiclass cross-entropy + 0.5 soft Dice loss |
| Initial learning rate: experimental-only and synthetic pretraining | 0.001 |
| Initial learning rate: experimental fine-tuning | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 4 |
| Maximum epochs | 150 per training stage |
| Early-stopping patience setting | 20 |
| Base feature channels | 32 |
| Random seeds | 42, 43, 44 |
| Numerical precision | FP32 |

Principal experimental-only configuration:

```text
unet/configs/reviewed_multimag_mixedclass_fp32_20261111T182044Z_e01af2.json
```

Principal synthetic-pretraining/fine-tuning configuration:

```text
unet/configs/synthetic_v1_fp32_20261114T173950.json
```

Refer to the configurations and training implementation for the complete
settings, checkpoint-selection logic, and output paths.

## Running the training protocols

Run commands from the repository root after preparing the required
datasets, manifests, and software environment.

The examples below run seed 42 and fold 1. They are not a complete
dataset-preparation or manuscript-reproduction procedure.

### Experimental-only training

```bash
python unet/src/train.py \
  --protocol real_only \
  --seed 42 \
  --fold 1 \
  --config unet/configs/reviewed_multimag_mixedclass_fp32_20261111T182044Z_e01af2.json
```

### Synthetic pretraining

```bash
python unet/src/train.py \
  --protocol synthetic_pretrain \
  --seed 42 \
  --fold 1 \
  --config unet/configs/synthetic_v1_fp32_20261114T173950.json
```

### Synthetic-only experimental evaluation

Run after the corresponding synthetic pretrained checkpoint exists:

```bash
python unet/src/train.py \
  --protocol synthetic_only \
  --seed 42 \
  --fold 1 \
  --config unet/configs/synthetic_v1_fp32_20261114T173950.json
```

### Experimental fine-tuning

Run after the corresponding synthetic pretrained checkpoint exists:

```bash
python unet/src/train.py \
  --protocol synthetic_to_real \
  --seed 42 \
  --fold 1 \
  --config unet/configs/synthetic_v1_fp32_20261114T173950.json
```

Repeat experimental training/evaluation across seeds 42, 43, and 44 and
folds 1–5 for the full comparison. Synthetic pretraining is performed
per seed; it is not repeated independently for each experimental fold.

## HPC execution

Submission scripts are provided for the original Slurm-based computing
environment.

Before using them, review and adapt:

- Partition names and resource requests.
- Module names and versions.
- Conda environment and Python executable paths.
- Project and dataset locations.
- Configuration paths.
- Job-array indices.

Some scripts contain absolute paths from the original cluster.
They are provided as site-specific execution records, not universally
portable launch scripts.

The general experimental dataset-preparation submission script defaults
to exploratory preparation. It should not be interpreted as the exact
command used to produce the final reviewed multimag­nification partitions.

## Data and model availability

The repository contains source code and selected configurations.
It does not currently include:

- Experimental micrographs and reference masks.
- Generated synthetic datasets.
- Complete dataset manifests.
- Trained checkpoints.
- Bulk prediction outputs.

Configurations reference manifests under `data/splits/`. Running the
workflow requires the corresponding resources or regeneration of suitable
datasets and manifests.

Exact reproduction of the reported results additionally requires matching
the original data, annotations, partitions, and code versions.
A successful syntax check alone does not establish full reproducibility.

### Synthetic dataset record

The recorded synthetic experiment uses:

| Partition | Patches |
|---|---:|
| Training | 16,800 |
| Validation | 3,600 |
| Test | 3,600 |

The setup record is available at:

```text
synthetic_data/synthetic_v1_fp32_20261114T173950_setup.json
```

The repository includes the synthetic microstructure generator and configuration used to generate the synthetic dataset in this study.

## Citation

Software citation metadata is provided in `CITATION.cff`.

Until a versioned archive is available, cite the repository and identify
the commit used:

> Arabkoohichetabi, M. (2026). CFRP Microstructure Segmentation
> [Computer software]. GitHub.
> https://github.com/Mah-sa-a/cfrp-microstructure-segmentation

The manuscript-associated release identifier and archival DOI will be
added when finalized.

## License

Original code is distributed under the MIT License. See `LICENSE`
for the full terms.

Third-party software, pretrained models, and other external resources
remain subject to their respective licenses. The software license
does not establish availability or licensing terms for experimental data.

## Software author

Mahsa Arabkoohi  
Université Paris Nanterre — LEME

Manuscript authorship and contributor acknowledgments will be documented
with the finalized publication information.
