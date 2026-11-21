# CFRP Microstructure Segmentation

Research code for three-phase segmentation of carbon fiber-reinforced
polymer (CFRP) optical micrographs into matrix, fiber, and void regions.

The workflow combines parameterized synthetic microstructure generation,
SAM-assisted experimental reference preparation, and PyTorch-based U-Net
training and evaluation.

## Study design

Three training strategies are compared:

1. Experimental-only training (`real_only`).
2. Synthetic-only training with direct experimental evaluation
   (`synthetic_pretrain` followed by `synthetic_only` evaluation).
3. Synthetic pretraining followed by experimental fine-tuning
   (`synthetic_to_real`).

Experimental evaluation uses five sample-wise folds and three random
seeds (42, 43, and 44). The strategies use identical experimental test
partitions. Experimental validation samples are separate from training
and test samples.

Fine-tuning uses the full experimental training partition available
within each fold, not the validation or test samples.

## Repository structure

- `sam_segmentation/`: SAM-assisted annotation code and HPC scripts.
- `synthetic_data/`: synthetic generation code, configurations, and
  HPC scripts.
- `unet/src/`: model, data preparation, training, prediction, and metrics.
- `unet/configs/`: experiment configurations.
- `unet/scripts/`: visualization and result-summary scripts.
- `unet/analysis/`: additional analysis utilities.
- `unet/hpc/`: HPC submission scripts.
- `post_processing/`: additional evaluation and summary utilities.
- `docs/`: workflow and dataset-preparation documentation.

Some files concern exploratory, legacy, or pilot experiments and are
not part of the principal manuscript comparison.

## Principal experiment configurations

Experimental-only training:

`unet/configs/reviewed_multimag_mixedclass_fp32_20261111T182044Z_e01af2.json`

Synthetic training and experimental fine-tuning:

`unet/configs/synthetic_v1_fp32_20261114T173950.json`

## Training settings

- Framework: PyTorch.
- Classes: matrix (0), fiber (1), and void (2).
- Loss: 0.5 multiclass cross-entropy + 0.5 soft Dice loss.
- Initial learning rate: 0.001 for experimental-only training and
  synthetic pretraining; 0.0001 for experimental fine-tuning.
- Weight decay: 0.0001.
- Batch size: 4.
- Maximum epochs: 150 per training stage.
- Early-stopping patience setting: 20.
- Numerical precision: FP32.

Refer to the configurations and training implementation for the
complete settings and checkpoint-selection logic.

## Dependencies

The U-Net requirements are listed in `unet/requirements.txt`.
SAM annotation requires additional dependencies and a compatible
SAM checkpoint.

Cluster module commands, environment names, and absolute filesystem
paths in HPC scripts are site-specific and must be adapted.

## Data and checkpoints

Experimental images, generated datasets, model checkpoints, and bulk
prediction outputs are not included in this code release. Dataset
availability and any separately archived resources will be documented
for the manuscript release.

Configurations reference dataset manifests under `data/splits/`.
Reproduction requires the corresponding data and manifests, or their
regeneration using the documented preparation workflow.

## Interpretation of results

Reported segmentation metrics quantify agreement with visually
screened SAM-assisted experimental references. They do not independently
resolve ambiguous small-void assignments or fiber interfaces.

## Citation

Citation metadata is provided in `CITATION.cff`. A version identifier
and archival DOI will be added when the manuscript-associated release
is finalized.

## Author

Mahsa Arabkoohichetabi
Université Paris Nanterre (LEME)

## License

Original code in this repository is released under the MIT License.
See `LICENSE` for the full terms. Third-party software and model
checkpoints remain subject to their respective licenses.
