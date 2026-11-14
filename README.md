# medi-msde

Mean Shift Density Enhancement for medical-image anomaly detection.

This repository extends the [MedIAnomaly](https://github.com/caiyu6666/MedIAnomaly)
benchmark with a frozen BiomedCLIP feature pipeline and MSDE-based anomaly detectors.
The implementation targets one-class anomaly detection: normal training images are used
to fit the reference model, while test labels are used only to compute evaluation metrics.

The MSDE operation is adapted from the original tabular implementation available at
<https://github.com/rbordoloi/msde>. This repository extends that idea to medical-image
embeddings and integrates it with the MedIAnomaly evaluation protocol.

## Overview

The main method is **MSDE-GDE**:

1. Extract frozen BiomedCLIP embeddings.
2. Standardise features using normal training statistics.
3. Apply Mean Shift Density Enhancement to normal training embeddings and fit a PCA-256
   Gaussian density estimator.
4. Apply MSDE to the unlabelled test cohort.
5. Score shifted test embeddings with Mahalanobis distance.

The repository also includes MSDE-⁠kNN, an inductive MLP surrogate for scoring
individual images after calibration, controlled detector comparisons, protocol ablations,
test-order sensitivity experiments, and paper-figure scripts.

The benchmark covers RSNA Pneumonia, VinDr-CXR, Brain Tumor, LAG, BraTS2021,
Camelyon16, and ISIC 2018.

## Repository layout

```text
medi-msde/
├── README.md
├── LICENSE
├── extract_features.py
├── MSDE_v3.py
├── benchmark.py
├── benchmark_utils.py
├── benchmark_biomedclip_detectors.py
├── ablation_msde_protocols.py
├── inductive_msde_surrogate.py
├── run_msde_gde.py
├── run_resnet_gde.py
├── statistical_test_wilcoxon.py
├── test_order_permutation.py
├── plot_paper_figures.py
├── plot_embedding_shift.py
├── reconstruction/
└── ssl/
```

`reconstruction/` and `ssl/` contain the inherited baseline implementations used by
the benchmark. Generated results, extracted embeddings, checkpoints, logs, and figures
are not required in the source repository.

### Provenance of baseline code

The `reconstruction/` and `ssl/` directories are derived from the original
[MedIAnomaly repository](https://github.com/caiyu6666/MedIAnomaly). They are included
here so that the benchmark can be reproduced from a single repository; they are not
new implementations introduced by this project. Any local modifications made for
benchmark integration are retained in this copy. The upstream repository is MIT-licensed;
the original attribution and licence terms are preserved in `LICENSE`.

## Installation

The experiments were developed with Python 3.10, PyTorch 2.1.2, and CUDA 11.8.
Install a compatible PyTorch build first, then install the remaining dependencies:

```bash
conda create -n medianomaly python=3.10
conda activate medianomaly
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn matplotlib tqdm umap-learn pynndescent \
    open_clip_torch transformers pillow joblib
```

Adjust the PyTorch command for the available GPU and CUDA version. BiomedCLIP requires
`open_clip_torch` and `transformers`.

## Dataset preparation

Prepare the datasets using the official MedIAnomaly instructions:

<https://github.com/caiyu6666/MedIAnomaly>

The expected dataset root is:

```text
MedIAnomaly-Data/
├── RSNA/
├── VinCXR/
├── BrainTumor/
├── LAG/
├── ISIC2018_Task3/
├── Camelyon16/
└── BraTS2021/
```

The inherited baseline configuration may require the dataset root to be adjusted for
your machine. The datasets are not redistributed and remain subject to their licences.

## Feature extraction

Extract BiomedCLIP features for all datasets:

```bash
python extract_features.py \
    --dataset all \
    --backbone biomedclip \
    --gpu 0 \
    --batch-size 4 \
    --save-root features
```

Use `--cpu` for CPU inference. Features are saved as:

```text
features/<dataset>/biomedclip/
├── train_features.npy
├── train_labels.npy
├── train_names.npy
├── test_features.npy
├── test_labels.npy
└── test_names.npy
```

The extractor also supports the existing `resnet18` and `anatpaste` backbones, although
the reported MSDE experiments use BiomedCLIP.

## Main benchmark

Run the complete comparison with fold 0, seed 42, and 100 epochs for the two-stage
baselines:

```bash
python benchmark.py \
    --gpu 0 \
    --datasets all \
    --methods all \
    --include-feature-extraction \
    --feature-batch-size 4 \
    --workers 2 \
    --epochs 100 \
    --output-dir results/benchmark_results \
    --continue-on-error
```

If BiomedCLIP features already exist under `features/`, omit
`--include-feature-extraction`. Results and logs are written to `results/benchmark_results/`.

The main benchmark contains AE, AE-Perceptual, VAE, MemAE, GANomaly, two-stage CutPaste,
AnatPaste, ImageNet ResNet18-GDE, ImageNet ResNet152-GDE, and MSDE-GDE with BiomedCLIP.

## Controlled BiomedCLIP detector comparison

```bash
python benchmark_biomedclip_detectors.py \
    --datasets all \
    --detectors all \
    --backbone biomedclip \
    --seed 42 \
    --save-dir results/results_biomedclip_detectors
```

This evaluates GDE, (k)NN, LOF, Isolation Forest, OC-SVM, MSDE-reference, MSDE-(k)NN,
and MSDE-GDE on the same saved BiomedCLIP feature space.

## MSDE protocol ablation

```bash
python ablation_msde_protocols.py \
    --datasets all \
    --backbone biomedclip \
    --protocols all \
    --seed 42 \
    --save-dir results/results_msde_ablation
```

The five protocols are no shift, train-only field, shifted train only, shifted test only,
and transductive shifting of both sides.

## Inductive MSDE-MLP surrogate

Run five random splits with an 80/20 calibration/held-out partition:

```bash
python inductive_msde_surrogate.py \
    --datasets all \
    --splits 5 \
    --teacher-fraction 0.8 \
    --seed 42 \
    --save-dir results/results_inductive_msde_surrogate_5splits
```

The calibration cohort is unlabelled. Held-out images do not enter the MSDE shift or
MLP training. Use `--teacher-fraction 0.5` or `0.3` for the appendix experiments.

## Test-order sensitivity

```bash
python test_order_permutation.py \
    --datasets all \
    --backbone biomedclip \
    --n-permutations 5 \
    --seed 42 \
    --save-dir results/results_order_permutation
```

This evaluates random permutations and label-sorted diagnostic orders for the
transductive method. Labels are used only for constructing the diagnostic orders and
computing final metrics.

## Standalone MSDE-GDE

```bash
python run_msde_gde.py \
    --feature-root features \
    --datasets all \
    --seed 42 \
    --save-dir results/results_msde_gde
```

## Statistical-test table

```bash
python statistical_test_wilcoxon.py \
    --results results/benchmark_results/all_results.csv \
    --output-dir results/statistical_test_results
```

The script reports exact signed-rank (p)-values, rank sums, and multiple-comparison
correction columns for transparent reporting.

## Paper figures

Generate the main figures from saved result files:

```bash
python plot_paper_figures.py \
    --root-dir . \
    --output-dir paper_figures \
    --include-runtime \
    --include-teacher-fraction
```

Generate raw-versus-shifted embedding visualisations:

```bash
python plot_embedding_shift.py \
    --datasets all \
    --feature-root features \
    --output-dir paper_figures/embedding_shift \
    --max-points 2500 \
    --seed 42
```

The latter creates PCA and UMAP grids, including a readable four-dataset PCA grid.
Labels are used only to color points in these visualisations.

## Reproducibility notes

- The reported benchmark uses fold 0 and seed 42.
- MSDE hyperparameters are fixed across all seven datasets and modalities.
- MSDE-GDE is transductive because the test-cohort shift is computed from the unlabelled
  test cohort.
- MSDE-MLP is an inductive surrogate after calibration on an unlabelled cohort.
- Metrics are image-level AUC, AP, and Precision@N.
- Runtime scopes are reported separately because learned baselines include training,
  whereas frozen-feature methods include feature extraction and detector execution.

## Acknowledgements

This work builds on the MedIAnomaly benchmark and related baseline implementations:

- Original tabular MSDE: <https://github.com/rbordoloi/msde>
- MedIAnomaly: <https://github.com/caiyu6666/MedIAnomaly>
- CutPaste: <https://github.com/Runinho/pytorch-cutpaste>
- AnatPaste: <https://github.com/jun-sato/AnatPaste>
- Natural Synthetic Anomalies: <https://github.com/hmsch/natural-synthetic-anomalies>

Please follow the licences and citation requirements of the original datasets and
repositories.

## Citation

```bibtex
@article{kar2026msde,
  title={Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement},
  author={Kar, Pritam and Lakshmi, Gouri and Bej, Saptarshi},
  journal={arXiv preprint arXiv:2604.19191},
  year={2026},
  doi={10.48550/arXiv.2604.19191}
}
```
