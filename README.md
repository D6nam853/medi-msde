# medi-msde

## Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement (MSDE)

This repository contains the official implementation of **MSDE (Mean Shift Density Enhancement)**, a novel framework for **one-class anomaly detection in medical imaging**.

Our method enhances latent feature representations through a density-driven manifold refinement process, improving the detection of subtle and rare abnormalities in low-label settings.

---

##  Overview

Anomaly detection in medical imaging is challenging due to the scarcity of annotated abnormal data. To address this, we propose **MSDE**, a lightweight and effective post-processing module that refines feature embeddings before anomaly scoring.

The pipeline consists of:

* Feature Extraction using pretrained backbones
* Mean Shift Density Enhancement (MSDE) for latent space refinement
* Gaussian Density Estimation (GDE) in PCA-reduced space
* Mahalanobis Distance-based Anomaly Scoring

MSDE shifts feature representations toward high-density regions, resulting in:

* More compact normal clusters
* Better separation of anomalies
* Improved detection performance

---

##  MSDE Pipeline

<p align="center">
  <img src="plots/msde_pipeline.png" width="100%">
</p>

<p align="center">
  <em>Overview of the MSDE pipeline: feature extraction, manifold refinement, density estimation, and anomaly scoring.</em>
</p>

---

##  Key Features

*  Novel **density-based manifold refinement (MSDE)**
*  Works with pretrained models (**no retraining required**)
*  Plug-and-play module for anomaly detection pipelines
*  Robust performance across multiple medical imaging datasets
*  Efficient and scalable (operates on feature embeddings)

---

##  Experimental Setup

We follow the benchmark and evaluation protocol from the **MedIAnomaly** framework, which includes:

* 7 medical datasets across multiple modalities (X-ray, MRI, histopathology, etc.)
* One-class training (only normal samples)
* Evaluation using **AUC-ROC** and **Average Precision (AP)**

---

##  Data Preparation

We follow the dataset preparation protocol from the original **MedIAnomaly** repository.

https://github.com/caiyu6666/MedIAnomaly

Datasets used:

* RSNA Pneumonia
* VinDr-CXR
* Brain Tumor
* LAG
* ISIC 2018
* Camelyon16
* BraTS2021

After downloading, organize the data according to the structure expected by the MedIAnomaly pipeline.

>  We do not redistribute datasets due to licensing restrictions.
> For detailed setup instructions, please refer to the official repository.

---

##  Repository Structure

```
medi-msde/
│── README.md
│── LICENSE
│
├── msde/                # MSDE core implementation
│   └── msde.py
│
├── scripts/             # pipeline scripts
│   ├── extract_features.py
│   └── run_msde.py
│
├── visualizations/      # embedding & MSDE plots
│   ├── visualize_embeddings.py
│   └── visualize_msde_shifted_embeddings.py
│
├── plots/               # figures (pipeline & result visualizations)
│   └── msde_pipeline.png
```

The repository is organized into modular components, separating **feature extraction**, **MSDE processing**, **visualization**, and **result artifacts** for clarity and reproducibility.

---

##  Installation

```bash
git clone https://github.com/D6nam853/medi-msde.git
cd medi-msde
pip install -r requirements.txt
```

---

##  Usage

### Step 1: Extract Features

```bash
python scripts/extract_features.py
```

### Step 2: Run MSDE

```bash
python scripts/run_msde.py
```

### Step 3: Visualization (Optional)

```bash
python visualizations/visualize_embeddings.py
python visualizations/visualize_msde_shifted_embeddings.py
```

---

##  Acknowledgements

This work builds upon the excellent **MedIAnomaly benchmark**:

*  *MedIAnomaly: A comparative study of anomaly detection in medical images*
*  https://github.com/caiyu6666/MedIAnomaly

We utilize their:

* Dataset preparation pipeline
* Evaluation protocol
* Feature extraction setup

Our work extends this framework by introducing **MSDE** as a latent space refinement module.

---

##  Our Contributions

Compared to the original MedIAnomaly repository, this repo includes:

* Implementation of **Mean Shift Density Enhancement (MSDE)**
* Integration of MSDE into the anomaly detection pipeline
* Extensive benchmarking with fixed hyperparameters
* Analysis of latent space refinement for anomaly detection

---

##  Important Note

This repository is an **extension of MedIAnomaly**, not a standalone reimplementation.

To fully reproduce experiments, please refer to the original repository for:

* Dataset preparation
* Baseline implementations
* Pretrained models

---

##  Results

MSDE achieves strong and consistent performance across datasets, including:

* State-of-the-art AUC on multiple benchmarks
* Near-perfect performance on Brain Tumor detection (~0.98 AUC/AP)
* Robust results with fixed hyperparameters

---

##  Citation

```bibtex
@article{kar2026msde,
  title={Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement},
  author={Kar, Pritam and others},
  year={2026}
}
```

---

##  License

This project is licensed under the MIT License.

---

##  Acknowledgment

If you find this repository useful, consider giving it a star 
