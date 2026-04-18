# medi-msde

## Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement (MSDE)

This repository contains the official implementation of **MSDE (Mean Shift Density Enhancement)**, a novel framework for **one-class anomaly detection in medical imaging**.

Our method enhances latent feature representations through a density-driven manifold refinement process, improving the detection of subtle and rare abnormalities in low-label settings.

---

##  Overview

Anomaly detection in medical imaging is challenging due to the scarcity of annotated abnormal data. To address this, we propose **MSDE**, a lightweight and effective post-processing module that refines feature embeddings before anomaly scoring.

The pipeline consists of:

1. **Feature Extraction** using pretrained backbones
2. **Mean Shift Density Enhancement (MSDE)** for latent space refinement
3. **Gaussian Density Estimation (GDE)** in PCA-reduced space
4. **Mahalanobis Distance-based Anomaly Scoring**

MSDE shifts feature representations toward high-density regions, resulting in:

* More compact normal clusters
* Better separation of anomalies
* Improved detection performance

---

##  Key Features

*  Novel **density-based manifold refinement (MSDE)**
*  Works with pretrained models (no retraining required)
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

##  Acknowledgements

This work builds upon the excellent **MedIAnomaly benchmark**:

*  Paper: *MedIAnomaly: A comparative study of anomaly detection in medical images* 
*  Code: https://github.com/caiyu6666/MedIAnomaly

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

##  Repository Structure

```
medi-msde/
│── README.md
│── requirements.txt
│
├── msde/              # MSDE core implementation
├── scripts/           # training / evaluation scripts
├── configs/           # hyperparameters
├── plots/             # figures (pipeline, UMAP, etc.)
```

---

##  Installation

```bash
git clone https://github.com/<your-username>/medi-msde.git
cd medi-msde
pip install -r requirements.txt
```

---

##  Usage

Run MSDE on extracted features:

```bash
python scripts/run_msde.py
```

*(Modify configs based on dataset and backbone settings)*

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

If you find this work useful, please cite:

```
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

If you find this repository useful, consider giving it a star ⭐

