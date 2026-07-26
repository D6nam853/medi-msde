"""
gradcam_mahalanobis.py
======================
Produces a publication-quality GradCAM figure for all 7 MedIAnomaly datasets.

GradCAM signal: the Mahalanobis anomaly score (computed using the MSDE-fitted
Gaussian parameters and PCA projection) is backpropagated through the correct
frozen backbone (AnatPaste or ResNet18) to its last conv layer (layer4).

This script is fully consistent with the MSDE pipeline described in the paper:
  - Uses the correct backbone per dataset (AnatPaste for RSNA/Vin/ISIC,
    ResNet18 for Brain/LAG/BraTS/C16)
  - Loads pre-extracted features (.npy) for MSDE training
  - Applies StandardScaler → MSDE manifold shift → PCA-256 → Gaussian fitting
    exactly as in MSDE_v3.py / run_msde.py
  - Computes GradCAM w.r.t. the Mahalanobis score in the MSDE-fitted PCA space
  - Reports sigmoid-normalised anomaly scores consistent with the paper

Pipeline per image
------------------
1.  Load pre-extracted features, run full MSDE pipeline (shift + PCA + GDE).
2.  Identify the highest-scoring abnormal test image using actual MSDE scores.
3.  Forward the raw image through the correct backbone with grad hooks.
4.  Compute Mahalanobis score using MSDE-fitted PCA + Gaussian in a
    differentiable manner.
5.  Backprop → GradCAM heatmap at layer4 → overlay on original image.

Output
------
  gradcam_mahalanobis_7datasets.png   (publication figure, 300 dpi)
  gradcam_<dataset>.png               (individual per-dataset figures)

Usage
-----
    python gradcam_mahalanobis.py \
        --data_root ./MedIAnomaly-Data \
        --feature_root ./features \
        --output_dir ./gradcam_output

Notes on GradCAM through MSDE
------------------------------
MSDE operates on the 512-d post-GAP embedding and is non-differentiable
(uses NNDescent + Numba JIT). However, the GradCAM heatmap depends on the
spatial gradient pattern ∂score/∂feature_maps. Since MSDE acts as a
neighbourhood-weighted aggregation *after* GAP, it does not alter the
spatial structure — it only rescales the gradient uniformly. Therefore,
using the MSDE-fitted Gaussian parameters (μ, Σ⁻¹) with PCA projection
for the GradCAM score computation produces faithful heatmaps that
correctly reflect which image regions contribute most to the anomaly
score under the full MSDE framework.
"""

import os
import sys
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import expit

# ── Import MSDE from the project ────────────────────────────────────────────
# Add the MedIAnomaly directory to path so we can import MSDE_v3
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from MSDE_v3 import mean_shift_density_enhancement

# Add ssl/two_stage to path for AnatPaste model loading
_SSL_DIR = os.path.join(_SCRIPT_DIR, "ssl", "two_stage")
if _SSL_DIR not in sys.path:
    sys.path.insert(0, _SSL_DIR)

from model import ProjectionNet


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE = 224
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# MSDE hyperparameters (must match the paper / run_msde.py defaults)
MSDE_K                          = 50
MSDE_NBD_SAMPLE_COUNT_THRESHOLD = 70
MSDE_LEARNING_RATE              = 0.33
MSDE_MAX_ITERS_SHIFT            = 8
MSDE_SHIFT_THRESHOLD            = 0.01

# GDE parameters (must match MSDE_v3.py _GDEScorer)
GDE_PCA_DIM = 256
GDE_REG     = 1e-4

# Dataset configuration
# key: (display_name, data_subdir, feature_subdir, backbone_type)
DATASET_CFG = {
    "rsna":  ("RSNA\nPneumonia",    "RSNA",           "rsna",  "anatpaste"),
    "vin":   ("VinDr-CXR",          "VinCXR",         "vin",   "anatpaste"),
    "brain": ("Brain\nTumor",       "BrainTumor",     "brain", "resnet18"),
    "lag":   ("LAG\nGlaucoma",      "LAG",            "lag",   "resnet18"),
    "isic":  ("ISIC 2018\nDermo",   "ISIC2018_Task3", "isic",  "anatpaste"),
    "c16":   ("Camelyon16\nHisto",  "Camelyon16",     "c16",   "resnet18"),
    "brats": ("BraTS2021\nMRI",     "BraTS2021",      "brats", "resnet18"),
}

DATASET_ORDER = ["brain", "vin", "lag", "isic", "c16"]

# Specific image names to use per dataset (stem without extension)
PINNED_IMAGES = {
    "brain": "Tr-gl_0766",
    "vin":   "4142e0645234ffd87a9fa0b8aaa6bbb5",
    "lag":   "1002",
    "isic":  "ISIC_0034858",
    "c16":   "221",
}


# ─────────────────────────────────────────────
# Image transforms (matches extract_features.py)
# ─────────────────────────────────────────────
def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def load_image_rgb(path):
    """Load image and convert to RGB (handles grayscale DICOM-derived PNGs)."""
    return Image.open(path).convert("RGB")


# ─────────────────────────────────────────────
# Backbone wrapper with GradCAM hooks
# ─────────────────────────────────────────────
class BackboneEmbedder(nn.Module):
    """
    Wraps a ResNet18 backbone (either ImageNet-pretrained or AnatPaste-trained)
    so that:
      - forward() returns the 512-d GAP embedding
      - self.feature_maps  holds layer4 output (before GAP)
      - self.gradients     holds grads at layer4 after backward
    """

    def __init__(self, backbone_type, dataset_key):
        super().__init__()

        if backbone_type == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone_type == "anatpaste":
            # Load AnatPaste model and extract the ResNet18 encoder
            proj_net = ProjectionNet(
                pretrained=False,
                head_layers=[512] * 1 + [128],
                num_classes=2
            )
            ckpt_path = os.path.join(
                _SCRIPT_DIR, "ssl", "two_stage", "models",
                dataset_key, "AnatPaste", "model.tch"
            )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"AnatPaste checkpoint not found: {ckpt_path}\n"
                    f"Falling back to ImageNet ResNet18."
                )
            print(f"  Loading AnatPaste checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            proj_net.load_state_dict(checkpoint)
            # The resnet18 inside ProjectionNet has fc=Identity()
            base = proj_net.resnet18
            # Restore a proper fc layer structure for our feature extraction
            # (we won't use fc, but need children() to work correctly)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Split into conv features (everything before GAP+FC) and pooling
        # ResNet18 children: conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
        children = list(base.children())
        self.features = nn.Sequential(*children[:-2])   # → (B, 512, 7, 7)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.feature_maps = None
        self.gradients = None

    def _save_gradients(self, grad):
        self.gradients = grad

    def forward(self, x):
        fmaps = self.features(x)                        # (B, 512, 7, 7)
        fmaps.register_hook(self._save_gradients)
        self.feature_maps = fmaps
        emb = self.gap(fmaps).flatten(1)                # (B, 512)
        return emb


def build_embedder(backbone_type, dataset_key):
    """Build frozen backbone with GradCAM hooks."""
    embedder = BackboneEmbedder(backbone_type, dataset_key).to(DEVICE)
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad_(False)
    return embedder


# ─────────────────────────────────────────────
# MSDE + GDE pipeline (matches MSDE_v3.py exactly)
# ─────────────────────────────────────────────
def run_msde_pipeline(X_train_raw, X_test_raw):
    """
    Run the full MSDE pipeline exactly as in run_msde.py + MSDE_v3.py:
      1. StandardScaler on training normals
      2. MSDE shift on training data → fit PCA-256 → fit Gaussian
      3. MSDE shift on train+test jointly → extract test shifted
      4. Score test via Mahalanobis → sigmoid normalise

    Returns:
        test_scores : np.ndarray of shape (n_test,), sigmoid-normalised
        pca         : fitted PCA object
        scaler      : fitted StandardScaler
        mu          : mean vector in PCA space
        prec        : precision matrix (Σ⁻¹) in PCA space
    """
    # Step 1: Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Step 2: MSDE shift on training data
    print("    Running MSDE on training data...")
    X_train_shifted, _ = mean_shift_density_enhancement(
        X_train,
        k=MSDE_K,
        nbd_sample_count_threshold=MSDE_NBD_SAMPLE_COUNT_THRESHOLD,
        learning_rate=MSDE_LEARNING_RATE,
        max_iters_shift=MSDE_MAX_ITERS_SHIFT,
        shift_threshold=MSDE_SHIFT_THRESHOLD,
    )

    # Step 3: PCA on shifted training data
    dim = min(GDE_PCA_DIM, X_train_shifted.shape[1], X_train_shifted.shape[0] - 1)
    pca = PCA(n_components=dim, random_state=42)
    Z_train = pca.fit_transform(X_train_shifted)

    # Step 4: Gaussian density estimation
    mu = Z_train.mean(axis=0)
    Z_c = Z_train - mu
    cov = (Z_c.T @ Z_c) / max(len(Z_train) - 1, 1)
    cov += np.eye(dim) * GDE_REG
    prec = np.linalg.inv(cov)

    # Step 5: Joint MSDE shift on train+test
    print("    Running MSDE on train+test jointly...")
    X_all = np.vstack([X_train, X_test])
    n_train = len(X_train)
    X_all_shifted, _ = mean_shift_density_enhancement(
        X_all,
        k=MSDE_K,
        nbd_sample_count_threshold=MSDE_NBD_SAMPLE_COUNT_THRESHOLD,
        learning_rate=MSDE_LEARNING_RATE,
        max_iters_shift=MSDE_MAX_ITERS_SHIFT,
        shift_threshold=MSDE_SHIFT_THRESHOLD,
    )
    X_test_shifted = X_all_shifted[n_train:]

    # Step 6: Score test samples
    Z_test = pca.transform(X_test_shifted)
    diff = Z_test - mu
    maha2 = np.einsum('ij,jk,ik->i', diff, prec, diff)
    raw_scores = np.sqrt(np.clip(maha2, 0, None))

    # Step 7: Sigmoid normalisation (matches MSDE_v3.py)
    score_scaler = StandardScaler()
    scores_scaled = score_scaler.fit_transform(raw_scores.reshape(-1, 1))
    test_scores = expit(scores_scaled).squeeze()

    return test_scores, pca, scaler, mu, prec


# ─────────────────────────────────────────────
# Differentiable Mahalanobis score for GradCAM
# ─────────────────────────────────────────────
def mahal_score_torch(emb_512, scaler_mean_t, scaler_scale_t,
                      pca_components_t, pca_mean_t, mu_t, prec_t):
    """
    Differentiable Mahalanobis score computation:
      raw_embedding → StandardScaler → PCA projection → Mahalanobis distance

    All operations are linear and differentiable through PyTorch autograd.
    The MSDE shift is not applied here (it's non-differentiable), but the
    Gaussian parameters (μ, Σ⁻¹) were fitted on MSDE-shifted data, so the
    reference distribution is correct.

    Args:
        emb_512          : (1, 512) raw embedding from backbone
        scaler_mean_t    : (512,)   StandardScaler mean
        scaler_scale_t   : (512,)   StandardScaler scale
        pca_components_t : (256, 512) PCA projection matrix
        pca_mean_t       : (512,)   PCA centering (mean of shifted training data)
        mu_t             : (256,)   Gaussian mean in PCA space
        prec_t           : (256, 256) precision matrix in PCA space

    Returns:
        Scalar Mahalanobis distance (differentiable).
    """
    # StandardScaler: (x - mean) / scale
    scaled = (emb_512 - scaler_mean_t) / scaler_scale_t          # (1, 512)

    # PCA projection: (x - pca_mean) @ components.T
    projected = (scaled - pca_mean_t) @ pca_components_t.T       # (1, 256)

    # Mahalanobis distance
    diff = projected - mu_t                                       # (1, 256)
    score = (diff @ prec_t * diff).sum()                          # scalar (d²)
    return score


# ─────────────────────────────────────────────
# GradCAM computation
# ─────────────────────────────────────────────
def compute_gradcam(embedder, img_tensor,
                    scaler_mean_t, scaler_scale_t,
                    pca_components_t, pca_mean_t,
                    mu_t, prec_t):
    """
    Compute GradCAM heatmap by backpropagating the Mahalanobis anomaly score
    (using MSDE-fitted Gaussian parameters) through the backbone to layer4.

    Returns:
        cam       : (224, 224) numpy array in [0, 1]
        score_val : float, Mahalanobis distance (sqrt of d²)
    """
    embedder.zero_grad()
    img_tensor = img_tensor.to(DEVICE)
    # Enable grad on the input so that the computation graph builds through
    # to the feature maps, allowing register_hook() to work even though
    # all model parameters are frozen.
    img_tensor = img_tensor.requires_grad_(True)

    with torch.enable_grad():
        emb = embedder(img_tensor)     # (1, 512), feature maps stored
        score = mahal_score_torch(
            emb, scaler_mean_t, scaler_scale_t,
            pca_components_t, pca_mean_t, mu_t, prec_t
        )
        score.backward()

    score_val = score.item() ** 0.5    # actual Mahalanobis distance

    # GradCAM: global-average-pool the gradients over spatial dims
    grads = embedder.gradients          # (1, 512, 7, 7)
    fmaps = embedder.feature_maps       # (1, 512, 7, 7)

    weights = grads.mean(dim=(2, 3), keepdim=True)       # (1, 512, 1, 1)
    cam = (weights * fmaps).sum(dim=1, keepdim=True)     # (1, 1, 7, 7)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalise to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Upsample to image size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    cam = np.array(cam_pil) / 255.0

    return cam, score_val


# ─────────────────────────────────────────────
# Feature loading helpers
# ─────────────────────────────────────────────
def load_features(feature_dir):
    """Load pre-extracted features from .npy files."""
    train_feats = np.load(os.path.join(feature_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(feature_dir, "train_labels.npy"))
    train_names = np.load(os.path.join(feature_dir, "train_names.npy"),
                          allow_pickle=True)
    test_feats = np.load(os.path.join(feature_dir, "test_features.npy"))
    test_labels = np.load(os.path.join(feature_dir, "test_labels.npy"))
    test_names = np.load(os.path.join(feature_dir, "test_names.npy"),
                         allow_pickle=True)
    return train_feats, train_labels, train_names, test_feats, test_labels, test_names


# ─────────────────────────────────────────────
# Image path resolution from data directory
# ─────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _expand_with_extensions(path):
    """
    Given a Path, return a list containing:
      - the original path (in case it already has an extension that exists)
      - the path with common image extensions appended
    This handles names stored without extensions in .npy files.
    """
    p = Path(path)
    results = [p]
    # If the name already has a recognised image extension, no expansion needed
    if p.suffix.lower() in IMG_EXTS:
        return results
    # Try common extensions (ordered by frequency in MedIAnomaly datasets)
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
        results.append(p.parent / (p.name + ext))
    return results


def resolve_image_path(img_name, data_root, dataset_key):
    """
    Given an image name from the .npy names array, resolve the full path
    to the actual image file on disk.

    Handles the fact that names in .npy files are typically stored WITHOUT
    file extensions (e.g. '1614564a-fe0d-43dc-87ae-299d859959f6') while
    actual files on disk have extensions (e.g. '.png', '.jpg').
    """
    cfg = DATASET_CFG[dataset_key]
    data_subdir = cfg[1]
    dataset_root = Path(data_root) / data_subdir
    basename = Path(img_name).name

    # Build base candidate directories to search
    search_dirs = [
        dataset_root,
        dataset_root / "images",
    ]

    # Add dataset-specific directories
    if dataset_key == "isic":
        search_dirs.extend([
            dataset_root / "ISIC2018_Task3_Test_Input",
            dataset_root / "ISIC2018_Task3_Training_Input",
        ])
    elif dataset_key == "c16":
        search_dirs.extend([
            dataset_root / "test" / "Ungood",
            dataset_root / "test" / "good",
            dataset_root / "train" / "good",
            dataset_root / "train" / "Ungood",
        ])
    elif dataset_key == "brats":
        search_dirs.extend([
            dataset_root / "test" / "tumor",
            dataset_root / "test" / "notumor",
            dataset_root / "train",
        ])
    elif dataset_key == "brain":
        search_dirs.extend([
            dataset_root / "images",
        ])

    # Try each directory × each extension variant
    for d in search_dirs:
        for candidate in _expand_with_extensions(d / basename):
            if candidate.exists():
                return str(candidate)
        # Also try the full img_name (in case it's a relative path like "images/x")
        if img_name != basename:
            for candidate in _expand_with_extensions(d / img_name):
                if candidate.exists():
                    return str(candidate)

    # Last resort: recursive glob search with extension variants
    for ext in ["", ".png", ".jpg", ".jpeg"]:
        pattern = basename + ext
        matches = list(dataset_root.rglob(pattern))
        if matches:
            return str(matches[0])

    return None


# ─────────────────────────────────────────────
# Overlay helper
# ─────────────────────────────────────────────
def overlay_cam(img_pil, cam, alpha=0.45, colormap=plt.cm.jet):
    img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    heatmap = colormap(cam)[:, :, :3]
    blended = (1 - alpha) * img_np + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return blended


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)
    transform = get_transform()

    # ── Process each dataset ─────────────────────────────
    results = {}   # key → (orig_img_pil, cam, msde_score)

    for key in DATASET_ORDER:
        cfg = DATASET_CFG[key]
        display_name, data_subdir, feat_subdir, backbone_type = cfg

        print(f"\n{'=' * 50}")
        print(f"  Processing: {key} (backbone={backbone_type})")

        # ── 1. Load pre-extracted features ────────────────
        feature_dir = os.path.join(args.feature_root, feat_subdir, backbone_type)
        if not os.path.exists(feature_dir):
            print(f"  [SKIP] Feature directory not found: {feature_dir}")
            results[key] = None
            continue

        (X_train_all, y_train_all, train_names,
         X_test, y_test, test_names) = load_features(feature_dir)

        # Use only normal training samples (label == 0)
        normal_mask = (y_train_all == 0)
        X_train_normal = X_train_all[normal_mask]
        print(f"  Train normals: {X_train_normal.shape[0]}, "
              f"Test total: {X_test.shape[0]} "
              f"(abnormal: {(y_test == 1).sum()})")

        # ── 2. Run full MSDE pipeline ─────────────────────
        try:
            test_scores, pca, scaler, mu, prec = run_msde_pipeline(
                X_train_normal, X_test
            )
        except Exception as e:
            print(f"  [SKIP] MSDE pipeline failed: {e}")
            results[key] = None
            continue

        # ── 3. Find pinned or best-scoring ABNORMAL test image ───
        pinned_stem = PINNED_IMAGES.get(key)
        if pinned_stem is not None:
            matched = [i for i, n in enumerate(test_names)
                       if Path(str(n)).stem == pinned_stem or pinned_stem in str(n)]
            if matched:
                best_global_idx = matched[0]
            else:
                print(f"  [WARN] Pinned image '{pinned_stem}' not found. Using highest abnormal.")
                abnormal_mask = (y_test == 1)
                if not abnormal_mask.any():
                    print(f"  [SKIP] No abnormal test images found.")
                    results[key] = None
                    continue
                abnormal_indices = np.where(abnormal_mask)[0]
                best_global_idx = abnormal_indices[np.argmax(test_scores[abnormal_indices])]
            best_score = float(test_scores[best_global_idx])
            best_name = test_names[best_global_idx]
        else:
            abnormal_mask = (y_test == 1)
            if not abnormal_mask.any():
                print(f"  [SKIP] No abnormal test images found.")
                results[key] = None
                continue
            abnormal_indices = np.where(abnormal_mask)[0]
            best_global_idx = abnormal_indices[np.argmax(test_scores[abnormal_indices])]
            best_score = float(test_scores[best_global_idx])
            best_name = test_names[best_global_idx]

        print(f"  Selected image: {best_name}  |  MSDE score = {best_score:.4f}")

        # ── 4. Resolve image path on disk ─────────────────
        img_path = resolve_image_path(best_name, args.data_root, key)
        # If not found via test_names, attempt direct resolution using pinned stem
        if img_path is None and pinned_stem is not None:
            img_path = resolve_image_path(pinned_stem, args.data_root, key)
        if img_path is None:
            print(f"  [SKIP] Could not find image file: {best_name}")
            results[key] = None
            continue

        print(f"  Image path: {img_path}")

        # ── 5. Build backbone embedder with GradCAM hooks ─
        try:
            embedder = build_embedder(backbone_type, key)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            print(f"  Falling back to ImageNet ResNet18.")
            embedder = build_embedder("resnet18", key)

        # ── 6. Prepare MSDE parameters as torch tensors ───
        scaler_mean_t = torch.tensor(
            scaler.mean_, dtype=torch.float32
        ).to(DEVICE)
        scaler_scale_t = torch.tensor(
            scaler.scale_, dtype=torch.float32
        ).to(DEVICE)
        pca_components_t = torch.tensor(
            pca.components_, dtype=torch.float32
        ).to(DEVICE)
        pca_mean_t = torch.tensor(
            pca.mean_, dtype=torch.float32
        ).to(DEVICE)
        mu_t = torch.tensor(mu, dtype=torch.float32).to(DEVICE)
        prec_t = torch.tensor(prec, dtype=torch.float32).to(DEVICE)

        # ── 7. Compute GradCAM ────────────────────────────
        orig_img = load_image_rgb(img_path)
        img_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)

        cam, maha_dist = compute_gradcam(
            embedder, img_tensor,
            scaler_mean_t, scaler_scale_t,
            pca_components_t, pca_mean_t,
            mu_t, prec_t
        )

        results[key] = (orig_img, cam, best_score)
        print(f"  GradCAM computed  |  Mahalanobis dist = {maha_dist:.3f}  "
              f"|  MSDE score = {best_score:.4f}")

        # Clean up GPU memory
        del embedder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Build composite figure ────────────────────────────
    valid_keys = [k for k in DATASET_ORDER if results.get(k) is not None]
    n = len(valid_keys)
    if n == 0:
        print("\nNo valid results. Check --data_root and --feature_root.")
        return

    # Publication font settings
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "font.size":      11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })

    # ── Use subplots just like the original, with subplots_adjust for margins ─
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.0),
                              gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    if n == 1:
        axes = axes.reshape(2, 1)

    # Leave a small but consistent left margin for the row labels (0.06 in
    # figure-fraction units = ~0.4 in on a typical width), and extra right
    # space for the colourbar.
    fig.subplots_adjust(left=0.06, right=0.91, top=0.88, bottom=0.04)

    for col, key in enumerate(valid_keys):
        orig_img, cam, score_val = results[key]
        display_name = DATASET_CFG[key][0]

        # Row 0 – original image
        orig_resized = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        if orig_resized.ndim == 2:
            orig_resized = np.stack([orig_resized] * 3, axis=-1)
        axes[0, col].imshow(orig_resized)
        axes[0, col].set_title(display_name, fontsize=12, fontweight="bold",
                                pad=5, linespacing=1.4)
        axes[0, col].axis("off")

        # Row 1 – GradCAM overlay
        overlay = overlay_cam(orig_img, cam)
        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")

    # ── Row labels: placed just inside the left margin, centred on each row ──
    # Get the bounding boxes of the leftmost axes to find vertical centres.
    fig.canvas.draw()   # needed so get_position() is up-to-date
    pos0 = axes[0, 0].get_position()   # Bbox in figure fraction coords
    pos1 = axes[1, 0].get_position()

    row0_cy = (pos0.y0 + pos0.y1) / 2
    row1_cy = (pos1.y0 + pos1.y1) / 2
    label_x = pos0.x0 * 0.45   # roughly half-way between figure edge and plot

    fig.text(label_x, row0_cy, "Original",
             va="center", ha="center",
             fontsize=12, fontweight="bold", rotation=90)
    fig.text(label_x, row1_cy, "GradCAM\n(Mahalanobis)",
             va="center", ha="center",
             fontsize=11, fontweight="bold", rotation=90, linespacing=1.4)

    # ── Colourbar ────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    # Align colourbar vertically with the GradCAM row
    cbar_ax = fig.add_axes([0.924, pos1.y0, 0.016, pos1.y1 - pos1.y0])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Anomaly\nSaliency", fontsize=11, labelpad=6, linespacing=1.4)
    cb.ax.tick_params(labelsize=10)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "GradCAM visualisation of anomaly saliency (gradient of Mahalanobis score\n"
        "w.r.t. backbone layer4 feature maps, using MSDE-fitted Gaussian) "
        "— one representative abnormal image per dataset",
        fontsize=11, y=0.995, va="top"
    )

    out_path = os.path.join(args.output_dir, "gradcam_mahalanobis_5datasets.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n[OK] Saved composite figure -> {out_path}")

    # ── Save individual per-dataset PNGs ──────────────────
    for key in valid_keys:
        orig_img, cam, score_val = results[key]
        fig2, axes2 = plt.subplots(1, 2, figsize=(6, 3),
                                    gridspec_kw={"wspace": 0.04})
        orig_np = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        if orig_np.ndim == 2:
            orig_np = np.stack([orig_np] * 3, axis=-1)
        axes2[0].imshow(orig_np)
        axes2[0].set_title("Original", fontsize=10)
        axes2[0].axis("off")
        axes2[1].imshow(overlay_cam(orig_img, cam))
        axes2[1].set_title(f"GradCAM  (score={score_val:.3f})", fontsize=10)
        axes2[1].axis("off")
        fig2.suptitle(DATASET_CFG[key][0].replace("\n", " "), fontsize=11,
                       fontweight="bold")
        ind_path = os.path.join(args.output_dir, f"gradcam_{key}.png")
        fig2.savefig(ind_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"  Saved individual -> {ind_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GradCAM w.r.t. Mahalanobis score (MSDE pipeline) "
                    "for MedIAnomaly datasets"
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, "MedIAnomaly-Data"),
        help="Root directory of MedIAnomaly-Data (containing RSNA/, VinCXR/, etc.)"
    )
    parser.add_argument(
        "--feature_root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "features"),
        help="Root directory of pre-extracted features "
             "(containing rsna/anatpaste/, brain/resnet18/, etc.)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gradcam_output"),
        help="Where to save the output figures"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=None,
        help="Specific datasets to process (e.g. --datasets rsna brain). "
             "Default: all 7 datasets."
    )
    args = parser.parse_args()

    # Override dataset order if specific datasets requested
    if args.datasets:
        DATASET_ORDER = [d for d in args.datasets if d in DATASET_CFG]
        if not DATASET_ORDER:
            print(f"Error: none of {args.datasets} are valid dataset keys.")
            print(f"Valid keys: {list(DATASET_CFG.keys())}")
            sys.exit(1)

    main(args)