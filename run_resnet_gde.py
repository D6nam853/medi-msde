"""Standalone ResNet18-IN / ResNet152-IN baselines for the benchmark.

Protocol from MedIAnomaly:
  frozen ImageNet ResNet -> L2-normalized embeddings -> GDE/Mahalanobis score.

The command includes feature extraction and GDE fitting in its wall-clock
runtime, allowing a fair end-to-end comparison with the other benchmark
methods.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reconstruction"))
from dataloaders.dataload import (  # noqa: E402
    BraTSAD,
    Camelyon16AD,
    ISIC2018,
    MedAD,
)


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
BACKBONES = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet152": models.ResNet152_Weights.DEFAULT,
}


class GrayToRGB:
    def __call__(self, image):
        return image.convert("RGB")


def data_path(dataset: str) -> str:
    root = r"C:\Users\user\Pritam\MediAnomaly\MedIAnomaly-Data"
    mapping = {
        "rsna": "RSNA",
        "vin": "VinCXR",
        "brain": "BrainTumor",
        "lag": "LAG",
        "brats": "BraTS2021",
        "c16": "Camelyon16",
        "isic": "ISIC2018_Task3",
    }
    return os.path.join(root, mapping[dataset])


def dataset_class(dataset: str):
    if dataset in {"rsna", "vin", "brain", "lag"}:
        return MedAD
    if dataset == "brats":
        return BraTSAD
    if dataset == "c16":
        return Camelyon16AD
    if dataset == "isic":
        return ISIC2018
    raise ValueError(f"Unknown dataset: {dataset}")


def build_dataset(dataset: str, mode: str, transform):
    return dataset_class(dataset)(
        main_path=data_path(dataset),
        img_size=224,
        transform=transform,
        mode=mode,
    )


def build_backbone(name: str, device: torch.device) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=BACKBONES[name])
    elif name == "resnet152":
        model = models.resnet152(weights=BACKBONES[name])
    else:
        raise ValueError(f"Unknown backbone: {name}")
    model.fc = nn.Identity()
    return model.to(device).eval()


@torch.no_grad()
def extract(model, loader, device, desc: str):
    features, labels, names = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["img"].to(device, non_blocking=True)
        output = model(images).flatten(1)
        features.append(output.cpu().numpy())
        labels.append(np.asarray(batch["label"]))
        names.extend(list(batch["name"]))
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0).astype(int),
        np.asarray(names),
    )


class GaussianDensity:
    def fit(self, embeddings: np.ndarray):
        self.mean = embeddings.mean(axis=0)
        self.precision = LedoitWolf().fit(embeddings).precision_
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        delta = embeddings - self.mean
        squared = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
        return np.sqrt(np.clip(squared, 0.0, None))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    n_anomalies = int(np.sum(labels == 1))
    top_n = np.argsort(scores)[-n_anomalies:] if n_anomalies else []
    precision_at_n = float(np.mean(labels[top_n] == 1)) if n_anomalies else 0.0
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "precision_at_n": precision_at_n,
        "n_anomalies": n_anomalies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--backbone", required=True, choices=list(BACKBONES))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    started = time.perf_counter()
    print(f"Using {device}; backbone={args.backbone}; dataset={args.dataset}")

    transform = transforms.Compose([
        GrayToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    train_set = build_dataset(args.dataset, "train", transform)
    test_set = build_dataset(args.dataset, "test", transform)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=torch.cuda.is_available(),
    )

    model = build_backbone(args.backbone, device)
    train_features, train_labels, _ = extract(
        model, train_loader, device, "Extracting train features"
    )
    test_features, test_labels, test_names = extract(
        model, test_loader, device, "Extracting test features"
    )

    train_features = train_features[train_labels == 0]
    # Match the two-stage GDE protocol: density is fitted on normalized
    # frozen features and test features are normalized with the same operation.
    train_norm = train_features / np.maximum(
        np.linalg.norm(train_features, axis=1, keepdims=True), 1e-12
    )
    test_norm = test_features / np.maximum(
        np.linalg.norm(test_features, axis=1, keepdims=True), 1e-12
    )
    scores = GaussianDensity().fit(train_norm).score(test_norm)
    result_metrics = metrics(test_labels, scores)
    runtime = time.perf_counter() - started

    run_dir = args.save_dir / args.backbone / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "name": test_names,
        "label": test_labels,
        "score": scores,
    }).to_csv(run_dir / "test_scores.csv", index=False)
    result = {
        "method": f"ResNet{args.backbone.replace('resnet', '')}-IN",
        "dataset": args.dataset,
        "backbone": args.backbone,
        "runtime_seconds": runtime,
        "runtime_scope": "feature_extraction_plus_gde",
        **result_metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(
        f"AUC={result['auc']:.5f} AP={result['ap']:.5f} "
        f"P@N={result['precision_at_n']:.5f} time={runtime:.2f}s"
    )


if __name__ == "__main__":
    main()
