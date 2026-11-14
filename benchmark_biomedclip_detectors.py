"""Benchmark anomaly detectors using one fixed BiomedCLIP feature space.

Detectors:
  - GDE/Mahalanobis in PCA space
  - k-nearest-neighbor distance
  - Local Outlier Factor
  - Isolation Forest
  - One-Class SVM
  - MSDE-GDE
  - MSDE-distance (fully unsupervised test-only scoring)
  - MSDE-kNN (MSDE features followed by normal-reference kNN)
  - MSDE-reference (normal-reference MSDE displacement)

All detectors except MSDE-distance use normal training embeddings. The
MSDE-distance scorer uses only the test embeddings and does not fit a model
on the training set. Test labels are used only for final metric computation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.special import expit
from pynndescent import NNDescent

from MSDE_v3 import (
    _GDEScorer,
    get_empirical_weights,
    mean_shift_density_enhancement,
)
from MSDE_implementation import mean_shift_manifold_learning
from benchmark_utils import compute_metrics


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
DETECTORS = [
    "gde", "knn", "lof", "iforest", "ocsvm", "msde-gde",
    "msde-distance", "msde-knn", "msde-reference",
]

# Keep the MSDE configuration explicit so benchmark results do not depend on
# defaults that may change inside MSDE_v3.py.
MSDE_K = 100
MSDE_LR = 0.33
MSDE_MAX_ITERS = 8
MSDE_SHIFT_THRESH = 0.01
MSDE_NBD_THRESHOLD = 70

# Configuration used by the original fully unsupervised MSDE-distance
# implementation (MSDE_implementation.py). It is intentionally kept
# separate from the MSDE-GDE configuration above.
MSDE_DISTANCE_K = 100
MSDE_DISTANCE_LR = 0.1
MSDE_DISTANCE_MAX_ITERS = 6
MSDE_DISTANCE_SHIFT_THRESH = 0.003
MSDE_DISTANCE_NBD_THRESHOLD = 70


def load_features(path: Path):
    train_x = np.load(path / "train_features.npy")
    train_y = np.load(path / "train_labels.npy")
    test_x = np.load(path / "test_features.npy")
    test_y = np.load(path / "test_labels.npy")
    test_names = np.load(path / "test_names.npy", allow_pickle=True)
    return train_x, train_y, test_x, test_y, test_names


def shift_test_against_reference(
    test_x: np.ndarray,
    reference_x: np.ndarray,
    k: int,
    nbd_sample_count_threshold: int,
    learning_rate: float,
    max_iters_shift: int,
    shift_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift test points using neighbours selected only from a reference set."""
    if len(reference_x) == 0:
        raise ValueError("The MSDE reference set cannot be empty")

    n_reference_neighbors = min(k, len(reference_x))
    reference_weights = get_empirical_weights(
        reference_x,
        nbd_sample_count_threshold=nbd_sample_count_threshold,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        batch_size=1000,
    )
    reference_index = NNDescent(
        reference_x,
        n_neighbors=n_reference_neighbors,
        metric="euclidean",
        random_state=42,
    )

    shifted = test_x.copy()
    total_distance = np.zeros(len(test_x), dtype=np.float64)

    for _ in range(max_iters_shift):
        indices, _ = reference_index.query(
            shifted, k=n_reference_neighbors
        )
        neighbour_weights = reference_weights[indices]
        denominator = np.maximum(neighbour_weights.sum(axis=1), 1e-12)
        target = (
            reference_x[indices] * neighbour_weights[..., None]
        ).sum(axis=1) / denominator[:, None]

        delta = target - shifted
        change = np.linalg.norm(delta, axis=1)
        total_distance += change
        shifted = shifted + learning_rate * delta

        if change.mean() < shift_threshold:
            break

    return shifted, total_distance


def score_detector(name: str, train_x: np.ndarray, test_x: np.ndarray,
                   seed: int) -> np.ndarray:
    if name == "msde-distance":
        # Fully unsupervised: test_x is the only data used. The original
        # scorer returns cumulative MSDE displacement, which is converted
        # to the original sigmoid-normalized anomaly score.
        _, total_distance = mean_shift_manifold_learning(
            test_x,
            k=MSDE_DISTANCE_K,
            nbd_sample_count_threshold=MSDE_DISTANCE_NBD_THRESHOLD,
            learning_rate=MSDE_DISTANCE_LR,
            max_iters_shift=MSDE_DISTANCE_MAX_ITERS,
            shift_threshold=MSDE_DISTANCE_SHIFT_THRESH,
        )
        distance = StandardScaler().fit_transform(
            total_distance.reshape(-1, 1)
        )
        return expit(distance).ravel()

    if name in {"msde-knn", "msde-reference"}:
        # Shift the normal training set first and use it as the reference.
        # Unlike msde-distance, these variants use normal training data.
        train_shifted, _ = mean_shift_density_enhancement(
            train_x,
            k=MSDE_K,
            nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
            learning_rate=MSDE_LR,
            max_iters_shift=MSDE_MAX_ITERS,
            shift_threshold=MSDE_SHIFT_THRESH,
        )

        if name == "msde-reference":
            _, total_distance = shift_test_against_reference(
                test_x,
                train_shifted,
                k=MSDE_K,
                nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
                learning_rate=MSDE_LR,
                max_iters_shift=MSDE_MAX_ITERS,
                shift_threshold=MSDE_SHIFT_THRESH,
            )
            distance = StandardScaler().fit_transform(
                total_distance.reshape(-1, 1)
            )
            return expit(distance).ravel()

        test_shifted, _ = mean_shift_density_enhancement(
            test_x,
            k=MSDE_K,
            nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
            learning_rate=MSDE_LR,
            max_iters_shift=MSDE_MAX_ITERS,
            shift_threshold=MSDE_SHIFT_THRESH,
        )
        n_neighbors = min(5, len(train_shifted))
        model = NearestNeighbors(n_neighbors=n_neighbors).fit(train_shifted)
        distances, _ = model.kneighbors(test_shifted)
        return distances.mean(axis=1)

    if name == "gde":
        model = _GDEScorer().fit(train_x)
        return model.score(test_x)

    if name == "knn":
        n_neighbors = min(5, len(train_x))
        model = NearestNeighbors(n_neighbors=n_neighbors).fit(train_x)
        distances, _ = model.kneighbors(test_x)
        return distances.mean(axis=1)

    if name == "lof":
        n_neighbors = min(20, max(2, len(train_x) - 1))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors, novelty=True, contamination="auto"
        )
        model.fit(train_x)
        return -model.score_samples(test_x)

    if name == "iforest":
        model = IsolationForest(random_state=seed, contamination="auto")
        model.fit(train_x)
        return -model.score_samples(test_x)

    if name == "ocsvm":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        model.fit(train_x)
        return -model.decision_function(test_x)

    if name == "msde-gde":
        train_shifted, _ = mean_shift_density_enhancement(
            train_x,
            k=MSDE_K,
            nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
            learning_rate=MSDE_LR,
            max_iters_shift=MSDE_MAX_ITERS,
            shift_threshold=MSDE_SHIFT_THRESH,
        )
        test_shifted, _ = mean_shift_density_enhancement(
            test_x,
            k=MSDE_K,
            nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
            learning_rate=MSDE_LR,
            max_iters_shift=MSDE_MAX_ITERS,
            shift_threshold=MSDE_SHIFT_THRESH,
        )
        model = _GDEScorer().fit(train_shifted)
        return model.score(test_shifted)

    raise ValueError(f"Unknown detector: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--backbone", default="biomedclip")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--detectors", nargs="+", default=DETECTORS)
    parser.add_argument("--save-dir", type=Path,
                        default=Path("results") / "results_biomedclip_detectors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = DATASETS
    if args.detectors == ["all"]:
        args.detectors = DETECTORS

    unknown_datasets = [x for x in args.datasets if x not in DATASETS]
    unknown_detectors = [x for x in args.detectors if x not in DETECTORS]
    if unknown_datasets or unknown_detectors:
        raise ValueError(
            f"Unknown datasets={unknown_datasets}, detectors={unknown_detectors}"
        )

    np.random.seed(args.seed)
    rows = []

    for dataset in args.datasets:
        feature_dir = args.feature_root / dataset / args.backbone
        if not feature_dir.exists():
            print(f"[SKIP] missing feature directory: {feature_dir}")
            continue

        train_x, train_y, test_x, test_y, test_names = load_features(feature_dir)
        train_x = train_x[train_y == 0]
        test_x_raw = test_x.copy()

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        for detector in args.detectors:
            print(f"\n[{detector}] {dataset}")
            started = time.perf_counter()
            detector_test_x = test_x_raw if detector == "msde-distance" else test_x
            detector_train_x = (
                np.empty((0, test_x.shape[1]), dtype=test_x.dtype)
                if detector == "msde-distance" else train_x
            )
            scores = score_detector(
                detector, detector_train_x, detector_test_x, args.seed
            )
            runtime = time.perf_counter() - started
            metrics = compute_metrics(test_y, scores)

            run_dir = args.save_dir / detector / dataset
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "name": test_names,
                "label": test_y,
                "score": scores,
            }).to_csv(run_dir / "test_scores.csv", index=False)

            result = {
                "method": detector,
                "dataset": dataset,
                "backbone": args.backbone,
                "seed": args.seed,
                "runtime_seconds": runtime,
                "runtime_scope": "feature_detector_only",
                **metrics,
            }
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            rows.append(result)
            print(
                f"AUC={metrics['auc']:.5f} AP={metrics['ap']:.5f} "
                f"P@N={metrics['precision_at_n']:.5f} "
                f"time={runtime:.2f}s"
            )

    frame = pd.DataFrame(rows)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.save_dir / "all_results.csv", index=False)
    frame.groupby("method").agg(
        auc_mean=("auc", "mean"),
        ap_mean=("ap", "mean"),
        precision_at_n_mean=("precision_at_n", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).reset_index().to_csv(args.save_dir / "summary_by_method.csv", index=False)
    frame.groupby("dataset").agg(
        auc_mean=("auc", "mean"),
        ap_mean=("ap", "mean"),
        precision_at_n_mean=("precision_at_n", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).reset_index().to_csv(args.save_dir / "summary_by_dataset.csv", index=False)
    print(f"\nSaved results to {args.save_dir}")


if __name__ == "__main__":
    main()
