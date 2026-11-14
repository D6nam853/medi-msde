"""Ablate MSDE inference protocols on fixed BiomedCLIP embeddings.

Protocols:
  no_shift       : GDE directly on standardized embeddings.
  train_only     : fit the shift field on normal training data and transform
                   test queries using only that fixed training field.
  shifted_train_raw_test : shift normal training data, but score raw test data.
  raw_train_shifted_test : fit GDE on raw training data, but shift the test cohort.
  transductive   : shift the normal train set and the complete unlabeled test
                   cohort separately; this is the current MSDE-GDE protocol.

No test labels are used for fitting or shifting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent

from MSDE_v3 import (
    _GDEScorer,
    get_empirical_weights,
    mean_shift_density_enhancement,
)
from benchmark_utils import compute_metrics


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
PROTOCOLS = [
    "no_shift",
    "train_only",
    "shifted_train_raw_test",
    "raw_train_shifted_test",
    "transductive",
]

MSDE_K = 100
MSDE_LR = 0.33
MSDE_MAX_ITERS = 8
MSDE_SHIFT_THRESH = 0.01
MSDE_NBD_THRESHOLD = 70


def load_features(path: Path):
    train_x = np.load(path / "train_features.npy")
    train_y = np.load(path / "train_labels.npy")
    test_x = np.load(path / "test_features.npy")
    test_y = np.load(path / "test_labels.npy")
    test_names = np.load(path / "test_names.npy", allow_pickle=True)
    return train_x, train_y, test_x, test_y, test_names


def shift_train(x_train: np.ndarray):
    return mean_shift_density_enhancement(
        x_train,
        k=MSDE_K,
        nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
        learning_rate=MSDE_LR,
        max_iters_shift=MSDE_MAX_ITERS,
        shift_threshold=MSDE_SHIFT_THRESH,
    )[0]


def shift_queries_using_train_field(
    queries: np.ndarray,
    train_raw: np.ndarray,
    train_shifted: np.ndarray,
) -> np.ndarray:
    """Transform queries using a fixed field learned only from train normals."""
    weights = get_empirical_weights(
        train_raw,
        nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        n_neighbors=15,
        batch_size=1000,
    )
    n_neighbors = min(MSDE_K, len(train_shifted))
    reference = train_shifted.copy()
    shifted = queries.copy()
    # Match the transductive MSDE implementation: use NN-Descent rather than
    # exact sklearn nearest neighbors for the train-only comparison.
    index = NNDescent(
        reference,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=42,
    )

    for _ in range(MSDE_MAX_ITERS):
        indices, _ = index.query(shifted, k=n_neighbors)
        local_weights = weights[indices]
        denom = np.maximum(local_weights.sum(axis=1, keepdims=True), 1e-6)
        centroids = (
            local_weights[:, :, None] * reference[indices]
        ).sum(axis=1) / denom
        delta = centroids - shifted
        distances = np.linalg.norm(delta, axis=1)
        step = np.minimum(MSDE_LR, 1.0) * delta
        shifted = shifted + step
        if distances.mean() < MSDE_SHIFT_THRESH:
            break

    return shifted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--backbone", default="biomedclip")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    parser.add_argument("--save-dir", type=Path,
                        default=Path("results") / "results_msde_ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = DATASETS
    if args.protocols == ["all"]:
        args.protocols = PROTOCOLS
    if any(x not in DATASETS for x in args.datasets):
        raise ValueError(f"Unknown dataset in {args.datasets}")
    if any(x not in PROTOCOLS for x in args.protocols):
        raise ValueError(f"Unknown protocol in {args.protocols}")

    np.random.seed(args.seed)
    rows = []

    for dataset in args.datasets:
        feature_dir = args.feature_root / dataset / args.backbone
        if not feature_dir.exists():
            print(f"[SKIP] missing feature directory: {feature_dir}")
            continue

        train_x, train_y, test_x, test_y, test_names = load_features(feature_dir)
        train_x = train_x[train_y == 0]
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        for protocol in args.protocols:
            print(f"\n[{protocol}] {dataset}")
            started = time.perf_counter()

            if protocol == "no_shift":
                train_final, test_final = train_x, test_x
            elif protocol == "train_only":
                train_final = shift_train(train_x)
                test_final = shift_queries_using_train_field(
                    test_x, train_x, train_final
                )
            elif protocol == "shifted_train_raw_test":
                train_final, test_final = shift_train(train_x), test_x
            elif protocol == "raw_train_shifted_test":
                test_shifted = mean_shift_density_enhancement(
                    test_x,
                    k=MSDE_K,
                    nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
                    learning_rate=MSDE_LR,
                    max_iters_shift=MSDE_MAX_ITERS,
                    shift_threshold=MSDE_SHIFT_THRESH,
                )[0]
                train_final, test_final = train_x, test_shifted
            elif protocol == "transductive":
                train_final = shift_train(train_x)
                test_final = mean_shift_density_enhancement(
                    test_x,
                    k=MSDE_K,
                    nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
                    learning_rate=MSDE_LR,
                    max_iters_shift=MSDE_MAX_ITERS,
                    shift_threshold=MSDE_SHIFT_THRESH,
                )[0]
            else:
                raise ValueError(f"Unknown protocol: {protocol}")

            scores = _GDEScorer().fit(train_final).score(test_final)
            runtime = time.perf_counter() - started
            metrics = compute_metrics(test_y, scores)

            run_dir = args.save_dir / protocol / dataset
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "name": test_names,
                "label": test_y,
                "score": scores,
            }).to_csv(run_dir / "test_scores.csv", index=False)

            result = {
                "method": f"MSDE-GDE-{protocol}",
                "protocol": protocol,
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
                f"P@N={metrics['precision_at_n']:.5f} time={runtime:.2f}s"
            )

    frame = pd.DataFrame(rows)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.save_dir / "all_results.csv", index=False)
    frame.groupby("protocol").agg(
        auc_mean=("auc", "mean"),
        ap_mean=("ap", "mean"),
        precision_at_n_mean=("precision_at_n", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).reset_index().to_csv(args.save_dir / "summary_by_protocol.csv", index=False)
    print(f"\nSaved ablation results to {args.save_dir}")


if __name__ == "__main__":
    main()
