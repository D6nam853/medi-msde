"""Order-sensitivity test for transductive MSDE-GDE.

The MSDE test transformation is applied to several orderings of the same
unlabelled test cohort:

  - random_0 ... random_{n-1}: seeded random permutations;
  - label_ascending / label_descending: diagnostic stress tests only.

For every run, scores are mapped back to the original test-sample order before
metrics are computed. Labels are used only to construct the two diagnostic
orders and to compute final metrics; they are never passed to MSDE or GDE.

Example:
  python test_order_permutation.py --datasets all --n-permutations 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from MSDE_v3 import _GDEScorer, mean_shift_density_enhancement
from benchmark_utils import compute_metrics


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]

# Keep this protocol explicit and synchronized with the final benchmark.
MSDE_K = 100
MSDE_LR = 0.33
MSDE_MAX_ITERS = 8
MSDE_SHIFT_THRESH = 0.01
MSDE_NBD_THRESHOLD = 70


def load_features(feature_dir: Path):
    train_x = np.load(feature_dir / "train_features.npy")
    train_y = np.load(feature_dir / "train_labels.npy")
    test_x = np.load(feature_dir / "test_features.npy")
    test_y = np.load(feature_dir / "test_labels.npy")
    test_names = np.load(feature_dir / "test_names.npy", allow_pickle=True)
    return train_x, train_y, test_x, test_y, test_names


def shift(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return x.copy()
    return mean_shift_density_enhancement(
        x,
        k=min(MSDE_K, len(x) - 1),
        nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
        learning_rate=MSDE_LR,
        max_iters_shift=MSDE_MAX_ITERS,
        shift_threshold=MSDE_SHIFT_THRESH,
    )[0]


def build_orders(labels: np.ndarray, n_permutations: int, seed: int):
    rng = np.random.default_rng(seed)
    orders = []
    for index in range(n_permutations):
        orders.append((f"random_{index}", rng.permutation(len(labels)), "random"))

    ascending = np.argsort(labels, kind="stable")
    descending = ascending[::-1]
    orders.append(("label_ascending", ascending, "label_diagnostic"))
    orders.append(("label_descending", descending, "label_diagnostic"))
    return orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--backbone", default="biomedclip")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path,
                        default=Path("results") / "results_order_permutation")
    args = parser.parse_args()

    datasets = DATASETS if args.datasets == ["all"] else args.datasets
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1")

    rows = []
    for dataset in tqdm(datasets, desc="Datasets", unit="dataset"):
        feature_dir = args.feature_root / dataset / args.backbone
        if not feature_dir.exists():
            print(f"[SKIP] missing feature directory: {feature_dir}")
            continue

        train_x, train_y, test_x, test_y, test_names = load_features(feature_dir)
        train_x = train_x[train_y == 0]
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        # The reference model is fixed across all ordering conditions.
        train_shifted = shift(train_x)
        gde = _GDEScorer().fit(train_shifted)
        orders = build_orders(test_y, args.n_permutations, args.seed)

        for condition, order, order_type in tqdm(
            orders, desc=dataset, unit="order", leave=False
        ):
            started = time.perf_counter()
            test_permuted = test_x[order]
            shifted_permuted = shift(test_permuted)
            scores_permuted = gde.score(shifted_permuted)

            # Restore scores to the original test-file order. This ensures
            # metrics compare the same image-label pairs in every condition.
            scores_original = np.empty_like(scores_permuted)
            scores_original[order] = scores_permuted
            runtime = time.perf_counter() - started
            metrics = compute_metrics(test_y, scores_original)

            run_dir = args.save_dir / condition / dataset
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "name": test_names,
                "label": test_y,
                "score": scores_original,
            }).to_csv(run_dir / "test_scores.csv", index=False)
            pd.DataFrame({
                "original_index": np.arange(len(order)),
                "permuted_index": order,
                "label_used_only_for_diagnostic_order": test_y[order],
            }).to_csv(run_dir / "permutation.csv", index=False)

            result = {
                "method": "MSDE-GDE-transductive",
                "experiment": "test_order_permutation",
                "condition": condition,
                "order_type": order_type,
                "dataset": dataset,
                "backbone": args.backbone,
                "seed": args.seed,
                "n_test_samples": int(len(test_y)),
                "runtime_seconds": runtime,
                "runtime_scope": "feature_detector_only",
                **metrics,
            }
            with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            rows.append(result)

    frame = pd.DataFrame(rows)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.save_dir / "all_results.csv", index=False)

    if not frame.empty:
        frame.groupby("order_type").agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            precision_at_n_mean=("precision_at_n", "mean"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
        ).reset_index().to_csv(args.save_dir / "summary_by_order_type.csv", index=False)

        random_frame = frame[frame["order_type"] == "random"]
        random_summary = random_frame.groupby("dataset").agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            precision_at_n_mean=("precision_at_n", "mean"),
            precision_at_n_std=("precision_at_n", "std"),
        ).reset_index()
        random_summary.to_csv(args.save_dir / "summary_random_permutations.csv", index=False)

        frame.groupby(["dataset", "condition", "order_type"]).agg(
            auc=("auc", "mean"),
            ap=("ap", "mean"),
            precision_at_n=("precision_at_n", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        ).reset_index().to_csv(args.save_dir / "summary_by_condition.csv", index=False)

    print(f"\nSaved order-permutation results to {args.save_dir}")


if __name__ == "__main__":
    main()
