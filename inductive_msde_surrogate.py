"""Evaluate an inductive MLP surrogate for transductive MSDE-GDE.

For each dataset and split:
  1. Fit the normal-side MSDE-GDE reference exactly as usual.
  2. Run transductive MSDE-GDE on an unlabeled 80% teacher cohort.
  3. Train an MLP to predict those teacher scores from teacher embeddings.
  4. Score the held-out 20% with the MLP only.

Test labels are never used for splitting, training, or model selection. They
are used only after inference to calculate evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from MSDE_v3 import _GDEScorer, mean_shift_density_enhancement
from benchmark_utils import compute_metrics


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]

# Match the primary MSDE-GDE benchmark configuration.
MSDE_K = 100
MSDE_LR = 0.33
MSDE_MAX_ITERS = 8
MSDE_SHIFT_THRESH = 0.01
MSDE_NBD_THRESHOLD = 70


def load_features(feature_root: Path, dataset: str):
    path = feature_root / dataset / "biomedclip"
    train_x = np.load(path / "train_features.npy")
    train_y = np.load(path / "train_labels.npy")
    test_x = np.load(path / "test_features.npy")
    test_y = np.load(path / "test_labels.npy")
    test_names = np.load(path / "test_names.npy", allow_pickle=True)
    return train_x, train_y, test_x, test_y, test_names


def run_msde_shift(X: np.ndarray):
    return mean_shift_density_enhancement(
        X,
        k=MSDE_K,
        nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
        learning_rate=MSDE_LR,
        max_iters_shift=MSDE_MAX_ITERS,
        shift_threshold=MSDE_SHIFT_THRESH,
    )


def fit_normal_reference(train_x: np.ndarray, train_y: np.ndarray):
    train_normal = train_x[train_y == 0]
    train_shifted, _ = run_msde_shift(train_normal)
    gde = _GDEScorer().fit(train_shifted)
    return gde


def make_split(n_samples: int, seed: int, teacher_fraction: float):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)
    n_teacher = int(round(n_samples * teacher_fraction))
    teacher_indices = permutation[:n_teacher]
    heldout_indices = permutation[n_teacher:]
    return teacher_indices, heldout_indices


def fit_student(
    teacher_x: np.ndarray,
    teacher_scores: np.ndarray,
    hidden_layers: tuple[int, ...],
    max_iter: int,
    seed: int,
):
    input_scaler = StandardScaler().fit(teacher_x)
    teacher_x_scaled = input_scaler.transform(teacher_x)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        batch_size=min(64, len(teacher_x_scaled)),
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(teacher_x_scaled, teacher_scores)
    return input_scaler, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--teacher-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[128, 64],
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("results") / "results_inductive_msde_surrogate",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save each fitted MLP and input scaler with joblib",
    )
    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = DATASETS
    unknown = [dataset for dataset in args.datasets if dataset not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if not 0.0 < args.teacher_fraction < 1.0:
        raise ValueError("--teacher-fraction must be between 0.0 and 1.0")
    if args.splits < 1:
        raise ValueError("--splits must be at least 1")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        print(f"\n[{dataset}] loading BiomedCLIP features")
        train_x, train_y, test_x, test_y, test_names = load_features(
            args.feature_root, dataset
        )

        train_normal = train_x[train_y == 0]
        standardizer = StandardScaler().fit(train_normal)
        train_scaled = standardizer.transform(train_x)
        test_scaled = standardizer.transform(test_x)

        setup_started = time.perf_counter()
        gde = fit_normal_reference(train_scaled, train_y)
        setup_runtime = time.perf_counter() - setup_started

        for split_id in range(args.splits):
            split_seed = args.seed + split_id
            teacher_indices, heldout_indices = make_split(
                len(test_scaled), split_seed, args.teacher_fraction
            )
            teacher_x = test_scaled[teacher_indices]
            heldout_x = test_scaled[heldout_indices]

            teacher_started = time.perf_counter()
            teacher_shifted, _ = run_msde_shift(teacher_x)
            teacher_scores = gde.score(teacher_shifted)
            teacher_runtime = time.perf_counter() - teacher_started

            student_started = time.perf_counter()
            input_scaler, student = fit_student(
                teacher_x,
                teacher_scores,
                tuple(args.hidden_layers),
                args.max_iter,
                split_seed,
            )
            heldout_scores = student.predict(input_scaler.transform(heldout_x))
            student_runtime = time.perf_counter() - student_started

            heldout_y = test_y[heldout_indices]
            metrics = compute_metrics(heldout_y, heldout_scores)
            teacher_fit_metrics = {
                "teacher_score_rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            teacher_scores,
                            student.predict(input_scaler.transform(teacher_x)),
                        )
                    )
                ),
                "teacher_score_r2": float(
                    r2_score(
                        teacher_scores,
                        student.predict(input_scaler.transform(teacher_x)),
                    )
                ),
            }

            run_dir = args.save_dir / dataset / f"split_{split_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "name": test_names[heldout_indices],
                    "label": heldout_y,
                    "score": heldout_scores,
                    "split": "heldout_20_percent",
                }
            ).to_csv(run_dir / "heldout_scores.csv", index=False)

            if args.save_models:
                joblib.dump(input_scaler, run_dir / "input_scaler.joblib")
                joblib.dump(student, run_dir / "mlp_surrogate.joblib")

            result = {
                "method": "MSDE-GDE-MLP-surrogate",
                "dataset": dataset,
                "backbone": "biomedclip",
                "split_id": split_id,
                "seed": split_seed,
                "teacher_fraction": args.teacher_fraction,
                "n_teacher": len(teacher_indices),
                "n_heldout": len(heldout_indices),
                "mlp_hidden_layers": str(tuple(args.hidden_layers)),
                "mlp_max_iter": args.max_iter,
                "mlp_actual_iter": int(student.n_iter_),
                "setup_runtime_seconds": setup_runtime,
                "teacher_runtime_seconds": teacher_runtime,
                "student_runtime_seconds": student_runtime,
                "runtime_seconds": setup_runtime + teacher_runtime + student_runtime,
                **metrics,
                **teacher_fit_metrics,
            }
            rows.append(result)
            print(
                f"  split={split_id} heldout={len(heldout_indices)} "
                f"AUC={metrics['auc']:.5f} AP={metrics['ap']:.5f} "
                f"P@N={metrics['precision_at_n']:.5f} "
                f"teacher_RMSE={teacher_fit_metrics['teacher_score_rmse']:.5f}"
            )

    all_results = pd.DataFrame(rows)
    all_results.to_csv(args.save_dir / "all_results.csv", index=False)
    all_results.groupby("dataset").agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        ap_mean=("ap", "mean"),
        ap_std=("ap", "std"),
        precision_at_n_mean=("precision_at_n", "mean"),
        precision_at_n_std=("precision_at_n", "std"),
        teacher_score_rmse_mean=("teacher_score_rmse", "mean"),
        teacher_score_r2_mean=("teacher_score_r2", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).reset_index().to_csv(args.save_dir / "summary_by_dataset.csv", index=False)
    overall = pd.DataFrame(
        [
            {
                "auc_mean": all_results["auc"].mean(),
                "auc_std": all_results["auc"].std(),
                "ap_mean": all_results["ap"].mean(),
                "ap_std": all_results["ap"].std(),
                "precision_at_n_mean": all_results["precision_at_n"].mean(),
                "precision_at_n_std": all_results["precision_at_n"].std(),
                "teacher_score_rmse_mean": all_results["teacher_score_rmse"].mean(),
                "teacher_score_r2_mean": all_results["teacher_score_r2"].mean(),
                "runtime_mean_seconds": all_results["runtime_seconds"].mean(),
            }
        ]
    )
    overall.to_csv(args.save_dir / "summary_overall.csv", index=False)

    with (args.save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "teacher_fraction": args.teacher_fraction,
                "splits": args.splits,
                "base_seed": args.seed,
                "hidden_layers": args.hidden_layers,
                "max_iter": args.max_iter,
                "msde_k": MSDE_K,
                "msde_learning_rate": MSDE_LR,
                "msde_max_iters": MSDE_MAX_ITERS,
                "msde_shift_threshold": MSDE_SHIFT_THRESH,
                "msde_nbd_threshold": MSDE_NBD_THRESHOLD,
            },
            handle,
            indent=2,
        )

    print(f"\nSaved results to {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
