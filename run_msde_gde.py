"""
run_msde_gde.py
───────────────
Inductive one-class scoring: MSDE (train-only) + GDE.

Pipeline (per dataset × backbone):
  1. Load pre-extracted embeddings from disk.
  2. Standardise on normal train samples only; apply to test.
  3. Run mean-shift on normal train embeddings only.
  4. Fit GDE (PCA-256 + Mahalanobis) on the shifted normal embeddings.
  5. Shift the test batch using a test-derived MSDE neighborhood field.
  6. Score shifted test samples directly via the fitted GDE.
     → Transductive: test samples influence their MSDE transformation, but
       test labels are never used.
  7. Evaluate AUC / AP. Save results.

Usage
─────
python run_msde_gde.py \
    --feature-root features \
    --save-dir results/results_msde_gde \
    [--seed 42]
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from MSDE_v3 import mean_shift_density_enhancement, _GDEScorer


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATASETS  = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
# BACKBONES = ["resnet18", "anatpaste", "biomedclip"]
BACKBONES = ["biomedclip"]
# MSDE hyper-parameters (identical to run_msde.py)
MSDE_K             = 100
MSDE_LR            = 0.33
MSDE_MAX_ITERS     = 8
MSDE_SHIFT_THRESH  = 0.01
MSDE_NBD_THRESHOLD = 70


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_features(feature_dir: str) -> tuple:
    def _load(prefix):
        feats  = np.load(os.path.join(feature_dir, f"{prefix}_features.npy"))
        labels = np.load(os.path.join(feature_dir, f"{prefix}_labels.npy"))
        names  = np.load(os.path.join(feature_dir, f"{prefix}_names.npy"),
                         allow_pickle=True)
        return feats, labels, names

    return (*_load("train"), *_load("test"))


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.0, "ap": 0.0, "precision_at_n": 0.0,
                "n_anomalies": int(np.sum(y_true == 1))}

    n_anomalies = int(np.sum(y_true == 1))
    if n_anomalies == 0:
        precision_at_n = 0.0
    else:
        top_n = np.argsort(scores)[-n_anomalies:]
        precision_at_n = float(np.mean(y_true[top_n] == 1))

    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ap":  float(average_precision_score(y_true, scores)),
        "precision_at_n": precision_at_n,
        "n_anomalies": n_anomalies,
    }


def upsert_global_csv(path: str, row: dict, keys: list[str]) -> None:
    new_df = pd.DataFrame([row])
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = np.nan
        old_df = old_df.reindex(columns=new_df.columns)

        mask = pd.Series(True, index=old_df.index)
        for k in keys:
            mask &= old_df[k].astype(str).fillna("") == str(new_df.iloc[0][k])

        final = old_df.copy()
        if mask.any():
            final.loc[mask] = new_df.iloc[0].values
        else:
            final = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final = new_df

    final.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Transductive MSDE-GDE: train/test MSDE fields, no test labels"
    )
    p.add_argument("--feature-root", type=str, default="features",
                   help="Root folder containing extracted embeddings")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=DATASETS,
                   help="Datasets to process (default: all)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--save-dir",     type=str, default=os.path.join("results", "results_msde_gde"),
                   help="Root directory to write results")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    global_csv  = os.path.join(args.save_dir, "all_runs.csv")
    key_columns = ["dataset", "backbone", "seed"]

    all_rows = []

    for dataset in args.datasets:
        for backbone in BACKBONES:

            feature_dir = os.path.join(args.feature_root, dataset, backbone)
            if not os.path.exists(feature_dir):
                print(f"[SKIP] {dataset}/{backbone} — not found: {feature_dir}")
                continue

            print(f"\n{'='*60}")
            print(f"  Dataset : {dataset}  |  Backbone : {backbone}")
            print(f"{'='*60}")

            # ── Load embeddings ───────────────────────────────────────
            X_train_all, y_train_all, _, X_test, y_test, test_names = \
                load_features(feature_dir)

            print(f"  Train shape : {X_train_all.shape}")
            print(f"  Test  shape : {X_test.shape}")

            # One-class: keep only normal training samples
            X_train_normal = X_train_all[y_train_all == 0]
            print(f"  Normal train samples : {len(X_train_normal)}")

            # ── Standardise on normal train; apply same scaler to test ─
            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_normal)
            X_test_scaled  = scaler.transform(X_test)

            t0 = time.perf_counter()

            # ── Step 1: Shift normal train embeddings ─────────────────
            print("  Mean-shifting normal train embeddings ...")
            X_train_shifted, _ = mean_shift_density_enhancement(
                X_train_scaled,
                k=MSDE_K,
                nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
                learning_rate=MSDE_LR,
                max_iters_shift=MSDE_MAX_ITERS,
                shift_threshold=MSDE_SHIFT_THRESH,
            )

            # ── Step 2: Fit GDE on shifted normal embeddings ──────────
            print("  Fitting GDE (PCA-256 + Mahalanobis) ...")
            gde = _GDEScorer()
            gde.fit(X_train_shifted)

            # ── Step 3: Shift test embeddings independently ───────────
            print("  Mean-shifting test embeddings ...")
            X_test_shifted, _ = mean_shift_density_enhancement(
                X_test_scaled,
                k=MSDE_K,
                nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
                learning_rate=MSDE_LR,
                max_iters_shift=MSDE_MAX_ITERS,
                shift_threshold=MSDE_SHIFT_THRESH,
            )

            # ── Step 4: Score test via GDE ────────────────────────────
            test_scores = gde.score(X_test_shifted)
            elapsed     = time.perf_counter() - t0

            metrics = compute_metrics(y_test, test_scores)

            print(f"\n  AUC : {metrics['auc']:.5f}")
            print(f"  AP  : {metrics['ap']:.5f}")
            print(f"  P@N : {metrics['precision_at_n']:.5f} (N={metrics['n_anomalies']})")
            print(f"  Time: {elapsed:.2f}s")

            # ── Save per-run artefacts ────────────────────────────────
            run_dir = os.path.join(args.save_dir, dataset, backbone)
            os.makedirs(run_dir, exist_ok=True)

            result = {
                "dataset":         dataset,
                "backbone":        backbone,
                "method":          "MSDE-GDE (transductive)",
                "seed":            args.seed,
                "auc":             metrics["auc"],
                "ap":              metrics["ap"],
                "precision_at_n": metrics["precision_at_n"],
                "n_anomalies":    metrics["n_anomalies"],
                "runtime_seconds": elapsed,
            }

            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(result, f, indent=4)

            pd.DataFrame({
                "name":  test_names,
                "label": y_test,
                "score": test_scores,
            }).to_csv(os.path.join(run_dir, "test_scores.csv"), index=False)

            all_rows.append(result)
            upsert_global_csv(global_csv, result, key_columns)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    summary = pd.DataFrame(all_rows)
    print(summary[["dataset", "backbone", "auc", "ap", "precision_at_n",
                   "runtime_seconds"]].to_string(index=False))

    summary.to_csv(os.path.join(args.save_dir, "summary.csv"), index=False)
    print(f"\nAll results : {args.save_dir}/summary.csv")
    print(f"Global CSV  : {global_csv}")


if __name__ == "__main__":
    main()
