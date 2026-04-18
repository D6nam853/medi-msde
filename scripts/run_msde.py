import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from msde.msde import MSDE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. rsna, brain, vin, lag, brats, c16")
    parser.add_argument("--feature-root", type=str, default="features",
                        help="Root folder where extracted features are stored")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone feature folder name")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="PCA dimension; set 0 to disable PCA")
    parser.add_argument("--seed", type=int, default=42)

    # MSDE hyperparameters
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--nbd-sample-count-threshold", type=int, default=70)
    parser.add_argument("--learning-rate", type=float, default=0.33)
    parser.add_argument("--max-iters-shift", type=int, default=8)
    parser.add_argument("--shift-threshold", type=float, default=0.01)

    parser.add_argument("--save-dir", type=str, default="results",
                        help="Root directory to save results")
    return parser.parse_args()


def load_features(feature_dir):
    train_features = np.load(os.path.join(feature_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(feature_dir, "train_labels.npy"))
    train_names = np.load(os.path.join(feature_dir, "train_names.npy"), allow_pickle=True)

    test_features = np.load(os.path.join(feature_dir, "test_features.npy"))
    test_labels = np.load(os.path.join(feature_dir, "test_labels.npy"))
    test_names = np.load(os.path.join(feature_dir, "test_names.npy"), allow_pickle=True)

    return train_features, train_labels, train_names, test_features, test_labels, test_names


def upsert_global_csv(global_csv_path, row_dict, key_columns):
    new_row_df = pd.DataFrame([row_dict])

    if os.path.exists(global_csv_path):
        old_df = pd.read_csv(global_csv_path)

        for col in new_row_df.columns:
            if col not in old_df.columns:
                old_df[col] = np.nan
        for col in old_df.columns:
            if col not in new_row_df.columns:
                new_row_df[col] = np.nan

        old_df = old_df[new_row_df.columns]

        mask = pd.Series(True, index=old_df.index)
        for col in key_columns:
            old_vals = old_df[col].astype(str).fillna("")
            new_val = str(new_row_df.iloc[0][col]) if col in new_row_df.columns else ""
            mask &= (old_vals == new_val)

        if mask.any():
            old_df.loc[mask, :] = new_row_df.iloc[0].values
            final_df = old_df
            print("Updated existing row in all_runs.csv")
        else:
            final_df = pd.concat([old_df, new_row_df], ignore_index=True)
            print("Added new row to all_runs.csv")
    else:
        final_df = new_row_df
        print("Created all_runs.csv")

    final_df.to_csv(global_csv_path, index=False)


def main():
    args = parse_args()

    np.random.seed(args.seed)

    feature_dir = os.path.join(args.feature_root, args.dataset, args.backbone)
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    print(f"Loading features from: {feature_dir}")
    X_train, y_train, train_names, X_test, y_test, test_names = load_features(feature_dir)

    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape : {X_test.shape}")
    print(f"Unique train labels : {np.unique(y_train)}")
    print(f"Unique test labels  : {np.unique(y_test)}")

    # Use only normal training samples
    X_train_normal = X_train[y_train == 0]
    train_names_normal = train_names[y_train == 0]
    print(f"Training on normal samples only: {X_train_normal.shape[0]}")

    # Standardize using only normal training samples
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test)

    # Optional PCA
    if args.pca_dim > 0:
        pca_dim = min(args.pca_dim, X_train_scaled.shape[1], X_train_scaled.shape[0])
        print(f"Applying PCA: {X_train_scaled.shape[1]} -> {pca_dim}")
        pca = PCA(n_components=pca_dim, random_state=args.seed)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        print("PCA disabled")
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    print(f"Final train shape: {X_train_final.shape}")
    print(f"Final test shape : {X_test_final.shape}")

    detector = MSDE(
        seed=args.seed,
        model_name="MSDE",
        k=args.k,
        nbd_sample_count_threshold=args.nbd_sample_count_threshold,
        learning_rate=args.learning_rate,
        max_iters_shift=args.max_iters_shift,
        shift_threshold=args.shift_threshold,
        scaler=StandardScaler()
    )

    print("Fitting MSDE...")
    detector.fit(X_train_final)

    print("Scoring train samples...")
    train_scores = detector.predict_score(X_train_final)

    print("Scoring test samples...")
    test_scores = detector.predict_score(X_test_final)

    auc = roc_auc_score(y_test, test_scores)
    ap = average_precision_score(y_test, test_scores)

    print("\n===== MSDE Results =====")
    print(f"Dataset : {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"AUC     : {auc:.5f}")
    print(f"AP      : {ap:.5f}")

    save_dir = os.path.join(args.save_dir, args.dataset, args.backbone, "msde")
    os.makedirs(save_dir, exist_ok=True)

    result_dict = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": "msde",
        "seed": args.seed,
        "pca_dim": args.pca_dim,
        "k": args.k,
        "nbd_sample_count_threshold": args.nbd_sample_count_threshold,
        "learning_rate": args.learning_rate,
        "max_iters_shift": args.max_iters_shift,
        "shift_threshold": args.shift_threshold,
        "auc": float(auc),
        "ap": float(ap),
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

    pd.DataFrame([result_dict]).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    train_scores_df = pd.DataFrame({
        "name": train_names_normal,
        "label": np.zeros(len(train_names_normal), dtype=int),
        "score": train_scores
    })
    train_scores_df.to_csv(os.path.join(save_dir, "train_scores.csv"), index=False)

    test_scores_df = pd.DataFrame({
        "name": test_names,
        "label": y_test,
        "score": test_scores
    })
    test_scores_df.to_csv(os.path.join(save_dir, "test_scores.csv"), index=False)

    global_csv_path = os.path.join(args.save_dir, "all_runs.csv")
    key_columns = [
        "dataset", "backbone", "method", "seed", "pca_dim",
        "k", "nbd_sample_count_threshold", "learning_rate",
        "max_iters_shift", "shift_threshold"
    ]
    upsert_global_csv(global_csv_path, result_dict, key_columns)

    print(f"\nSaved results to: {save_dir}")
    print(f"Updated global summary: {global_csv_path}")


if __name__ == "__main__":
    main()
