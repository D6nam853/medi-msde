"""
visualize_msde_shifted_only.py
------------------------------
Plots t-SNE / UMAP of the MSDE-shifted embeddings only.
Designed to sit side-by-side with visualize_embeddings.py output in a paper.

All visual parameters (colours, markers, font, DPI, figure size) are
intentionally identical to visualize_embeddings.py so the two plots
are directly comparable.

Four groups plotted:
  ● Normal   train  (shifted)   — teal  circle
  ● Anomaly  train  (shifted)   — coral circle   [if any exist]
  ● Normal   test   (shifted)   — teal  triangle
  ● Anomaly  test   (shifted)   — coral triangle

Usage
-----
python visualize_msde_shifted_only.py --dataset rsna --backbone resnet18 --method tsne
python visualize_msde_shifted_only.py --dataset brats --backbone anatpaste --method umap
python visualize_msde_shifted_only.py --dataset isic  --backbone resnet18 --method both
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from msde.msde import MSDE, mean_shift_density_enhancement

# ── global style — identical to visualize_embeddings.py ──────────────────────
plt.rcParams.update({
    "font.size":       12,
    "font.family":     "serif",
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize":  9,
})

# ── colour / marker scheme — identical to visualize_embeddings.py ─────────────
SPLIT_STYLE = {
    "train": dict(marker="o", alpha=0.7,  s=35, edgecolors="none"),
    "test":  dict(marker="^", alpha=0.85, s=40, edgecolors="none"),
}
CLASS_COLOR = {0: "#1D9E75", 1: "#D85A30"}
CLASS_LABEL = {0: "Normal",  1: "Anomaly"}


# ── patched MSDE that stores shifted arrays ───────────────────────────────────
class MSDEWithShift(MSDE):
    def fit(self, X_train, y_train=None):
        self.X_train_ref = np.asarray(X_train).copy()
        X_shifted, _ = mean_shift_density_enhancement(
            self.X_train_ref,
            k=self.k,
            nbd_sample_count_threshold=self.nbd_sample_count_threshold,
            learning_rate=self.learning_rate,
            max_iters_shift=self.max_iters_shift,
            shift_threshold=self.shift_threshold,
        )
        self.X_train_shifted_ = X_shifted

        from msde import _GDEScorer
        self._gde = _GDEScorer().fit(X_shifted)
        return self

    def predict_score(self, X):
        from scipy.special import expit
        X = np.asarray(X)
        X_all   = np.vstack([self.X_train_ref, X])
        n_train = len(self.X_train_ref)

        X_shifted_all, _ = mean_shift_density_enhancement(
            X_all,
            k=self.k,
            nbd_sample_count_threshold=self.nbd_sample_count_threshold,
            learning_rate=self.learning_rate,
            max_iters_shift=self.max_iters_shift,
            shift_threshold=self.shift_threshold,
        )
        self.X_joint_shifted_ = X_shifted_all
        self.n_train_          = n_train

        X_shifted_test = X_shifted_all[n_train:]
        gde_scores     = self._gde.score(X_shifted_test)
        scores = self.scaler.fit_transform(gde_scores.reshape(-1, 1))
        return expit(scores).squeeze()


# ── data helpers ──────────────────────────────────────────────────────────────
def load_features(feature_dir):
    def _l(name):
        return np.load(os.path.join(feature_dir, name), allow_pickle=True)
    X_train = _l("train_features.npy")
    y_train = _l("train_labels.npy").astype(int).flatten()
    X_test  = _l("test_features.npy")
    y_test  = _l("test_labels.npy").astype(int).flatten()
    return X_train, y_train, X_test, y_test


def preprocess(X_train_normal, X_test, pca_dim=0, seed=42):
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train_normal)
    X_te   = scaler.transform(X_test)
    if pca_dim > 0:
        dim = min(pca_dim, X_tr.shape[1], X_tr.shape[0] - 1)
        pca = PCA(n_components=dim, random_state=seed)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)
    return X_tr, X_te


# ── dim-reduction — same defaults as visualize_embeddings.py ─────────────────
def reduce_tsne(X, perplexity=30, seed=42):
    from sklearn.manifold import TSNE
    print(f"  t-SNE (n={len(X)}, perplexity={perplexity}) …")
    return TSNE(
        n_components=2, perplexity=perplexity, random_state=seed,
        init="pca", learning_rate="auto", n_jobs=-1
    ).fit_transform(X)


def reduce_umap(X, n_neighbors=15, min_dist=0.1, seed=42):
    import umap
    print(f"  UMAP (n={len(X)}, k={n_neighbors}, min_dist={min_dist}) …")
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=seed
    ).fit_transform(X)


# ── plotting helpers — identical style to visualize_embeddings.py ─────────────
def scatter_2d(ax, coords, labels, split):
    style = SPLIT_STYLE[split]
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLASS_COLOR[cls],
            label=f"{CLASS_LABEL[cls]} ({split})",
            **style
        )


def build_legend(ax):
    handles = []
    for cls in (0, 1):
        handles.append(mpatches.Patch(color=CLASS_COLOR[cls],
                                      label=CLASS_LABEL[cls]))
    for split, sty in SPLIT_STYLE.items():
        handles.append(plt.Line2D(
            [0], [0], marker=sty["marker"], color="gray",
            linestyle="None", markersize=7, label=f"{split} split"
        ))
    ax.legend(handles=handles, framealpha=0.4, loc="best")


def plot_projection(ax, title, coords_list, labels_list, splits):
    for coords, labels, split in zip(coords_list, labels_list, splits):
        scatter_2d(ax, coords, labels, split)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    build_legend(ax)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--backbone",  default="resnet18",
                        choices=["resnet18", "anatpaste"])
    parser.add_argument("--method",    default="tsne",
                        choices=["tsne", "umap", "both"])
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--save-dir",  default="plots")
    parser.add_argument("--fmt",       default="png",
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi",       type=int,   default=600)
    parser.add_argument("--pca-dim",   type=int,   default=0)
    parser.add_argument("--seed",      type=int,   default=42)

    # t-SNE — same default as visualize_embeddings.py
    parser.add_argument("--perplexity",  type=float, default=30)

    # UMAP — same defaults as visualize_embeddings.py
    parser.add_argument("--n-neighbors", type=int,   default=15)
    parser.add_argument("--min-dist",    type=float, default=0.1)

    # MSDE hyperparams
    parser.add_argument("--k",                          type=int,   default=20)
    parser.add_argument("--nbd-sample-count-threshold", type=int,   default=10)
    parser.add_argument("--learning-rate",              type=float, default=0.1)
    parser.add_argument("--max-iters-shift",            type=int,   default=5)
    parser.add_argument("--shift-threshold",            type=float, default=0.003)

    args = parser.parse_args()

    feature_dir = os.path.join(args.feature_root, args.dataset, args.backbone)
    print(f"[INFO] Feature dir: {feature_dir}")

    X_train, y_train, X_test, y_test = load_features(feature_dir)

    print(f"  Train normal : {np.sum(y_train==0)}  "
          f"Train anomaly: {np.sum(y_train==1)}")
    print(f"  Test  normal : {np.sum(y_test==0)}  "
          f"Test  anomaly: {np.sum(y_test==1)}")

    # Preprocess using normal train samples only (mirrors run_msde.py)
    X_train_normal = X_train[y_train == 0]
    X_tr, X_te = preprocess(X_train_normal, X_test,
                             pca_dim=args.pca_dim, seed=args.seed)

    # ── run MSDE ─────────────────────────────────────────────────────────────
    detector = MSDEWithShift(
        seed=args.seed,
        model_name="MSDE",
        k=args.k,
        nbd_sample_count_threshold=args.nbd_sample_count_threshold,
        learning_rate=args.learning_rate,
        max_iters_shift=args.max_iters_shift,
        shift_threshold=args.shift_threshold,
        scaler=StandardScaler(),
    )
    print("\n[MSDE] Fitting on normal train …")
    detector.fit(X_tr)

    print("[MSDE] Shifting train + test jointly …")
    _ = detector.predict_score(X_te)

    # ── retrieve shifted coordinates ─────────────────────────────────────────
    n_tr = detector.n_train_
    X_train_shifted = detector.X_joint_shifted_[:n_tr]   # normal train, shifted
    X_test_shifted  = detector.X_joint_shifted_[n_tr:]   # test, shifted

    # Labels: train is all-normal (label=0); test keeps original labels
    y_train_shifted = np.zeros(n_tr, dtype=int)
    y_test_shifted  = y_test.copy()

    # Stack for a single joint projection (same as visualize_embeddings.py)
    X_all = np.concatenate([X_train_shifted, X_test_shifted])
    y_all = np.concatenate([y_train_shifted, y_test_shifted])
    split_ids = np.array(
        ["train"] * n_tr + ["test"] * len(X_test_shifted)
    )

    # ── plot ─────────────────────────────────────────────────────────────────
    methods = ["tsne", "umap"] if args.method == "both" else [args.method]

    fig, axes = plt.subplots(1, len(methods),
                             figsize=(8 * len(methods), 7))
    if len(methods) == 1:
        axes = [axes]

    fig.suptitle(
        f"{args.dataset} — {args.backbone} (MSDE shifted)",
        fontsize=13, y=1.02
    )

    for ax, method in zip(axes, methods):
        if method == "tsne":
            coords = reduce_tsne(X_all, args.perplexity, args.seed)
            title  = f"t-SNE (perp={args.perplexity})"
        else:
            coords = reduce_umap(X_all, args.n_neighbors,
                                 args.min_dist, args.seed)
            title  = f"UMAP (k={args.n_neighbors})"

        coords_list, labels_list, splits_used = [], [], []
        for split in ["train", "test"]:
            mask = split_ids == split
            coords_list.append(coords[mask])
            labels_list.append(y_all[mask])
            splits_used.append(split)

        plot_projection(ax, title, coords_list, labels_list, splits_used)

    plt.tight_layout()

    os.makedirs(args.save_dir, exist_ok=True)
    fname = f"{args.dataset}_{args.backbone}_msde_shifted_{args.method}.{args.fmt}"
    fpath = os.path.join(args.save_dir, fname)
    fig.savefig(fpath, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[Saved] {fpath}")


if __name__ == "__main__":
    main()
