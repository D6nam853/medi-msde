"""
visualize_embeddings.py
-----------------------
Plot t-SNE or UMAP projections of features extracted by extract_features.py.

Usage examples
--------------
# t-SNE, ResNet-18, RSNA dataset
python visualize_embeddings.py --dataset rsna --backbone resnet18 --method tsne

# UMAP, AnatPaste, BraTS
python visualize_embeddings.py --dataset brats --backbone anatpaste --method umap

# Both methods side-by-side, save as PDF
python visualize_embeddings.py --dataset isic --backbone resnet18 --method both --fmt pdf

# Show test split only
python visualize_embeddings.py --dataset rsna --backbone resnet18 --split test
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── global style (LaTeX-friendly) ─────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})

# ── colour / marker scheme ────────────────────────────────────────────────────
SPLIT_STYLE = {
    "train": dict(marker="o", alpha=0.7, s=35, edgecolors="none"),
    "test":  dict(marker="^", alpha=0.85, s=40, edgecolors="none"),
}

CLASS_COLOR = {
    0: "#1D9E75",   # normal
    1: "#D85A30",   # anomaly
}

CLASS_LABEL = {0: "Normal", 1: "Anomaly"}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_split(feature_dir: str, split: str):
    feat_path  = os.path.join(feature_dir, f"{split}_features.npy")
    label_path = os.path.join(feature_dir, f"{split}_labels.npy")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features not found: {feat_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels not found: {label_path}")

    feats  = np.load(feat_path)
    labels = np.load(label_path).astype(int).flatten()

    print(f"[{split}] features={feats.shape} "
          f"normal={np.sum(labels==0)} anomaly={np.sum(labels==1)}")

    return feats, labels


def reduce_tsne(X, perplexity=30, seed=42):
    from sklearn.manifold import TSNE
    print(f"Running t-SNE (n={len(X)}, perplexity={perplexity})")
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        n_jobs=-1
    ).fit_transform(X)


def reduce_umap(X, n_neighbors=15, min_dist=0.1, seed=42):
    import umap
    print(f"Running UMAP (n={len(X)}, k={n_neighbors})")
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed
    ).fit_transform(X)


def scatter_2d(ax, coords, labels, split):
    style = SPLIT_STYLE[split]
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=CLASS_COLOR[cls],
            label=f"{CLASS_LABEL[cls]} ({split})",
            **style
        )


def build_legend(ax):
    handles = []

    for cls in (0, 1):
        handles.append(mpatches.Patch(
            color=CLASS_COLOR[cls],
            label=CLASS_LABEL[cls]
        ))

    for split, sty in SPLIT_STYLE.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=sty["marker"],
                color="gray",
                linestyle="None",
                markersize=7,
                label=f"{split} split"
            )
        )

    ax.legend(handles=handles, framealpha=0.4, loc="best")


def plot_projection(ax, title, coords_list, labels_list, splits):
    for coords, labels, split in zip(coords_list, labels_list, splits):
        scatter_2d(ax, coords, labels, split)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    build_legend(ax)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "anatpaste"])
    parser.add_argument("--method", default="tsne",
                        choices=["tsne", "umap", "both"])
    parser.add_argument("--split", default="both",
                        choices=["train", "test", "both"])
    parser.add_argument("--feature-root", default="features")
    parser.add_argument("--save-dir", default="plots")
    parser.add_argument("--fmt", default="png",
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)

    # t-SNE
    parser.add_argument("--perplexity", type=float, default=30)

    # UMAP
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    feature_dir = os.path.join(args.feature_root, args.dataset, args.backbone)
    print(f"Feature dir: {feature_dir}")

    splits = ["train", "test"] if args.split == "both" else [args.split]

    feats, labels = [], []
    for s in splits:
        f, l = load_split(feature_dir, s)
        feats.append(f)
        labels.append(l)

    X = np.concatenate(feats)
    y = np.concatenate(labels)

    split_ids = np.concatenate([
        np.full(len(f), s) for f, s in zip(feats, splits)
    ])

    methods = ["tsne", "umap"] if args.method == "both" else [args.method]


    fig, axes = plt.subplots(1, len(methods),
                             figsize=(8 * len(methods), 7))

    if len(methods) == 1:
        axes = [axes]

    fig.suptitle(
        f"{args.dataset} — {args.backbone}",
        fontsize=13,
        y=1.02
    )

    for ax, method in zip(axes, methods):
        if method == "tsne":
            coords = reduce_tsne(X, args.perplexity, args.seed)
            title = f"t-SNE (perp={args.perplexity})"
        else:
            coords = reduce_umap(X, args.n_neighbors,
                                 args.min_dist, args.seed)
            title = f"UMAP (k={args.n_neighbors})"

        coords_list, labels_list = [], []

        for s in splits:
            mask = split_ids == s
            coords_list.append(coords[mask])
            labels_list.append(y[mask])

        plot_projection(ax, title, coords_list, labels_list, splits)

    plt.tight_layout()

    os.makedirs(args.save_dir, exist_ok=True)

    fname = f"{args.dataset}_{args.backbone}_{args.method}_{args.split}.{args.fmt}"
    fpath = os.path.join(args.save_dir, fname)

    fig.savefig(
        fpath,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main()
