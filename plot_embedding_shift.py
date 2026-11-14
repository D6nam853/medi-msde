"""Visualize raw and MSDE-shifted BiomedCLIP test embeddings.

The test labels are used only to color the visualization. They are never
passed to MSDE. PCA is fitted on the concatenation of raw and shifted test
embeddings so both panels use identical coordinates and axis limits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from MSDE_v3 import mean_shift_density_enhancement


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
DATASET_LABELS = {
    "rsna": "RSNA",
    "vin": "VinDr-CXR",
    "brain": "Brain Tumor",
    "lag": "LAG",
    "brats": "BraTS2021",
    "c16": "Camelyon16",
    "isic": "ISIC 2018",
}

# Match the benchmark configuration exactly.
MSDE_K = 100
MSDE_LR = 0.33
MSDE_MAX_ITERS = 8
MSDE_SHIFT_THRESH = 0.01
MSDE_NBD_THRESHOLD = 70

NORMAL_COLOR = "#3B6EA8"
ANOMALY_COLOR = "#B94745"
ARROW_COLOR = "#6B7280"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def load_dataset(feature_root: Path, dataset: str):
    path = feature_root / dataset / "biomedclip"
    train_x = np.load(path / "train_features.npy")
    train_y = np.load(path / "train_labels.npy")
    test_x = np.load(path / "test_features.npy")
    test_y = np.load(path / "test_labels.npy")
    return train_x, train_y, test_x, test_y


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf")
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=600)
    plt.close(fig)


def shift_test_embeddings(test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted, total_distance = mean_shift_density_enhancement(
        test_x,
        k=MSDE_K,
        nbd_sample_count_threshold=MSDE_NBD_THRESHOLD,
        learning_rate=MSDE_LR,
        max_iters_shift=MSDE_MAX_ITERS,
        shift_threshold=MSDE_SHIFT_THRESH,
    )
    return shifted, total_distance


def plot_dataset(
    dataset: str,
    test_x: np.ndarray,
    shifted_x: np.ndarray,
    test_y: np.ndarray,
    output_dir: Path,
    seed: int,
    max_points: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    n_samples = len(test_x)
    if max_points > 0 and n_samples > max_points:
        plot_indices = np.sort(rng.choice(n_samples, size=max_points, replace=False))
    else:
        plot_indices = np.arange(n_samples)

    raw_plot = test_x[plot_indices]
    shifted_plot = shifted_x[plot_indices]
    labels_plot = test_y[plot_indices]

    # Use one coordinate system for both panels.
    combined = np.vstack([raw_plot, shifted_plot])
    pca = PCA(n_components=2, random_state=seed)
    combined_2d = pca.fit_transform(combined)
    raw_2d = combined_2d[: len(raw_plot)]
    shifted_2d = combined_2d[len(raw_plot) :]

    # Fit UMAP jointly as well, so raw and shifted points are directly
    # comparable in the same two-dimensional coordinate system.
    umap_neighbors = min(30, max(2, len(combined) - 1))
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    )
    combined_umap = umap_model.fit_transform(combined)
    raw_umap = combined_umap[: len(raw_plot)]
    shifted_umap = combined_umap[len(raw_plot) :]

    limits = np.vstack([raw_2d, shifted_2d])
    lower = limits.min(axis=0)
    upper = limits.max(axis=0)
    margin = np.maximum((upper - lower) * 0.06, 1e-6)
    lower -= margin
    upper += margin

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharex=True, sharey=True)
    panels = [(axes[0], raw_2d, "Raw BiomedCLIP"),
              (axes[1], shifted_2d, "After MSDE")]

    # Draw a small, fixed subset of movement arrows so dense datasets remain
    # readable while the direction of the transformation is visible.
    arrow_count = min(120, len(raw_2d))
    arrow_indices = rng.choice(len(raw_2d), size=arrow_count, replace=False)

    for ax, coordinates, title in panels:
        for idx in arrow_indices:
            ax.plot(
                [raw_2d[idx, 0], shifted_2d[idx, 0]],
                [raw_2d[idx, 1], shifted_2d[idx, 1]],
                color=ARROW_COLOR,
                alpha=0.10,
                linewidth=0.35,
                zorder=1,
            )

        normal = labels_plot == 0
        abnormal = labels_plot == 1
        ax.scatter(coordinates[normal, 0], coordinates[normal, 1],
                   s=5, alpha=0.42, c=NORMAL_COLOR, linewidths=0, zorder=2)
        ax.scatter(coordinates[abnormal, 0], coordinates[abnormal, 1],
                   s=7, alpha=0.55, c=ANOMALY_COLOR, linewidths=0, zorder=3)
        ax.set_title(title)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.grid(color="#D9DEE5", linewidth=0.45, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("PC1")

    axes[0].set_ylabel("PC2")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NORMAL_COLOR,
               markeredgewidth=0, markersize=5, label="Normal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ANOMALY_COLOR,
               markeredgewidth=0, markersize=5, label="Abnormal"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: BiomedCLIP test embeddings",
        y=1.08,
        fontsize=10,
    )
    fig.text(
        0.5,
        -0.01,
        "PCA fitted jointly to raw and shifted embeddings; arrows show sample movement",
        ha="center",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.78, wspace=0.08)
    save_figure(fig, output_dir, f"embedding_shift_{dataset}")

    displacement = np.linalg.norm(shifted_x - test_x, axis=1)
    summary = {
        "dataset": dataset,
        "n_test": n_samples,
        "n_plotted": len(plot_indices),
        "pca_variance_pc1": pca.explained_variance_ratio_[0],
        "pca_variance_pc2": pca.explained_variance_ratio_[1],
        "mean_shift_displacement": float(displacement.mean()),
        "median_shift_displacement": float(np.median(displacement)),
        "normal_mean_displacement": float(displacement[test_y == 0].mean()),
        "abnormal_mean_displacement": float(displacement[test_y == 1].mean()),
    }
    projection = {
        "pca_raw_2d": raw_2d,
        "pca_shifted_2d": shifted_2d,
        "umap_raw_2d": raw_umap,
        "umap_shifted_2d": shifted_umap,
        "labels": labels_plot,
        "pca_lower": lower,
        "pca_upper": upper,
    }
    return summary, projection


def plot_grid(
    projections: list[tuple[str, dict]],
    output_dir: Path,
    projection_name: str,
    x_label: str,
    y_label: str,
    footer: str,
    stem: str | None = None,
) -> None:
    """Save a compact 2-by-7 overview for one projection method."""
    n_columns = len(projections)
    figure_width = max(8.8, 3.55 * n_columns)
    readable_layout = n_columns == 4
    figure_height = 7.2 if readable_layout else 4.7
    fig, axes = plt.subplots(2, n_columns, figsize=(figure_width, figure_height))
    if len(projections) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    raw_key = f"{projection_name}_raw_2d"
    shifted_key = f"{projection_name}_shifted_2d"

    # Use one set of limits per dataset, shared by its raw and shifted panels.
    limits_by_dataset = {}
    for dataset, projection in projections:
        coordinates = np.vstack([projection[raw_key], projection[shifted_key]])
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        margin = np.maximum((upper - lower) * 0.06, 1e-6)
        limits_by_dataset[dataset] = (lower - margin, upper + margin)

    for column, (dataset, projection) in enumerate(projections):
        for row, key, title in [
            (0, raw_key, "Before MSDE"),
            (1, shifted_key, "After MSDE"),
        ]:
            ax = axes[row, column]
            coordinates = projection[key]
            labels = projection["labels"]
            normal = labels == 0
            abnormal = labels == 1

            ax.scatter(coordinates[normal, 0], coordinates[normal, 1],
                       s=2.2, alpha=0.36, c=NORMAL_COLOR, linewidths=0)
            ax.scatter(coordinates[abnormal, 0], coordinates[abnormal, 1],
                       s=3.0, alpha=0.48, c=ANOMALY_COLOR, linewidths=0)
            lower, upper = limits_by_dataset[dataset]
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            ax.grid(color="#D9DEE5", linewidth=0.35, alpha=0.65)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8 if readable_layout else 6, pad=2 if readable_layout else 1)
            if readable_layout:
                ax.set_box_aspect(1)

            if row == 0:
                ax.set_title(
                    DATASET_LABELS[dataset],
                    fontsize=13 if readable_layout else 8.5,
                    pad=6 if readable_layout else 3,
                )
            if row == 1:
                ax.set_xlabel(
                    x_label,
                    fontsize=10 if readable_layout else 7.5,
                    labelpad=3 if readable_layout else 1,
                )
            else:
                ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel(
                    f"{title}\n{y_label}",
                    fontsize=13 if readable_layout else 7.5,
                    labelpad=7 if readable_layout else 2,
                )
            else:
                ax.set_yticklabels([])

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NORMAL_COLOR,
               markeredgewidth=0, markersize=5, label="Normal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ANOMALY_COLOR,
               markeredgewidth=0, markersize=5, label="Abnormal"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=13 if readable_layout else 8,
    )
    fig.suptitle(
        "BiomedCLIP test embeddings before and after MSDE",
        y=1.06 if readable_layout else 1.07,
        fontsize=15 if readable_layout else 10,
    )
    fig.text(
        0.5,
        0.005,
        footer,
        ha="center",
        fontsize=14 if readable_layout else 7.5,
    )
    if readable_layout:
        fig.subplots_adjust(
            left=0.075, right=0.995, bottom=0.10, top=0.88,
            wspace=0.13, hspace=0.20,
        )
    else:
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.13, top=0.80,
                            wspace=0.08, hspace=0.13)
    if stem is None:
        stem = f"embedding_shift_grid_{projection_name}"
    save_figure(fig, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("paper_figures") / "embedding_shift")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-points",
        type=int,
        default=2500,
        help="Maximum points per panel; use 0 to plot every test image",
    )
    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = DATASETS
    unknown = [dataset for dataset in args.datasets if dataset not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    configure_matplotlib()
    rows = []
    projections = []

    for dataset in args.datasets:
        print(f"[{dataset}] loading BiomedCLIP features")
        train_x, train_y, test_x, test_y = load_dataset(args.feature_root, dataset)

        # Match the benchmark: standardise with normal training statistics.
        train_normal = train_x[train_y == 0]
        scaler = StandardScaler().fit(train_normal)
        test_scaled = scaler.transform(test_x)

        print(f"[{dataset}] applying MSDE to the test cohort")
        shifted_scaled, _ = shift_test_embeddings(test_scaled)
        summary, projection = plot_dataset(
                dataset,
                test_scaled,
                shifted_scaled,
                test_y,
                args.output_dir,
                args.seed,
                args.max_points,
            )
        rows.append(summary)
        projections.append((dataset, projection))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_grid(
        projections,
        args.output_dir,
        projection_name="pca",
        x_label="PC1",
        y_label="PC2",
        footer=(
            "PCA fitted jointly to raw and shifted embeddings; "
            "labels are used only for visualization"
        ),
    )
    readable_datasets = {"rsna", "vin", "brain", "lag"}
    readable_projections = [
        item for item in projections if item[0] in readable_datasets
    ]
    if len(readable_projections) == len(readable_datasets):
        plot_grid(
            readable_projections,
            args.output_dir,
            projection_name="pca",
            x_label="PC1",
            y_label="PC2",
            footer=(
                "PCA fitted jointly to raw and shifted embeddings; "
                "labels are used only for visualization"
            ),
            stem="embedding_shift_grid_pca_readable",
        )
    plot_grid(
        projections,
        args.output_dir,
        projection_name="umap",
        x_label="UMAP 1",
        y_label="UMAP 2",
        footer=(
            "UMAP fitted jointly to raw and shifted embeddings "
            "(n_neighbors=30, min_dist=0.1); labels are used only for visualization"
        ),
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "embedding_shift_summary.csv", index=False)
    print(f"Saved embedding plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
