"""Generate the small set of paper figures from benchmark CSV files.

Default outputs:
  1. main_auc_heatmap.pdf/png       Overall benchmark AUC.
  2. detector_auc_comparison.pdf/png Fixed BiomedCLIP detector comparison.
  3. protocol_ablation_auc.pdf/png  Five MSDE inference protocols.
  4. order_sensitivity_auc.pdf/png  Random-order variation and diagnostics.

Use --include-runtime to additionally create a runtime figure.
All PDF files are vector figures; PNG files are 600-dpi previews.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
DATASET_LABELS = {
    "rsna": "RSNA",
    "vin": "VinDr",
    "brain": "Brain",
    "lag": "LAG",
    "brats": "BraTS",
    "c16": "C16",
    "isic": "ISIC",
}

MAIN_METHODS = [
    "AE",
    "AE-Perceptual",
    "VAE",
    "MemAE",
    "GANomaly",
    "CutPaste-2Stage",
    "AnatPaste",
    "ResNet18-IN",
    "ResNet152-IN",
    "MSDE-GDE-BiomedCLIP",
]
MAIN_METHOD_LABELS = {
    "AE": "AE",
    "AE-Perceptual": "AE-Perceptual",
    "VAE": "VAE",
    "MemAE": "MemAE",
    "GANomaly": "GANomaly",
    "CutPaste-2Stage": "CutPaste-2Stage",
    "AnatPaste": "AnatPaste",
    "ResNet18-IN": "ResNet18-IN",
    "ResNet152-IN": "ResNet152-IN",
    "MSDE-GDE-BiomedCLIP": r"MSDE-GDE$^{\dagger}$",
}

DETECTOR_METHODS = [
    "gde",
    "knn",
    "lof",
    "iforest",
    "ocsvm",
    "msde-reference",
    "msde-knn",
    "msde-gde",
]
DETECTOR_LABELS = {
    "gde": "GDE",
    "knn": "kNN",
    "lof": "LOF",
    "iforest": "Isolation Forest",
    "ocsvm": "OC-SVM",
    "msde-distance": "MSDE-distance",
    "msde-reference": "MSDE-reference",
    "msde-knn": "MSDE-kNN",
    "msde-gde": "MSDE-GDE",
}

PROTOCOLS = [
    "no_shift",
    "train_only",
    "shifted_train_raw_test",
    "raw_train_shifted_test",
    "transductive",
]
PROTOCOL_LABELS = {
    "no_shift": "No shift",
    "train_only": "Train-only shift field",
    "shifted_train_raw_test": "Shifted train only",
    "raw_train_shifted_test": "Shifted test only",
    "transductive": "Transductive (both)",
}

COLORS = {
    "neutral": "#7A8793",
    "blue": "#3B6EA8",
    "teal": "#2A9D8F",
    "orange": "#E38B3B",
    "red": "#B94745",
    "dark": "#243447",
    "grid": "#D9DEE5",
}


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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf")
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=600)
    plt.close(fig)


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def plot_main_auc(root: Path, output_dir: Path) -> None:
    frame = read_csv(
        root / "results" / "benchmark_results" / "all_results.csv",
        {"method", "dataset", "auc"},
    )
    frame = frame[frame["method"].isin(MAIN_METHODS)]
    table = frame.pivot_table(index="method", columns="dataset", values="auc", aggfunc="mean")
    table = table.reindex(index=MAIN_METHODS, columns=DATASETS)
    table["Mean"] = table.mean(axis=1)

    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    image = ax.imshow(
        values,
        cmap="YlGnBu",
        norm=Normalize(vmin=0.35, vmax=1.0),
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(table.columns)),
        [DATASET_LABELS.get(x, x) if x != "Mean" else "Mean" for x in table.columns],
    )
    ax.set_yticks(
        np.arange(len(table.index)),
        [MAIN_METHOD_LABELS.get(x, x) for x in table.index],
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Method")
    ax.set_title("Image-level AUC across the MedIAnomaly benchmark")
    ax.axvline(len(DATASETS) - 0.5, color="white", linewidth=1.5)
    ax.axhline(len(MAIN_METHODS) - 1.5, color=COLORS["dark"], linewidth=1.2)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = "white" if value >= 0.78 else COLORS["dark"]
            weight = "bold" if table.index[row] == "MSDE-GDE-BiomedCLIP" else "normal"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color=text_color, fontsize=7.3, fontweight=weight)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("AUC")
    colorbar.ax.tick_params(labelsize=7)
    ax.text(
        0.0,
        -0.16,
        r"$\dagger$ transductive; all other methods are inductive",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.22, right=0.93, bottom=0.24, top=0.90)
    save_figure(fig, output_dir, "main_auc_heatmap")


def plot_detector_auc(root: Path, output_dir: Path, include_msde_distance: bool) -> None:
    frame = read_csv(
        root / "results" / "results_biomedclip_detectors" / "all_results.csv",
        {"method", "dataset", "auc"},
    )
    methods = DETECTOR_METHODS.copy()
    if include_msde_distance:
        methods.insert(5, "msde-distance")
    frame = frame[frame["method"].isin(methods)]
    grouped = frame.groupby("method")["auc"]
    means = grouped.mean().reindex(methods)
    stds = grouped.std(ddof=1).reindex(methods).fillna(0.0)

    labels = [DETECTOR_LABELS[x] for x in methods]
    y = np.arange(len(methods))
    transductive = {"msde-gde", "msde-knn", "msde-distance"}
    colors = [
        COLORS["orange"] if method in transductive else COLORS["blue"]
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(5.65, 3.55))
    bars = ax.barh(
        y,
        means,
        xerr=stds,
        color=colors,
        alpha=0.9,
        error_kw={"ecolor": COLORS["dark"], "capsize": 2, "elinewidth": 0.7},
        height=0.62,
    )
    for bar, method in zip(bars, methods):
        if method in transductive:
            bar.set_hatch("//")
            bar.set_edgecolor(COLORS["dark"])
            bar.set_linewidth(0.4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.55, 1.04)
    ax.set_xlabel("Mean AUC $\u00b1$ dataset SD (one run per method)")
    ax.set_title("Detector comparison on fixed BiomedCLIP features")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, value, spread in zip(y, means, stds):
        # Put the annotation beyond the error-bar cap so it remains legible.
        ax.text(
            min(value + spread + 0.012, 1.025),
            yi,
            f"{value:.3f}",
            va="center",
            fontsize=7.5,
        )
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["blue"], label="Inductive / normal-reference"),
            Patch(
                facecolor=COLORS["orange"],
                edgecolor=COLORS["dark"],
                hatch="//",
                label="Transductive",
            ),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.30, top=0.90)
    save_figure(fig, output_dir, "detector_auc_comparison")


def plot_protocol_ablation(root: Path, output_dir: Path) -> None:
    frame = read_csv(
        root / "results" / "results_msde_ablation" / "all_results.csv",
        {"protocol", "dataset", "auc"},
    )
    table = frame.pivot_table(index="protocol", columns="dataset", values="auc", aggfunc="mean")
    table = table.reindex(index=PROTOCOLS, columns=DATASETS)
    table["Mean"] = table.mean(axis=1)

    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    image = ax.imshow(
        values,
        cmap="YlGnBu",
        norm=Normalize(vmin=0.65, vmax=1.0),
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(table.columns)),
        [DATASET_LABELS.get(x, x) if x != "Mean" else "Mean" for x in table.columns],
    )
    ax.set_yticks(np.arange(len(table.index)), [PROTOCOL_LABELS[x] for x in table.index])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Inference protocol")
    ax.set_title("Ablation of the training- and test-side MSDE shifts")
    ax.axvline(len(DATASETS) - 0.5, color="white", linewidth=1.5)
    ax.axhline(2.5, color=COLORS["dark"], linewidth=1.2)
    ax.text(
        0.5,
        -0.29,
        "Rows 1--3: without test cohort   |   Rows 4--5: with test cohort",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = "white" if value >= 0.82 else COLORS["dark"]
            weight = "bold" if table.index[row] == "transductive" else "normal"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color=text_color, fontsize=7.5, fontweight=weight)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("AUC")
    colorbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.30, right=0.93, bottom=0.31, top=0.88)
    save_figure(fig, output_dir, "protocol_ablation_auc")


def plot_order_sensitivity(root: Path, output_dir: Path) -> None:
    frame = read_csv(
        root / "results" / "results_order_permutation" / "all_results.csv",
        {"order_type", "condition", "dataset", "auc"},
    )
    random_frame = frame[frame["order_type"] == "random"]
    random_stats = random_frame.groupby("dataset")["auc"].agg(["mean", "std"]).reindex(DATASETS)
    diagnostic = frame[frame["order_type"] == "label_diagnostic"]

    x = np.arange(len(DATASETS))
    fig, ax = plt.subplots(figsize=(6.7, 3.45))
    ax.errorbar(
        x,
        random_stats["mean"],
        yerr=random_stats["std"].fillna(0),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        elinewidth=1.0,
        capsize=2.5,
        markersize=4.5,
        label="Random permutations (mean $\u00b1$ SD)",
    )

    marker_by_condition = {
        "label_ascending": ("^", COLORS["orange"], "Label ascending"),
        "label_descending": ("v", COLORS["red"], "Label descending"),
    }
    for condition, (marker, color, label) in marker_by_condition.items():
        values = (
            diagnostic[diagnostic["condition"] == condition]
            .set_index("dataset")["auc"]
            .reindex(DATASETS)
        )
        ax.scatter(x, values, marker=marker, color=color, s=27,
                   zorder=3, label=label)

    ax.set_xticks(x, [DATASET_LABELS[x] for x in DATASETS])
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Dataset")
    ax.set_title("Sensitivity of transductive MSDE-GDE to test-cohort order")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.88)
    save_figure(fig, output_dir, "order_sensitivity_auc")


def plot_runtime(root: Path, output_dir: Path) -> None:
    frame = read_csv(
        root / "results" / "benchmark_results" / "all_results.csv",
        {"method", "runtime_seconds", "runtime_scope"},
    )
    frame = frame[frame["method"].isin(MAIN_METHODS)].copy()
    grouped = frame.groupby("method")["runtime_seconds"].mean().reindex(MAIN_METHODS).dropna()
    scopes = frame.groupby("method")["runtime_scope"].first().reindex(grouped.index)

    # The detector-only MSDE-GDE timing is available from the controlled
    # BiomedCLIP detector benchmark. Use it to show the approximate split
    # between feature extraction and MSDE-GDE scoring.
    detector_path = root / "results" / "results_biomedclip_detectors" / "all_results.csv"
    msde_detector_seconds = None
    if detector_path.exists():
        detector_frame = pd.read_csv(detector_path)
        detector_rows = detector_frame[detector_frame["method"] == "msde-gde"]
        if not detector_rows.empty:
            msde_detector_seconds = float(detector_rows["runtime_seconds"].mean())

    grouped = grouped.sort_values()
    labels = [MAIN_METHOD_LABELS.get(x, x) for x in grouped.index]
    y = np.arange(len(grouped))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for yi, (method, value) in enumerate(grouped.items()):
        if method == "MSDE-GDE-BiomedCLIP" and msde_detector_seconds is not None:
            detector_seconds = min(msde_detector_seconds, value)
            extraction_seconds = value - detector_seconds
            ax.barh(
                yi,
                extraction_seconds,
                color=COLORS["orange"],
                height=0.62,
                hatch="..",
                edgecolor=COLORS["dark"],
                linewidth=0.35,
            )
            ax.barh(
                yi,
                detector_seconds,
                left=extraction_seconds,
                color=COLORS["red"],
                height=0.62,
                hatch="//",
                edgecolor=COLORS["dark"],
                linewidth=0.35,
            )
        else:
            hatch = "//" if scopes.loc[method] == "train_plus_eval" else "\\\\"
            ax.barh(
                yi,
                value,
                color=COLORS["blue"],
                height=0.62,
                hatch=hatch,
                edgecolor=COLORS["dark"],
                linewidth=0.35,
            )
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(max(1.0, grouped.min() * 0.7), grouped.max() * 2.2)
    ax.set_xlabel("Mean runtime per dataset (s, logarithmic scale)")
    ax.set_title("Runtime comparison; scopes shown by hatch")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for yi, value in zip(y, grouped.values):
        ax.text(
            value * 1.18,
            yi,
            f"{value:.0f}",
            va="center",
            ha="left",
            fontsize=7.5,
            clip_on=False,
        )
    legend_handles = [
        Patch(facecolor=COLORS["blue"], edgecolor=COLORS["dark"], hatch="//",
              label="Training + evaluation"),
        Patch(facecolor=COLORS["blue"], edgecolor=COLORS["dark"], hatch="\\\\",
              label="Feature extraction + detector"),
    ]
    if msde_detector_seconds is not None:
        legend_handles.extend(
            [
                Patch(facecolor=COLORS["orange"], edgecolor=COLORS["dark"], hatch="..",
                      label="MSDE-GDE BiomedCLIP extraction"),
                Patch(facecolor=COLORS["red"], edgecolor=COLORS["dark"], hatch="//",
                      label="MSDE-GDE scoring"),
            ]
        )
    ax.legend(handles=legend_handles, frameon=False, fontsize=6.7,
              loc="lower right")
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.18, top=0.90)
    save_figure(fig, output_dir, "runtime_comparison")


def plot_teacher_fraction(root: Path, output_dir: Path) -> None:
    """Plot surrogate performance as the teacher-cohort fraction changes."""
    configurations = [
        (0.30, root / "results" / "results_surrogate_30_70" / "summary_by_dataset.csv"),
        (0.50, root / "results" / "results_surrogate_50_50" / "summary_by_dataset.csv"),
        (0.80, root / "results" / "results_inductive_msde_surrogate_5splits" / "summary_by_dataset.csv"),
    ]
    required = {
        "dataset",
        "auc_mean",
        "ap_mean",
        "precision_at_n_mean",
    }
    frames = []
    for fraction, path in configurations:
        frame = read_csv(path, required).copy()
        frame["teacher_fraction"] = fraction
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)

    metrics = [
        ("auc_mean", "Held-out AUC", 0.884),
        ("ap_mean", "Held-out AP", 0.880),
        ("precision_at_n_mean", "Held-out Precision@N", 0.807),
    ]
    dataset_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.75, len(DATASETS)))
    color_by_dataset = dict(zip(DATASETS, dataset_colors))

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 4.15), sharex=True)
    for ax, (metric, ylabel, reference) in zip(axes, metrics):
        for dataset in DATASETS:
            values = (
                frame[frame["dataset"] == dataset]
                .set_index("teacher_fraction")
                .reindex([0.30, 0.50, 0.80])[metric]
            )
            ax.plot(
                [30, 50, 80],
                values.to_numpy(dtype=float),
                marker="o",
                markersize=4,
                linewidth=1.25,
                alpha=0.82,
                color=color_by_dataset[dataset],
            )

        mean_values = (
            frame.groupby("teacher_fraction")[metric]
            .mean()
            .reindex([0.30, 0.50, 0.80])
        )
        ax.plot(
            [30, 50, 80],
            mean_values.to_numpy(dtype=float),
            color=COLORS["dark"],
            marker="o",
            markersize=5,
            linewidth=2.4,
            label="Mean across datasets",
            zorder=5,
        )
        ax.axhline(
            reference,
            color=COLORS["red"],
            linestyle="--",
            linewidth=0.9,
            label="Original MSDE-GDE (full test cohort)",
        )
        ax.set_title(ylabel, fontsize=9)
        ax.set_xlabel("Teacher cohort (%)")
        ax.set_xticks([30, 50, 80])
        ax.set_ylim(0.60, 1.0)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Metric value")

    handles = [
        Line2D([0], [0], color=COLORS["dark"], marker="o", linewidth=2,
               markersize=4, label="Mean across datasets"),
        Line2D([0], [0], color=COLORS["red"], linestyle="--", linewidth=0.9,
               label="Original MSDE-GDE (full test cohort)"),
    ]
    dataset_handles = [
        Line2D(
            [0], [0], color=color_by_dataset[dataset], marker="o",
            linewidth=1.25, markersize=4, label=DATASET_LABELS[dataset]
        )
        for dataset in DATASETS
    ]
    fig.legend(handles=handles, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=7.5)
    fig.legend(handles=dataset_handles, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.005), fontsize=7.2)
    fig.suptitle("Sensitivity to the teacher-cohort fraction", y=1.025, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.83, wspace=0.18)
    save_figure(fig, output_dir, "teacher_fraction_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    parser.add_argument(
        "--include-msde-distance",
        action="store_true",
        help="Include the weak test-only MSDE-distance variant in detector plot",
    )
    parser.add_argument(
        "--include-runtime",
        action="store_true",
        help="Also create the optional runtime figure",
    )
    parser.add_argument(
        "--include-teacher-fraction",
        action="store_true",
        help="Create the surrogate teacher-fraction sensitivity figure",
    )
    args = parser.parse_args()

    configure_matplotlib()
    root = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()

    plot_main_auc(root, output_dir)
    plot_detector_auc(root, output_dir, args.include_msde_distance)
    plot_protocol_ablation(root, output_dir)
    plot_order_sensitivity(root, output_dir)
    if args.include_runtime:
        plot_runtime(root, output_dir)
    if args.include_teacher_fraction:
        plot_teacher_fraction(root, output_dir)

    print(f"Saved paper figures to {output_dir}")


if __name__ == "__main__":
    main()
