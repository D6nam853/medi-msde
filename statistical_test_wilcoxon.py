"""Run the manuscript's paired Wilcoxon signed-rank comparisons.

The unit of analysis is the dataset: for each baseline, the seven AUC or AP
values are paired with the corresponding MSDE-GDE values. The script reports
exact two-sided p-values, W+ and W-, plus Holm-adjusted p-values for both the
nine comparisons within each metric and all 18 metric-method comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]
DEFAULT_BASELINES = [
    "AE",
    "AE-Perceptual",
    "VAE",
    "MemAE",
    "GANomaly",
    "CutPaste-2Stage",
    "AnatPaste",
    "ResNet18-IN",
    "ResNet152-IN",
]
METRICS = ["auc", "ap"]


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty_like(p_values)
    running_max = 0.0
    n = len(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (n - rank) * p_values[index])
        adjusted_sorted[rank] = running_max
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def signed_rank_result(ours: np.ndarray, baseline: np.ndarray) -> dict:
    differences = np.asarray(ours, dtype=float) - np.asarray(baseline, dtype=float)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return {
            "n": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "p_exact": 1.0,
            "mean_difference": 0.0,
        }

    ranks = rankdata(np.abs(nonzero), method="average")
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())

    try:
        test = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )
    except TypeError:  # compatibility with older SciPy releases
        test = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            mode="exact",
        )
    return {
        "n": int(len(nonzero)),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_exact": float(test.pvalue),
        "mean_difference": float(differences.mean()),
    }


def format_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"${value:.3f}$"


def make_latex_table(results: pd.DataFrame, output_path: Path) -> None:
    baselines = results["baseline"].drop_duplicates().tolist()
    rows = []
    for baseline in baselines:
        row = results[results["baseline"] == baseline].set_index("metric")
        auc = row.loc["auc"]
        ap = row.loc["ap"]
        rows.append(
            f"{baseline} & "
            f"{auc.w_plus:g}/{auc.w_minus:g}/{format_p(auc.p_exact)} & "
            f"{ap.w_plus:g}/{ap.w_minus:g}/{format_p(ap.p_exact)} \\\\"
        )

    table = """\\begin{table}[t]
\\centering
\\caption{Exact two-sided Wilcoxon signed-rank test comparing MSDE-GDE
with each baseline over the seven datasets. Entries are
$W^+/W^-/p$, where $W^+$ and $W^-$ are the sums of positive and negative
ranks, respectively.}
\\label{tab:wilcoxon}
\\small
\\begin{tabular}{@{}lcc@{}}
\\toprule
\\textbf{Baseline} & \\textbf{AUC} & \\textbf{AP} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    output_path.write_text(table, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results") / "benchmark_results" / "all_results.csv",
    )
    parser.add_argument("--method", default="MSDE-GDE-BiomedCLIP")
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "statistical_test_results",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    required = {"method", "dataset", "seed", "fold", "auc", "ap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {args.results}: {sorted(missing)}")

    frame = frame[
        (frame["seed"] == args.seed)
        & (frame["fold"] == args.fold)
        & (frame["method"].isin([args.method] + args.baselines))
    ].copy()
    frame = frame[frame["dataset"].isin(DATASETS)]

    duplicate_counts = frame.groupby(["method", "dataset"]).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        raise ValueError(
            "Expected one result per method/dataset after filtering, but found "
            f"duplicates:\n{duplicates}"
        )

    missing_methods = [
        method
        for method in [args.method] + args.baselines
        if method not in set(frame["method"])
    ]
    if missing_methods:
        raise ValueError(f"Missing methods in filtered results: {missing_methods}")

    rows = []
    for metric in METRICS:
        pivot = frame.pivot(index="dataset", columns="method", values=metric)
        pivot = pivot.reindex(DATASETS)
        if pivot.isna().any().any():
            raise ValueError(f"Missing {metric} values after dataset alignment")

        for baseline in args.baselines:
            result = signed_rank_result(
                pivot[args.method].to_numpy(),
                pivot[baseline].to_numpy(),
            )
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "method": args.method,
                    **result,
                }
            )

    results = pd.DataFrame(rows)
    results["p_holm_within_metric"] = np.nan
    for metric in METRICS:
        mask = results["metric"] == metric
        results.loc[mask, "p_holm_within_metric"] = holm_adjust(
            results.loc[mask, "p_exact"].to_numpy()
        )
    results["p_holm_all_18"] = holm_adjust(results["p_exact"].to_numpy())
    results["significant_raw_alpha_005"] = results["p_exact"] < 0.05
    results["significant_holm_within_metric"] = (
        results["p_holm_within_metric"] < 0.05
    )
    results["significant_holm_all_18"] = results["p_holm_all_18"] < 0.05

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "wilcoxon_results.csv", index=False)
    make_latex_table(results, args.output_dir / "wilcoxon_table.tex")

    print(f"Results saved to {args.output_dir.resolve()}")
    print(results[[
        "baseline", "metric", "n", "w_plus", "w_minus", "p_exact",
        "p_holm_within_metric", "p_holm_all_18",
    ]].to_string(index=False))
    print("\nRaw p<0.05 counts by metric:")
    print(results.groupby("metric")["significant_raw_alpha_005"].sum())
    print("\nHolm-adjusted p<0.05 counts within each metric:")
    print(results.groupby("metric")["significant_holm_within_metric"].sum())


if __name__ == "__main__":
    main()
