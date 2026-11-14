"""Run and aggregate the MedIAnomaly comparison benchmark.

The paper baselines use their native encoders. MSDE-GDE uses the
pre-extracted BiomedCLIP embeddings. Every run is timed externally from the
start of training/evaluation until the command exits, and image-level scores
are normalized into one CSV schema with AUC, AP, and precision@N.

Example:
    python benchmark.py --gpu 0 --datasets rsna vin --methods all

This is intentionally sequential because the methods share one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark_utils import compute_metrics, save_benchmark_result


DATASETS = ["rsna", "vin", "brain", "lag", "brats", "c16", "isic"]


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    native_name: str


METHODS = [
    Method("AE", "reconstruction", "ae"),
    Method("AE-Perceptual", "reconstruction", "ae-perceptual"),
    Method("VAE", "reconstruction", "vae"),
    Method("MemAE", "reconstruction", "memae"),
    Method("GANomaly", "reconstruction", "ganomaly"),
    Method("ResNet18-IN", "frozen_gde", "resnet18"),
    Method("ResNet152-IN", "frozen_gde", "resnet152"),
    Method("CutPaste-2Stage", "two_stage", "normal"),
    Method("AnatPaste", "two_stage", "anatpaste"),
    Method("MSDE-GDE-BiomedCLIP", "ours", "biomedclip"),
]


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def run_command(command: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode, time.perf_counter() - started


def command_for(method: Method, dataset: str, args: argparse.Namespace) -> tuple[list[str], Path]:
    py = sys.executable
    root = Path(__file__).resolve().parent

    if method.family == "reconstruction":
        return [
            py, "train.py", "-d", dataset, "-m", method.native_name,
            "-g", str(args.gpu), "-f", str(args.fold),
            "--train-seed", str(args.seed),
            "--result-root", str(args.reconstruction_root),
        ], root / "reconstruction"

    if method.family == "one_stage":
        # Training and evaluation are run as one timed unit by the caller.
        return [
            py, "train_med.py", "-d", dataset, "-s", method.native_name,
            "-g", str(args.gpu), "-f", str(args.fold),
            "--seed", str(args.seed), "--workers", str(args.workers),
        ], root / "ssl" / "one_stage"

    if method.family == "two_stage":
        return [
            py, "run_training.py", "--variant", method.native_name,
            "--type", dataset, "--no-pretrained", "--cuda", str(args.gpu),
            "--workers", str(args.workers), "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--model_dir", str(args.two_stage_model_dir),
        ], root / "ssl" / "two_stage"

    if method.family == "frozen_gde":
        return [
            py, "run_resnet_gde.py", "--dataset", dataset,
            "--backbone", method.native_name,
            "--gpu", str(args.gpu), "--batch-size", str(args.feature_batch_size),
            "--workers", str(args.workers),
            "--save-dir", str(args.output_dir / "_resnet_gde"),
        ], root

    return [
        py, "run_msde_gde.py", "--feature-root", str(getattr(args, "ours_feature_root", root / "features")),
        "--save-dir", str(args.output_dir / "_msde_gde"), "--seed", str(args.seed),
        "--datasets", dataset,
    ], root


def post_command_for(method: Method, dataset: str, args: argparse.Namespace) -> tuple[list[str], Path] | None:
    root = Path(__file__).resolve().parent
    py = sys.executable

    if method.family == "reconstruction":
        return [
            py, "test.py", "-d", dataset, "-m", method.native_name,
            "-g", str(args.gpu), "-f", str(args.fold), "-save",
            "--result-root", str(args.reconstruction_root),
        ], root / "reconstruction"

    if method.family == "one_stage":
        return [
            py, "med_evaluation.py", "-d", dataset, "-s", method.native_name,
            "-g", str(args.gpu), "-f", str(args.fold),
        ], root / "ssl" / "one_stage"

    return None


def score_path(method: Method, dataset: str, args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parent

    if method.family == "reconstruction":
        return Path(args.reconstruction_root) / dataset / method.native_name / f"fold_{args.fold}" / "test_results" / "test_scores.npz"
    if method.family == "one_stage":
        return root / "ssl" / "one_stage" / "output" / dataset / f"fold_{args.fold}" / f"{method.native_name}_test_scores.csv"
    if method.family == "two_stage":
        class_name = {"normal": "CutPasteNormal", "anatpaste": "AnatPaste"}[method.native_name]
        return Path(args.two_stage_model_dir) / dataset / class_name / "test_scores.csv"
    if method.family == "frozen_gde":
        return Path(args.output_dir) / "_resnet_gde" / method.native_name / dataset / "test_scores.csv"
    return Path(args.output_dir) / "_msde_gde" / dataset / "biomedclip" / "test_scores.csv"


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return data["names"], data["labels"], data["scores"]

    frame = pd.read_csv(path)
    names = frame["name"].to_numpy() if "name" in frame else np.arange(len(frame))
    return names, frame["label"].to_numpy(), frame["score"].to_numpy()


def load_existing_results(output_dir: Path) -> list[dict]:
    """Load successful per-run results so subset runs can be resumed safely."""
    results = []
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return results
    for result_path in runs_dir.glob("*/*/result.json"):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            if result.get("method") and result.get("dataset"):
                results.append(result)
        except (OSError, json.JSONDecodeError):
            continue
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--methods", nargs="+", default=[m.name for m in METHODS])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25,
                        help="Two-stage training epochs; use 100 for the original default")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--include-feature-extraction", action="store_true",
                        help="Include BiomedCLIP extraction in the MSDE runtime")
    parser.add_argument("--reconstruction-root", default=r"C:\Users\user\Pritam\MediAnomaly\Experiment\MedIAnomaly")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "benchmark_results")
    parser.add_argument("--two-stage-model-dir", type=Path,
                        default=Path("results") / "benchmark_results" / "two_stage_models")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.two_stage_model_dir = args.two_stage_model_dir.resolve()
    args.reconstruction_root = Path(args.reconstruction_root).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.datasets == ["all"]:
        args.datasets = DATASETS
    unknown_datasets = [name for name in args.datasets if name not in DATASETS]
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}. Available: {DATASETS}")
    selected = {m.name: m for m in METHODS}
    if args.methods == ["all"]:
        args.methods = [m.name for m in METHODS]
    unknown = [name for name in args.methods if name not in selected]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(selected)}")

    # Preserve successful methods from an earlier interrupted run. Results
    # generated in this invocation overwrite the matching method/dataset pair.
    existing_rows = load_existing_results(args.output_dir)
    rows = []
    for method_name in tqdm(args.methods, desc="Benchmark methods", unit="method"):
        method = selected[method_name]
        for dataset in tqdm(args.datasets, desc=method_name, unit="dataset", leave=False):
            run_id = f"{safe_name(method.name)}__{dataset}"
            log_path = args.output_dir / "logs" / f"{run_id}.log"
            print(f"\n[{method.name}] {dataset}")

            command, cwd = command_for(method, dataset, args)
            if method.family == "ours" and args.include_feature_extraction:
                args.ours_feature_root = args.output_dir / "_features"
                extraction_command = [
                    sys.executable, "extract_features.py",
                    "--dataset", dataset, "--backbone", "biomedclip",
                    "--gpu", str(args.gpu), "--batch-size", str(args.feature_batch_size),
                    "--save-root", str(args.ours_feature_root),
                ]
                extract_code, extract_runtime = run_command(
                    extraction_command, Path(__file__).resolve().parent,
                    log_path.with_name(log_path.stem + "__features.log")
                )
                if extract_code != 0:
                    returncode, train_runtime = extract_code, extract_runtime
                else:
                    command, cwd = command_for(method, dataset, args)
                    score_code, score_runtime = run_command(command, cwd, log_path)
                    returncode, train_runtime = score_code, extract_runtime + score_runtime
            else:
                returncode, train_runtime = run_command(command, cwd, log_path)

            total_runtime = train_runtime
            post = post_command_for(method, dataset, args)
            if returncode == 0 and post is not None:
                post_returncode, eval_runtime = run_command(
                    post[0], post[1], log_path.with_name(log_path.stem + "__eval.log")
                )
                returncode = post_returncode
                total_runtime += eval_runtime

            score_file = score_path(method, dataset, args)
            result = {
                "method": method.name,
                "dataset": dataset,
                "encoder": (
                    "biomedclip" if method.family == "ours" else
                    (method.native_name if method.family == "frozen_gde" else "native")
                ),
                "seed": args.seed,
                "fold": args.fold,
                "returncode": returncode,
                "runtime_seconds": total_runtime,
                "runtime_scope": (
                    "feature_extraction_plus_gde" if method.family == "frozen_gde" else
                    ("train_plus_eval" if method.family != "ours" else
                    ("feature_extraction_plus_scoring" if args.include_feature_extraction
                     else "feature_scoring_only"))
                ),
                "score_file": str(score_file),
            }

            if returncode == 0 and score_file.exists():
                names, labels, scores = load_scores(score_file)
                result.update(compute_metrics(labels, scores))
                result_path = args.output_dir / "runs" / safe_name(method.name) / dataset / "result.json"
                save_benchmark_result(result_path, result)
                print(f"  AUC={result['auc']:.5f} AP={result['ap']:.5f} P@N={result['precision_at_n']:.5f}")
            else:
                result["error"] = "command failed" if returncode else "score file not found"
                print(f"  WARNING: {result['error']}: {score_file}")

            rows.append(result)
            if returncode != 0 and not args.continue_on_error:
                raise RuntimeError(f"Benchmark command failed; see {log_path}")

    merged = {
        (row.get("method"), row.get("dataset")): row
        for row in existing_rows
    }
    for row in rows:
        merged[(row.get("method"), row.get("dataset"))] = row
    frame = pd.DataFrame(list(merged.values()))
    frame.to_csv(args.output_dir / "all_results.csv", index=False)
    frame.groupby("method", dropna=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        precision_at_n_mean=("precision_at_n", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).reset_index().to_csv(args.output_dir / "summary_by_method.csv", index=False)
    frame.groupby("dataset", dropna=False).agg(
        auc_mean=("auc", "mean"), ap_mean=("ap", "mean"),
        precision_at_n_mean=("precision_at_n", "mean"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
    ).reset_index().to_csv(args.output_dir / "summary_by_dataset.csv", index=False)
    print(f"\nSaved benchmark results to {args.output_dir}")


if __name__ == "__main__":
    main()
