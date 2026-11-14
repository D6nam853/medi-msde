"""Shared metrics and result-writing helpers for the MedIAnomaly benchmark."""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_n(labels: Iterable, scores: Iterable) -> tuple[float, int]:
    """Return precision at N, where N is the number of positive labels."""
    y_true = np.asarray(labels).astype(int)
    y_score = np.asarray(scores, dtype=float)
    n_anomalies = int(np.sum(y_true == 1))

    if n_anomalies == 0 or len(y_true) == 0:
        return 0.0, n_anomalies

    top_n = np.argsort(y_score)[-n_anomalies:]
    return float(np.mean(y_true[top_n] == 1)), n_anomalies


def compute_metrics(labels: Iterable, scores: Iterable) -> dict:
    y_true = np.asarray(labels).astype(int)
    y_score = np.asarray(scores, dtype=float)
    p_at_n, n_anomalies = precision_at_n(y_true, y_score)

    if len(np.unique(y_true)) < 2:
        auc = 0.0
        ap = 0.0
    else:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))

    return {
        "auc": auc,
        "ap": ap,
        "precision_at_n": p_at_n,
        "n_anomalies": n_anomalies,
    }


def save_score_file(path: str, names: Iterable, labels: Iterable,
                    scores: Iterable) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "name": list(names),
        "label": np.asarray(labels).astype(int),
        "score": np.asarray(scores, dtype=float),
    }).to_csv(path, index=False)


def save_benchmark_result(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
