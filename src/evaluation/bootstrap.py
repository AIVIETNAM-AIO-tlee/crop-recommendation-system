from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_metric_ci(
    values,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def paired_bootstrap_difference(
    baseline_values,
    challenger_values,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> dict[str, float | str]:
    baseline = np.asarray(baseline_values, dtype=float)
    challenger = np.asarray(challenger_values, dtype=float)
    if baseline.shape != challenger.shape or baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("paired arrays must be non-empty and have the same one-dimensional shape")

    diff = challenger - baseline
    ci = bootstrap_metric_ci(diff, confidence, n_resamples, random_state)
    if ci["ci_low"] > 0:
        conclusion = "challenger_better"
    elif ci["ci_high"] < 0:
        conclusion = "baseline_better"
    else:
        conclusion = "no_clear_difference"
    return {**ci, "conclusion": conclusion}


def confidence_interval_table(model_details: dict, metrics=("AP@K", "NDCG@K"), **kwargs):
    rows = []
    for model_name, detail in model_details.items():
        per_query = detail["per_query"] if isinstance(detail, dict) else detail
        for metric in metrics:
            ci = bootstrap_metric_ci(per_query[metric].to_numpy(), **kwargs)
            rows.append({"model": model_name, "metric": metric, **ci})
    return pd.DataFrame(rows)
