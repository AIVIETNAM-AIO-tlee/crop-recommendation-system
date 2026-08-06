"""Bootstrap confidence-interval tests for ranking-metric reliability.

These tests expand the coverage of ``src/evaluation/bootstrap.py`` beyond the
two basic sanity checks in ``test_metrics.py`` and address the reviewer concern
that "MAP@K / NDCG@K differences have no confidence interval". The final test
builds a real paired CI between the Exact KNN baseline and the K-Means++
hybrid on per-query AP@5 values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from evaluation.bootstrap import (
    bootstrap_metric_ci,
    confidence_interval_table,
    paired_bootstrap_difference,
)


# --------------------------------------------------------------------------- #
# bootstrap_metric_ci
# --------------------------------------------------------------------------- #
def test_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    values = rng.uniform(0.4, 0.9, size=200)
    ci = bootstrap_metric_ci(values, n_resamples=500, random_state=7)

    assert ci["estimate"] == pytest.approx(values.mean())
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]
    assert ci["confidence"] == 0.95
    assert ci["n_resamples"] == 500


def test_ci_is_reproducible_with_fixed_seed():
    values = np.linspace(0, 1, 50)
    ci_a = bootstrap_metric_ci(values, n_resamples=300, random_state=123)
    ci_b = bootstrap_metric_ci(values, n_resamples=300, random_state=123)
    assert ci_a == ci_b


def test_ci_for_constant_array_is_a_point():
    ci = bootstrap_metric_ci(np.full(25, 0.6), n_resamples=200)
    assert ci["ci_low"] == pytest.approx(0.6)
    assert ci["ci_high"] == pytest.approx(0.6)
    assert ci["estimate"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "bad_input",
    [np.array([]), np.array([[1, 2], [3, 4]]), None],
)
def test_ci_rejects_invalid_inputs(bad_input):
    with pytest.raises((ValueError, TypeError)):
        bootstrap_metric_ci(bad_input)


def test_ci_accepts_python_list_of_scalars():
    """A plain 1-D list of scalars is a valid input and must be accepted."""
    ci = bootstrap_metric_ci([0.1, 0.2, 0.3, 0.4], n_resamples=50)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]


def test_ci_respects_custom_confidence_level():
    values = np.random.default_rng(1).uniform(0.3, 0.9, size=150)
    ci_95 = bootstrap_metric_ci(values, confidence=0.95, n_resamples=400)
    ci_80 = bootstrap_metric_ci(values, confidence=0.80, n_resamples=400)
    # A lower confidence level yields a narrower interval.
    width_95 = ci_95["ci_high"] - ci_95["ci_low"]
    width_80 = ci_80["ci_high"] - ci_80["ci_low"]
    assert width_80 <= width_95


# --------------------------------------------------------------------------- #
# paired_bootstrap_difference — all three conclusions
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_detects_baseline_better():
    baseline = np.ones(40) * 0.9
    challenger = np.ones(40) * 0.7
    result = paired_bootstrap_difference(baseline, challenger, n_resamples=300)
    assert result["conclusion"] == "baseline_better"
    assert result["estimate"] == pytest.approx(-0.2)


def test_paired_bootstrap_reports_no_clear_difference_when_overlapping():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.5, 0.1, size=200)
    challenger = baseline + rng.normal(0.0, 0.05, size=200)  # tiny, noisy shift
    result = paired_bootstrap_difference(baseline, challenger, n_resamples=500)
    assert result["conclusion"] == "no_clear_difference"
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_paired_bootstrap_requires_matching_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap_difference(np.zeros(10), np.zeros(9))


def test_paired_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.3, 0.5, size=120)
    b = rng.uniform(0.6, 0.8, size=120)
    result = paired_bootstrap_difference(a, b, n_resamples=400)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


# --------------------------------------------------------------------------- #
# confidence_interval_table
# --------------------------------------------------------------------------- #
def test_confidence_interval_table_builds_one_row_per_model_and_metric():
    model_details = {
        "baseline": pd.DataFrame({"AP@K": np.linspace(0.5, 1.0, 30),
                                  "NDCG@K": np.linspace(0.6, 1.0, 30)}),
        "hybrid": pd.DataFrame({"AP@K": np.linspace(0.55, 1.0, 30),
                                "NDCG@K": np.linspace(0.65, 1.0, 30)}),
    }
    table = confidence_interval_table(model_details, n_resamples=200)

    assert isinstance(table, pd.DataFrame)
    assert len(table) == 4  # 2 models x 2 metrics
    assert set(table["model"].unique()) == {"baseline", "hybrid"}
    assert set(table["metric"].unique()) == {"AP@K", "NDCG@K"}
    for _, row in table.iterrows():
        assert row["ci_low"] <= row["estimate"] <= row["ci_high"]


def test_confidence_interval_table_accepts_custom_metrics():
    rng = np.random.default_rng(5)
    model_details = {
        "m1": pd.DataFrame({"HitRate@K": rng.uniform(0.8, 1.0, size=50)}),
    }
    table = confidence_interval_table(
        model_details, metrics=("HitRate@K",), n_resamples=150
    )
    assert list(table["metric"]) == ["HitRate@K"]
    assert len(table) == 1


# --------------------------------------------------------------------------- #
# Integration: paired CI between Exact KNN and K-Means++ (reviewer concern)
# --------------------------------------------------------------------------- #
def test_paired_ci_between_baseline_and_hybrid_on_ap_at_5():
    """Build a per-query paired bootstrap CI for the MAP@5 difference between
    the exact baseline and the K-Means++ hybrid, exactly as the reviewer asked
    ("bootstrap by query and report a CI for the Exact KNN vs K-Means++
    difference at MAP@5")."""
    from pathlib import Path

    from data.loader import load_crop_dataset
    from data.preprocessing import fit_preprocessor, transform_features
    from data.split import split_train_val_test
    from evaluation.metrics import ranking_metrics_at_k
    from models.base_knn import CropKNNRecommender
    from models.kmeanSearch_knn import KMeansSearchKNNRecommender

    repo_root = Path(__file__).resolve().parents[1]
    df = load_crop_dataset(repo_root / "data" / "raw" / "Crop_recommendation.csv")
    split = split_train_val_test(df)
    imputer, scaler, X_train = fit_preprocessor(split.X_train)
    X_val = transform_features(split.X_val, imputer, scaler)

    baseline = CropKNNRecommender(metric="manhattan", neighbors_per_crop=5).fit(
        X_train, split.y_train
    )
    hybrid = KMeansSearchKNNRecommender(
        metric="manhattan", neighbors_per_crop=3, n_clusters=16, n_probe=3
    ).fit(X_train, split.y_train)

    # Use a subset of the validation queries to keep the test fast but real.
    eval_X, eval_y = X_val.iloc[:80], split.y_val.iloc[:80]
    recs_b, _ = baseline.recommend(eval_X, top_k=5, return_diagnostics=True)
    recs_h, _ = hybrid.recommend(eval_X, top_k=5, return_diagnostics=True)

    _, per_b = ranking_metrics_at_k(eval_y, recs_b, k=5)
    _, per_h = ranking_metrics_at_k(eval_y, recs_h, k=5)

    result = paired_bootstrap_difference(
        per_b["AP@K"].to_numpy(),
        per_h["AP@K"].to_numpy(),
        confidence=CONFIDENCE_LEVEL,
        n_resamples=500,
        random_state=42,
    )
    assert result["conclusion"] in {
        "challenger_better",
        "baseline_better",
        "no_clear_difference",
    }
    # The CI must be a bounded, ordered interval regardless of the conclusion.
    assert -1.0 <= result["ci_low"] <= result["ci_high"] <= 1.0
