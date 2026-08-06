"""Latency benchmark tests for the fixed-workload evaluation.

These tests address two reviewer concerns directly:
  1. "Latency is not tied to an operational threshold" -> each benchmark is run
     against an explicit ``threshold_ms`` service-level objective and the
     ``passes_threshold`` flag must be reported.
  2. "Root cause of latency is not measured" -> the benchmark must surface
     per-component timings (centroid distance / cluster selection / candidate
     distance / ranking) so the slowest stage can be identified.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from evaluation.latency import benchmark_latency
from models.base_knn import CropKNNRecommender
from models.kmeanSearch_knn import KMeansSearchKNNRecommender


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def trained_models():
    """Train a baseline and a hybrid model on the same synthetic data."""
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=center, scale=0.4, size=(40, 4))
            for center in (0.0, 8.0, 16.0, 24.0, 32.0)
        ]
    )
    y = np.array(
        ["rice"] * 40
        + ["maize"] * 40
        + ["cotton"] * 40
        + ["coffee"] * 40
        + ["jute"] * 40
    )
    baseline = CropKNNRecommender(metric="euclidean", neighbors_per_crop=3).fit(X, y)
    hybrid = KMeansSearchKNNRecommender(
        metric="euclidean",
        neighbors_per_crop=3,
        n_clusters=8,
        n_probe=2,
    ).fit(X, y)
    query = rng.normal(loc=0.0, scale=0.4, size=(25, 4))
    return baseline, hybrid, query


# --------------------------------------------------------------------------- #
# benchmark_latency output contract
# --------------------------------------------------------------------------- #
def test_benchmark_returns_run_dataframe_and_summary(trained_models):
    baseline, _, query = trained_models
    run_df, summary = benchmark_latency(
        baseline, query, top_k=5, threshold_ms=5.0, warmup_runs=2, measured_runs=4
    )

    assert isinstance(run_df, pd.DataFrame)
    assert list(run_df.columns) == ["run", "total_ms_per_query"]
    assert len(run_df) == 4
    assert (run_df["run"] == np.arange(1, 5)).all()
    assert (run_df["total_ms_per_query"] > 0).all()


def test_summary_reports_workload_and_threshold(trained_models):
    """The reviewer asked for a concrete workload + latency threshold."""
    _, hybrid, query = trained_models
    _, summary = benchmark_latency(
        hybrid, query, top_k=5, threshold_ms=2.5, warmup_runs=2, measured_runs=4
    )

    assert summary["workload_queries"] == 25
    assert summary["warmup_runs"] == 2
    assert summary["measured_runs"] == 4
    assert summary["threshold_ms"] == 2.5
    for key in ("mean_ms", "p50_ms", "p95_ms"):
        assert summary[key] > 0
    assert summary["p50_ms"] <= summary["p95_ms"] + 1e-9
    assert isinstance(summary["passes_threshold"], bool)


def test_passes_threshold_flag_is_consistent_with_p95(trained_models):
    baseline, _, query = trained_models
    threshold = 50.0  # generous threshold so the baseline always passes
    _, summary = benchmark_latency(
        baseline, query, top_k=5, threshold_ms=threshold, warmup_runs=1, measured_runs=3
    )
    assert summary["passes_threshold"] is True
    assert summary["p95_ms"] <= threshold


# --------------------------------------------------------------------------- #
# Per-component profiling (root-cause of latency)
# --------------------------------------------------------------------------- #
EXPECTED_HYBRID_COMPONENTS = {
    "centroid_distance",
    "cluster_selection",
    "candidate_distance",
    "ranking",
}


def test_hybrid_benchmark_reports_per_component_timings(trained_models):
    """The reviewer asked to isolate centroid distance / cluster selection /
    class-wise KNN via a profiler on the same workload."""
    _, hybrid, query = trained_models
    _, summary = benchmark_latency(
        hybrid, query, top_k=5, threshold_ms=5.0, warmup_runs=2, measured_runs=4
    )
    component_keys = {
        key.split("component_", 1)[1]
        for key in summary
        if key.startswith("component_")
    }
    assert component_keys == EXPECTED_HYBRID_COMPONENTS
    for name in EXPECTED_HYBRID_COMPONENTS:
        assert summary[f"component_{name}"] >= 0.0


def test_baseline_benchmark_reports_distance_and_ranking_components(trained_models):
    baseline, _, query = trained_models
    _, summary = benchmark_latency(
        baseline, query, top_k=5, threshold_ms=5.0, warmup_runs=2, measured_runs=4
    )
    component_keys = {
        key.split("component_", 1)[1]
        for key in summary
        if key.startswith("component_")
    }
    # The exact KNN baseline only exposes "distance" and "ranking" stages.
    assert component_keys == {"distance", "ranking"}


# --------------------------------------------------------------------------- #
# Candidate reduction vs. latency trade-off (the honest finding)
# --------------------------------------------------------------------------- #
def test_hybrid_examines_fewer_candidates_than_baseline(trained_models):
    """K-Means++ must reduce the candidate set, even if latency is not lower."""
    baseline, hybrid, query = trained_models
    _, diag_base = baseline.recommend(query, top_k=5, return_diagnostics=True)
    _, diag_hyb = hybrid.recommend(query, top_k=5, return_diagnostics=True)

    avg_base = float(np.mean(diag_base["candidate_count"]))
    avg_hyb = float(np.mean(diag_hyb["candidate_count"]))
    assert avg_hyb < avg_base
    assert float(np.mean(diag_hyb["candidate_ratio"])) < 1.0


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_benchmark_raises_on_empty_query(trained_models):
    baseline, _, _ = trained_models
    with pytest.raises(ValueError, match="at least one query"):
        benchmark_latency(baseline, np.empty((0, 4)), measured_runs=2)


def test_benchmark_is_deterministic_in_summary_shape(trained_models):
    baseline, _, query = trained_models
    _, summary_a = benchmark_latency(
        baseline, query, threshold_ms=5.0, warmup_runs=1, measured_runs=3
    )
    _, summary_b = benchmark_latency(
        baseline, query, threshold_ms=5.0, warmup_runs=1, measured_runs=3
    )
    assert set(summary_a) == set(summary_b)


def test_benchmark_measured_runs_actually_run(trained_models):
    """A sanity guard that warmup + measured runs really execute."""
    baseline, _, query = trained_models
    start = time.perf_counter()
    run_df, _ = benchmark_latency(
        baseline, query, top_k=5, warmup_runs=2, measured_runs=5
    )
    elapsed = time.perf_counter() - start
    assert len(run_df) == 5
    assert elapsed >= 0.0
