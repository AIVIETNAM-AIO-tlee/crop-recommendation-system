import numpy as np
import pytest
from evaluation.metrics import ranking_metrics_at_k
from evaluation.bootstrap import bootstrap_metric_ci, paired_bootstrap_difference


def test_multilabel_ranking_metrics():
    truth = [["rice", "maize"]]
    recs = [["rice", "cotton", "maize"]]
    aggregate, per_query = ranking_metrics_at_k(truth, recs, k=3)
    assert aggregate["Precision@K"] == pytest.approx(2 / 3)
    assert aggregate["Recall@K"] == pytest.approx(1.0)
    assert aggregate["MAP@K"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert 0 < aggregate["NDCG@K"] <= 1
    assert per_query.loc[0, "relevant_count"] == 2


def test_bootstrap_ci_contains_constant_value():
    ci = bootstrap_metric_ci(np.ones(20) * 0.75, n_resamples=100)
    assert ci["estimate"] == pytest.approx(0.75)
    assert ci["ci_low"] == pytest.approx(0.75)
    assert ci["ci_high"] == pytest.approx(0.75)


def test_paired_bootstrap_detects_challenger_improvement():
    baseline = np.zeros(30)
    challenger = np.ones(30) * 0.1
    result = paired_bootstrap_difference(baseline, challenger, n_resamples=200)
    assert result["conclusion"] == "challenger_better"
