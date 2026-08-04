import numpy as np
import pytest
from sklearn.metrics import pairwise_distances


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("euclidean", 5.0),
        ("manhattan", 7.0),
        ("cosine", 1.0),
    ],
)
def test_pairwise_distance_known_values(metric, expected):
    x = np.array([[0.0, 0.0]]) if metric != "cosine" else np.array([[1.0, 0.0]])
    y = np.array([[3.0, 4.0]]) if metric != "cosine" else np.array([[0.0, 1.0]])
    value = pairwise_distances(x, y, metric=metric)[0, 0]
    assert value == pytest.approx(expected)


def test_distance_is_zero_for_identical_vectors():
    x = np.array([[1.5, -2.0, 0.25]])
    for metric in ("euclidean", "manhattan", "cosine"):
        assert pairwise_distances(x, x, metric=metric)[0, 0] == pytest.approx(0.0)
