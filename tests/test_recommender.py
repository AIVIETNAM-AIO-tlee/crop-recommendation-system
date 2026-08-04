import numpy as np
import pytest
from models.base_knn import CropKNNRecommender
from models.kmeanSearch_knn import KMeansSearchKNNRecommender


@pytest.fixture
def toy_data():
    X = np.array([
        [0.0, 0.0], [0.1, 0.0],
        [5.0, 5.0], [5.1, 5.0],
        [10.0, 0.0], [10.1, 0.0],
    ])
    y = np.array(["rice", "rice", "maize", "maize", "cotton", "cotton"])
    return X, y


def test_baseline_returns_nearest_unique_labels(toy_data):
    X, y = toy_data
    model = CropKNNRecommender(metric="euclidean", neighbors_per_crop=1).fit(X, y)
    recs = model.recommend([[0.05, 0.0]], top_k=3)[0].tolist()
    assert recs[0] == "rice"
    assert len(recs) == len(set(recs)) == 3


def test_top_k_larger_than_available_labels_is_clipped(toy_data):
    X, y = toy_data
    model = CropKNNRecommender().fit(X, y)
    recs = model.recommend([[0.0, 0.0]], top_k=99)[0].tolist()
    assert len(recs) == 3


def test_hybrid_returns_valid_unique_labels_and_diagnostics(toy_data):
    X, y = toy_data
    model = KMeansSearchKNNRecommender(
        metric="euclidean", neighbors_per_crop=1, n_clusters=3, n_probe=1
    ).fit(X, y)
    recs, diagnostics = model.recommend([[5.0, 5.0]], top_k=3, return_diagnostics=True)
    recs = recs[0].tolist()
    assert recs[0] == "maize"
    assert len(recs) == len(set(recs)) == 3
    assert diagnostics["candidate_count"][0] <= len(X)
    assert set(diagnostics["timing_ms"]) == {
        "centroid_distance", "cluster_selection", "candidate_distance", "ranking"
    }
