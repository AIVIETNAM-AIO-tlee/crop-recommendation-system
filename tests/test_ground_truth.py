import pandas as pd
from evaluation.ground_truth import (
    fit_crop_suitability_profiles,
    build_weak_multilabel_ground_truth,
)


def test_weak_ground_truth_keeps_original_and_adds_alternative():
    X = pd.DataFrame({
        "n": [10, 11, 50, 51],
        "p": [10, 11, 50, 51],
        "k": [10, 11, 50, 51],
        "temp": [20, 21, 30, 31],
        "humidity": [70, 71, 50, 51],
        "ph": [6, 6.1, 7, 7.1],
        "rain": [100, 101, 50, 51],
    })
    y = pd.Series(["rice", "rice", "maize", "maize"])
    profiles = fit_crop_suitability_profiles(X, y)
    labels = build_weak_multilabel_ground_truth(X.iloc[[0]], ["rice"], profiles, max_labels=3)
    assert labels[0][0] == "rice"
    assert 2 <= len(labels[0]) <= 3
    assert len(labels[0]) == len(set(labels[0]))
