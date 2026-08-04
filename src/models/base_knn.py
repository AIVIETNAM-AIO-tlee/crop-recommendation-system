from __future__ import annotations

import numpy as np
from time import perf_counter
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from config import DISTANCE_METRICS, TOP_K


class CropKNNRecommender:
    """Exact class-wise KNN recommender that searches the full training set."""

    def __init__(self, metric="euclidean", neighbors_per_crop=3):
        if metric not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric
        self.neighbors_per_crop = int(neighbors_per_crop)

    def _prepare_space(self, X):
        X = np.asarray(X, dtype=float)
        return normalize(X) if self.metric == "cosine" else X

    def fit(self, X, y):
        self.X_train_ = self._prepare_space(X)
        self.y_train_ = np.asarray(y).astype(str)
        self.labels_ = np.array(sorted(np.unique(self.y_train_)))
        self.class_indices_ = {
            label: np.flatnonzero(self.y_train_ == label) for label in self.labels_
        }
        return self

    def recommend(self, X, top_k=TOP_K, return_diagnostics=False):
        Xq = self._prepare_space(X)
        distance_start = perf_counter()
        distances = pairwise_distances(Xq, self.X_train_, metric=self.metric)
        distance_elapsed = (perf_counter() - distance_start) * 1000 / max(len(Xq), 1)

        ranking_start = perf_counter()
        crop_scores = np.empty((len(Xq), len(self.labels_)), dtype=float)
        for j, label in enumerate(self.labels_):
            class_distances = distances[:, self.class_indices_[label]]
            n = min(self.neighbors_per_crop, class_distances.shape[1])
            nearest = np.partition(class_distances, kth=n - 1, axis=1)[:, :n]
            crop_scores[:, j] = nearest.mean(axis=1)

        top_k = min(top_k, len(self.labels_))
        top_indices = np.argsort(crop_scores, axis=1)[:, :top_k]
        recommendations = self.labels_[top_indices]
        selected_distances = np.take_along_axis(crop_scores, top_indices, axis=1)

        # Convert distance to a normalized compatibility score for presentation only.
        compatibility = 1.0 / (selected_distances + 1e-9)
        compatibility = compatibility / compatibility.sum(axis=1, keepdims=True)
        ranking_elapsed = (perf_counter() - ranking_start) * 1000 / max(len(Xq), 1)

        if return_diagnostics:
            diagnostics = {
                "candidate_count": np.full(len(Xq), len(self.X_train_), dtype=int),
                "candidate_ratio": np.ones(len(Xq)),
                "clusters_scanned": np.full(len(Xq), np.nan),
                "selected_distances": selected_distances,
                "compatibility": compatibility,
                "timing_ms": {
                    "distance": np.full(len(Xq), distance_elapsed),
                    "ranking": np.full(len(Xq), ranking_elapsed),
                },
            }
            return recommendations, diagnostics
        return recommendations