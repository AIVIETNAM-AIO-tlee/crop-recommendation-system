from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from config import DISTANCE_METRICS, RANDOM_STATE, TOP_K


class KMeansSearchKNNRecommender:
    """Approximate KNN recommender accelerated by a K-Means++ centroid index."""

    def __init__(
        self,
        metric="euclidean",
        neighbors_per_crop=3,
        n_clusters=22,
        n_probe=2,
        random_state=RANDOM_STATE,
    ):
        if metric not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")
        self.metric = metric
        self.neighbors_per_crop = int(neighbors_per_crop)
        self.n_clusters = int(n_clusters)
        self.n_probe = int(n_probe)
        self.random_state = random_state

    def _prepare_space(self, X):
        X = np.asarray(X, dtype=float)
        return normalize(X) if self.metric == "cosine" else X

    def fit(self, X, y):
        self.X_train_ = self._prepare_space(X)
        self.y_train_ = np.asarray(y).astype(str)
        self.labels_ = np.array(sorted(np.unique(self.y_train_)))

        effective_clusters = min(self.n_clusters, len(self.X_train_))
        self.kmeans_ = KMeans(
            n_clusters=effective_clusters,
            init="k-means++",
            n_init=10,
            random_state=self.random_state,
        )
        self.cluster_ids_ = self.kmeans_.fit_predict(self.X_train_)
        self.centroids_ = self.kmeans_.cluster_centers_
        self.cluster_members_ = {
            cid: np.flatnonzero(self.cluster_ids_ == cid)
            for cid in range(effective_clusters)
        }
        return self

    def recommend(self, X, top_k=TOP_K, return_diagnostics=False):
        Xq = self._prepare_space(X)
        centroid_distances = pairwise_distances(Xq, self.centroids_, metric=self.metric)
        centroid_order = np.argsort(centroid_distances, axis=1)

        all_recommendations = []
        selected_distances_all = []
        compatibility_all = []
        candidate_counts = []
        clusters_scanned = []

        for row_idx, query in enumerate(Xq):
            chosen_clusters = []
            chosen_indices = np.array([], dtype=int)

            for cid in centroid_order[row_idx]:
                chosen_clusters.append(int(cid))
                chosen_indices = np.concatenate([
                    chosen_indices,
                    self.cluster_members_[int(cid)],
                ]).astype(int)

                distinct_labels = np.unique(self.y_train_[chosen_indices]).size
                enough_probes = len(chosen_clusters) >= self.n_probe
                enough_labels = distinct_labels >= min(top_k, len(self.labels_))
                if enough_probes and enough_labels:
                    break

            candidate_X = self.X_train_[chosen_indices]
            candidate_y = self.y_train_[chosen_indices]
            distances = pairwise_distances(
                query.reshape(1, -1), candidate_X, metric=self.metric
            ).ravel()

            label_scores = []
            for label in np.unique(candidate_y):
                label_distances = distances[candidate_y == label]
                n = min(self.neighbors_per_crop, len(label_distances))
                nearest = np.partition(label_distances, kth=n - 1)[:n]
                label_scores.append((label, float(nearest.mean())))

            label_scores.sort(key=lambda item: item[1])
            top_scores = label_scores[:top_k]
            recs = [label for label, _ in top_scores]
            dists = np.array([score for _, score in top_scores], dtype=float)

            comp = 1.0 / (dists + 1e-9)
            comp = comp / comp.sum()

            all_recommendations.append(recs)
            selected_distances_all.append(dists)
            compatibility_all.append(comp)
            candidate_counts.append(len(chosen_indices))
            clusters_scanned.append(len(chosen_clusters))

        recommendations = np.asarray(all_recommendations, dtype=object)

        if return_diagnostics:
            candidate_counts = np.asarray(candidate_counts)
            diagnostics = {
                "candidate_count": candidate_counts,
                "candidate_ratio": candidate_counts / len(self.X_train_),
                "clusters_scanned": np.asarray(clusters_scanned),
                "selected_distances": np.asarray(selected_distances_all),
                "compatibility": np.asarray(compatibility_all),
            }
            return recommendations, diagnostics
        return recommendations