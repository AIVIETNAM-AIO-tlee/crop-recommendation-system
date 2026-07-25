from __future__ import annotations

import pandas as pd
from config import DISTANCE_METRICS, TOP_K
from evaluation.evaluate import evaluate_recommender
from models.base_knn import CropKNNRecommender
from models.kmeanSearch_knn import KMeansSearchKNNRecommender


def benchmark_baseline_models(X_train_scaled, y_train, X_val_scaled, y_val):
    rows = []
    for metric in DISTANCE_METRICS:
        for neighbors_per_crop in [1, 3, 5, 10]:
            model = CropKNNRecommender(metric=metric, neighbors_per_crop=neighbors_per_crop).fit(
                X_train_scaled, y_train
            )
            metrics, _, _, _ = evaluate_recommender(model, X_val_scaled, y_val, top_k=TOP_K)
            rows.append(
                {
                    "model": "KNN baseline",
                    "metric": metric,
                    "neighbors_per_crop": neighbors_per_crop,
                    **metrics,
                }
            )

    baseline_validation = pd.DataFrame(rows)
    return baseline_validation.sort_values(
        ["metric", "MAP@K", "NDCG@K", "Latency_ms_per_query"],
        ascending=[True, False, False, True],
    )


def benchmark_hybrid_models(X_train_scaled, y_train, X_val_scaled, y_val):
    rows = []
    for metric in DISTANCE_METRICS:
        for n_clusters in [8, 12, 16, 22]:
            for n_probe in [1, 2, 3]:
                for neighbors_per_crop in [1, 3, 5]:
                    model = KMeansSearchKNNRecommender(
                        metric=metric,
                        neighbors_per_crop=neighbors_per_crop,
                        n_clusters=n_clusters,
                        n_probe=n_probe,
                    ).fit(X_train_scaled, y_train)
                    metrics, _, _, _ = evaluate_recommender(model, X_val_scaled, y_val, top_k=TOP_K)
                    rows.append(
                        {
                            "model": "KMeansSearch KNN",
                            "metric": metric,
                            "n_clusters": n_clusters,
                            "n_probe": n_probe,
                            "neighbors_per_crop": neighbors_per_crop,
                            **metrics,
                        }
                    )

    validation = pd.DataFrame(rows)
    return validation.sort_values(
        ["metric", "MAP@K", "NDCG@K", "Candidate_ratio", "Latency_ms_per_query"],
        ascending=[True, False, False, True, True],
    )