from __future__ import annotations

from time import perf_counter
import numpy as np
from config import TOP_K
from evaluation.metrics import ranking_metrics_at_k


def evaluate_recommender(model, X_eval, y_eval, top_k: int = TOP_K):
    start = perf_counter()
    recommendations, diagnostics = model.recommend(
        X_eval, top_k=top_k, return_diagnostics=True
    )
    elapsed = perf_counter() - start

    aggregate, per_query = ranking_metrics_at_k(y_eval, recommendations, k=top_k)
    aggregate.update(
        {
            "Latency_ms_per_query": elapsed * 1000 / len(X_eval),
            "Avg_candidates": float(np.mean(diagnostics["candidate_count"])),
            "Candidate_ratio": float(np.mean(diagnostics["candidate_ratio"])),
            "Avg_clusters_scanned": float(np.nanmean(diagnostics["clusters_scanned"]))
            if not np.all(np.isnan(diagnostics["clusters_scanned"]))
            else np.nan,
        }
    )
    for component, values in diagnostics.get("timing_ms", {}).items():
        aggregate[f"Latency_{component}_ms"] = float(np.mean(values))
    return aggregate, per_query, recommendations, diagnostics
