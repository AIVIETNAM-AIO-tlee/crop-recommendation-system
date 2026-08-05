from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd
from config import TOP_K


def _as_relevant_set(value) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(item) for item in value}
    return {str(value)}


def ranking_metrics_at_k(y_true, recommendations, k: int = TOP_K):
    """Compute Top-K metrics for either single-label or multi-label ground truth."""
    rows = []
    for truth, recs in zip(list(y_true), recommendations):
        relevant = _as_relevant_set(truth)
        recs = list(dict.fromkeys(map(str, list(recs)[:k])))
        hits = np.asarray([1.0 if item in relevant else 0.0 for item in recs])

        precision = hits.sum() / k
        recall = hits.sum() / len(relevant) if relevant else 0.0
        hit_rate = float(hits.sum() > 0)

        precisions = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        ap_denom = min(len(relevant), k)
        ap = float((precisions * hits).sum() / ap_denom) if ap_denom else 0.0

        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = float((hits * discounts).sum())
        ideal_hits = min(len(relevant), k)
        idcg = float(discounts[:ideal_hits].sum()) if ideal_hits else 0.0
        ndcg = dcg / idcg if idcg else 0.0

        first_rank = next((idx + 1 for idx, hit in enumerate(hits) if hit), np.nan)
        rows.append({
            "Precision@K": precision,
            "Recall@K": recall,
            "AP@K": ap,
            "HitRate@K": hit_rate,
            "NDCG@K": ndcg,
            "rank": first_rank,
            "relevant_count": len(relevant),
        })

    per_query = pd.DataFrame(rows)
    aggregate = {
        "Precision@K": per_query["Precision@K"].mean(),
        "Recall@K": per_query["Recall@K"].mean(),
        "MAP@K": per_query["AP@K"].mean(),
        "HitRate@K": per_query["HitRate@K"].mean(),
        "NDCG@K": per_query["NDCG@K"].mean(),
        "MeanRank_when_hit": per_query["rank"].mean(),
    }
    return aggregate, per_query
