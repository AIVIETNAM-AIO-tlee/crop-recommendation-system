from __future__ import annotations

import numpy as np
import pandas as pd
from config import TOP_K


def ranking_metrics_at_k(y_true, recommendations, k: int = TOP_K):
    """Calculate single-relevant-item ranking metrics for Top-K recommendations."""
    rows = []
    for truth, recs in zip(np.asarray(y_true), recommendations):
        recs = list(recs)[:k]
        rank = recs.index(truth) + 1 if truth in recs else None
        hit = float(rank is not None)
        rows.append(
            {
                "Precision@K": hit / k,
                "Recall@K": hit,
                "AP@K": (1.0 / rank) if rank is not None else 0.0,
                "HitRate@K": hit,
                "NDCG@K": (1.0 / np.log2(rank + 1)) if rank is not None else 0.0,
                "rank": rank if rank is not None else np.nan,
            }
        )

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