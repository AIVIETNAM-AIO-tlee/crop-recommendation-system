from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CropSuitabilityProfiles:
    medians: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    scales: pd.Series


def fit_crop_suitability_profiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> CropSuitabilityProfiles:
    frame = X_train.copy()
    frame["__label__"] = np.asarray(y_train).astype(str)
    grouped = frame.groupby("__label__")
    feature_columns = X_train.columns
    medians = grouped[list(feature_columns)].median()
    lower = grouped[list(feature_columns)].quantile(lower_quantile)
    upper = grouped[list(feature_columns)].quantile(upper_quantile)
    scales = X_train.std(ddof=0).replace(0, 1.0)
    return CropSuitabilityProfiles(medians, lower, upper, scales)


def build_weak_multilabel_ground_truth(
    X_query: pd.DataFrame,
    original_labels,
    profiles: CropSuitabilityProfiles,
    max_labels: int = 3,
    min_features_in_range: int = 5,
) -> list[list[str]]:
    """Create weak 2--3 crop labels using learned soil/climate suitability envelopes.

    The observed label is always retained. Alternatives must satisfy at least
    ``min_features_in_range`` crop-specific 10--90% feature ranges and are ranked
    by standardized distance to the crop median. This is a reproducible proxy,
    not expert-validated agronomic annotation.
    """
    if max_labels < 1:
        raise ValueError("max_labels must be at least 1")
    labels = profiles.medians.index.to_numpy(dtype=str)
    outputs: list[list[str]] = []

    for (_, row), original in zip(X_query.iterrows(), original_labels):
        original = str(original)
        in_range = ((row >= profiles.lower) & (row <= profiles.upper)).sum(axis=1)
        normalized = (profiles.medians - row).div(profiles.scales, axis=1)
        distances = np.sqrt((normalized**2).mean(axis=1))

        eligible = distances[in_range >= min_features_in_range].sort_values().index.tolist()
        ranked = [original] + [label for label in eligible if label != original]

        # Ensure at least two labels when another crop exists; fallback uses the
        # nearest profile but is explicitly still a weak/pseudo label.
        if len(ranked) < min(2, max_labels) and len(labels) > 1:
            fallback = [label for label in distances.sort_values().index if label != original]
            ranked.extend(fallback[: min(2, max_labels) - len(ranked)])

        outputs.append(list(dict.fromkeys(ranked))[:max_labels])
    return outputs
