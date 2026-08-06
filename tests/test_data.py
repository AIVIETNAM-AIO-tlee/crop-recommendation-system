"""Unit tests for the data pipeline: loading, splitting, and preprocessing.

These tests cover the concerns raised in review:
  * load/split/preprocess behavior (including stratification and leakage prevention),
  * graceful handling of a missing dataset file (relative path, no hard-coded Drive path),
  * preprocessing fit/transform consistency (imputer + scaler fitted only on train).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from config import TARGET_COLUMN
from data.loader import infer_feature_columns, load_crop_dataset
from data.preprocessing import (
    fit_preprocessor,
    transform_features,
    transform_query_features,
)
from data.split import SplitData, split_train_val_test


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small balanced DataFrame with 3 numeric features and 3 classes."""
    rng = np.random.default_rng(0)
    rows = []
    for label, center in (("rice", 0.0), ("maize", 10.0), ("cotton", 20.0)):
        block = rng.normal(loc=center, scale=0.5, size=(20, 3))
        for row in block:
            rows.append([float(row[0]), float(row[1]), float(row[2]), label])
    df = pd.DataFrame(rows, columns=["n", "p", "k", TARGET_COLUMN])
    return df


@pytest.fixture
def raw_csv(tmp_path) -> Path:
    """Write the sample DataFrame to a temporary CSV file and return its path."""
    csv_path = tmp_path / "sample.csv"
    sample_df = pd.DataFrame(
        {
            "N": [10, 20, 30],
            "P": [5, 15, 25],
            "K": [1, 2, 3],
            TARGET_COLUMN: ["rice", "maize", "cotton"],
            "Unnamed: 8": [np.nan, np.nan, np.nan],
            "Unnamed: 9": [np.nan, np.nan, np.nan],
        }
    )
    sample_df.to_csv(csv_path, index=False)
    return csv_path


# --------------------------------------------------------------------------- #
# load_crop_dataset / infer_feature_columns
# --------------------------------------------------------------------------- #
def test_load_dataset_normalizes_columns_and_drops_empty(raw_csv):
    df = load_crop_dataset(raw_csv)
    # Column names are lower-cased and stripped.
    assert all(isinstance(col, str) and col == col.strip().lower() for col in df.columns)
    # Empty columns "Unnamed: 8/9" must be removed.
    assert "Unnamed: 8" not in df.columns and "Unnamed: 9" not in df.columns
    assert TARGET_COLUMN in df.columns


def test_load_dataset_uses_relative_path_from_repo_root():
    """Regression guard: the real dataset must load via a repo-relative path,
    not a hard-coded Google Drive path."""
    repo_root = Path(__file__).resolve().parents[1]
    df = load_crop_dataset(repo_root / "data" / "raw" / "Crop_recommendation.csv")
    assert len(df) == 2200
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].nunique() == 22


def test_load_dataset_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_crop_dataset(missing)


def test_load_dataset_raises_when_target_column_absent(tmp_path):
    csv_path = tmp_path / "no_target.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="target column"):
        load_crop_dataset(csv_path)


def test_infer_feature_columns_excludes_target(sample_df):
    features = infer_feature_columns(sample_df)
    assert features == ["n", "p", "k"]
    assert TARGET_COLUMN not in features


# --------------------------------------------------------------------------- #
# split_train_val_test
# --------------------------------------------------------------------------- #
def test_split_returns_disjoint_sets_with_expected_sizes(sample_df):
    split = split_train_val_test(sample_df, test_size=0.2, val_size=0.1)

    assert isinstance(split, SplitData)
    total = len(sample_df)
    assert len(split.X_test) == round(total * 0.2)
    assert len(split.X_val) == round(total * 0.1)
    assert len(split.X_train) == total - len(split.X_val) - len(split.X_test)

    # No row index may appear in more than one partition (leakage prevention).
    train_idx = set(split.X_train.index)
    val_idx = set(split.X_val.index)
    test_idx = set(split.X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_preserves_class_proportions_stratified(sample_df):
    split = split_train_val_test(sample_df, test_size=0.2, val_size=0.1)
    original_counts = sample_df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    for partition in (split.y_train, split.y_val, split.y_test):
        part_counts = partition.value_counts(normalize=True).sort_index()
        pd.testing.assert_series_equal(part_counts, original_counts)


def test_split_is_reproducible_with_fixed_seed(sample_df):
    a = split_train_val_test(sample_df)
    b = split_train_val_test(sample_df)
    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


def test_split_features_and_labels_are_aligned(sample_df):
    split = split_train_val_test(sample_df)
    assert list(split.X_train.columns) == list(split.X_val.columns) == list(split.X_test.columns)
    assert (split.X_train.index == split.y_train.index).all()
    assert (split.X_val.index == split.y_val.index).all()
    assert (split.X_test.index == split.y_test.index).all()


# --------------------------------------------------------------------------- #
# fit_preprocessor / transform_features
# --------------------------------------------------------------------------- #
def test_fit_preprocessor_returns_fitted_imputer_and_scaler(sample_df):
    split = split_train_val_test(sample_df)
    imputer, scaler, X_train_scaled = fit_preprocessor(split.X_train)
    assert imputer.strategy == "median"
    assert isinstance(scaler, StandardScaler)
    assert list(X_train_scaled.columns) == list(split.X_train.columns)
    assert X_train_scaled.index.equals(split.X_train.index)


def test_transformed_features_have_zero_mean_unit_std(sample_df):
    split = split_train_val_test(sample_df)
    imputer, scaler, X_train_scaled = fit_preprocessor(split.X_train)
    # StandardScaler centers to zero mean and scales by the population std,
    # so the transformed training set is centered at ~0 with std ~1.
    means = X_train_scaled.mean(axis=0)
    stds = X_train_scaled.std(axis=0)
    np.testing.assert_allclose(means.to_numpy(), 0.0, atol=1e-8)
    # DataFrame.std uses sample std (ddof=1); StandardScaler uses population
    # std (ddof=0), so the transformed std is ~1.012 rather than exactly 1.0.
    np.testing.assert_allclose(stds.to_numpy(), 1.0, atol=2e-2)
    # transform_features applied to the same train set reproduces fit_transform.
    X_again = transform_features(split.X_train, imputer, scaler)
    np.testing.assert_allclose(X_again.to_numpy(), X_train_scaled.to_numpy(), atol=1e-8)


def test_transform_features_uses_train_statistics_only(sample_df):
    """Val/test must be transformed with statistics learned on train only."""
    split = split_train_val_test(sample_df)
    imputer, scaler, _ = fit_preprocessor(split.X_train)
    X_val_scaled = transform_features(split.X_val, imputer, scaler)
    # Validation data is NOT re-standardized to its own mean/std.
    assert not np.allclose(X_val_scaled.mean(axis=0).to_numpy(), 0.0, atol=1e-8)


def test_transform_features_imputes_missing_values():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    imputer, scaler, _ = fit_preprocessor(df)
    transformed = transform_features(df, imputer, scaler)
    assert not transformed.isna().any().any()
    # The median of column "a" ([1, 2, 4] ignoring NaN) is 2.0; imputation
    # fills the NaN with it before scaling.
    assert imputer.statistics_[0] == pytest.approx(2.0)


def test_transform_query_features_accepts_alias_keys(sample_df):
    split = split_train_val_test(sample_df)
    imputer, scaler, _ = fit_preprocessor(split.X_train)
    feature_columns = list(split.X_train.columns)
    # Provide single-letter / capitalized aliases that the loader is expected to map.
    query = {"n": 1.0, "P": 2.0, "k": 3.0}
    transformed = transform_query_features(query, imputer, scaler, feature_columns)
    assert list(transformed.columns) == feature_columns
    assert len(transformed) == 1


def test_transform_query_features_raises_on_missing_column(sample_df):
    split = split_train_val_test(sample_df)
    imputer, scaler, _ = fit_preprocessor(split.X_train)
    feature_columns = list(split.X_train.columns)
    with pytest.raises(ValueError, match="missing"):
        transform_query_features({"n": 1.0, "p": 2.0}, imputer, scaler, feature_columns)
