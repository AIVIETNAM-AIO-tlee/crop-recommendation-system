from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def fit_preprocessor(X_train: pd.DataFrame):
	imputer = SimpleImputer(strategy="median")
	scaler = StandardScaler()

	X_train_imputed = imputer.fit_transform(X_train)
	X_train_scaled = scaler.fit_transform(X_train_imputed)

	return imputer, scaler, pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)


def transform_features(
	X: pd.DataFrame,
	imputer: SimpleImputer,
	scaler: StandardScaler,
) -> pd.DataFrame:
	transformed = scaler.transform(imputer.transform(X))
	return pd.DataFrame(transformed, columns=X.columns, index=X.index)


def transform_query_features(
	raw_features,
	imputer: SimpleImputer,
	scaler: StandardScaler,
	feature_columns: list[str],
) -> pd.DataFrame:
	if isinstance(raw_features, dict):
		query_df = pd.DataFrame([raw_features], columns=feature_columns)
	else:
		query_df = pd.DataFrame(raw_features).copy()
		query_df = query_df[feature_columns]

	query_scaled = scaler.transform(imputer.transform(query_df))
	return pd.DataFrame(query_scaled, columns=feature_columns, index=query_df.index)


def recommend_from_raw_features(
	raw_features,
	model,
	imputer: SimpleImputer,
	scaler: StandardScaler,
	feature_columns: list[str],
	top_k: int = 5,
):
	query_scaled = transform_query_features(
		raw_features=raw_features,
		imputer=imputer,
		scaler=scaler,
		feature_columns=feature_columns,
	)
	return model.recommend(query_scaled, top_k=top_k, return_diagnostics=True)
