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
		query_df = pd.DataFrame([raw_features])
	else:
		query_df = pd.DataFrame(raw_features).copy()

	# Determine the expected input column names from the fitted preprocessor.
	if hasattr(imputer, "feature_names_in_"):
		expected_columns = list(imputer.feature_names_in_)
	elif hasattr(scaler, "feature_names_in_"):
		expected_columns = list(scaler.feature_names_in_)
	else:
		expected_columns = feature_columns

	alias_map = {
		"n": "nitrogen",
		"p": "phosphorus",
		"k": "potassium",
		"nitrogen": "N",
		"phosphorus": "P",
		"potassium": "K",
		"N": "nitrogen",
		"P": "phosphorus",
		"K": "potassium",
	}

	expected_lower = {col.lower(): col for col in expected_columns}
	query_column_map = {}

	for column in query_df.columns:
		if column in expected_columns:
			query_column_map[column] = column
		elif column.lower() in expected_lower:
			query_column_map[column] = expected_lower[column.lower()]
		elif column in alias_map and alias_map[column] in expected_columns:
			query_column_map[column] = alias_map[column]
		elif column.lower() in alias_map and alias_map[column.lower()] in expected_columns:
			query_column_map[column] = alias_map[column.lower()]
		else:
			# Preserve the original column until selection so we can raise a clear error later.
			query_column_map[column] = column

	query_df = query_df.rename(columns=query_column_map)

	missing = [col for col in expected_columns if col not in query_df.columns]
	if missing:
		raise ValueError(
			f"Feature columns missing from query input after normalization: {missing}. "
			f"Available columns: {list(query_df.columns)}"
		)

	query_df = query_df[expected_columns]
	query_scaled = scaler.transform(imputer.transform(query_df))
	return pd.DataFrame(query_scaled, columns=expected_columns, index=query_df.index)


def recommend_from_raw_features(
	raw_features,
	model,
	imputer: SimpleImputer,
	scaler: StandardScaler,
	feature_columns: list[str] | None = None,
	top_k: int = 5,
):
	if feature_columns is None:
		raise ValueError("feature_columns must be provided when calling recommend_from_raw_features")
	query_scaled = transform_query_features(
		raw_features=raw_features,
		imputer=imputer,
		scaler=scaler,
		feature_columns=feature_columns,
	)
	return model.recommend(query_scaled, top_k=top_k, return_diagnostics=True)
