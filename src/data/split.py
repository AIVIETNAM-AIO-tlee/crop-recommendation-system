from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TARGET_COLUMN
from data.loader import infer_feature_columns


@dataclass(frozen=True)
class SplitData:
	X_train: pd.DataFrame
	X_val: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_val: pd.Series
	y_test: pd.Series


def split_train_val_test(
	df: pd.DataFrame,
	target_column: str = TARGET_COLUMN,
	test_size: float = 0.2,
	val_size: float = 0.1,
	random_state: int = RANDOM_STATE,
) -> SplitData:
	feature_columns = infer_feature_columns(df, target_column=target_column)
	X = df[feature_columns].copy()
	y = df[target_column].copy()

	X_temp, X_test, y_temp, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)
	relative_val_size = val_size / (1.0 - test_size)
	X_train, X_val, y_train, y_val = train_test_split(
		X_temp,
		y_temp,
		test_size=relative_val_size,
		random_state=random_state,
		stratify=y_temp,
	)

	return SplitData(
		X_train=X_train,
		X_val=X_val,
		X_test=X_test,
		y_train=y_train,
		y_val=y_val,
		y_test=y_test,
	)
