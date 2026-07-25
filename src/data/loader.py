from __future__ import annotations
from pathlib import Path
import pandas as pd
from config import TARGET_COLUMN


def load_crop_dataset(csv_path: str | Path) -> pd.DataFrame:
	"""Load the crop dataset and normalize column names."""
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found: {path}")

	df = pd.read_csv(path)
	df.columns = [str(column).strip().lower() for column in df.columns]
	empty_columns = df.columns[df.isna().all()].tolist()
	if empty_columns:
		df = df.drop(columns=empty_columns)
	if TARGET_COLUMN not in df.columns:
		raise ValueError(f"Expected target column '{TARGET_COLUMN}', got {df.columns.tolist()}")
	return df


def infer_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
	return [column for column in df.columns if column != target_column]
