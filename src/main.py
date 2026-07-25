from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from config import TOP_K
from data.loader import load_crop_dataset
from data.preprocessing import fit_preprocessor
from data.split import split_train_val_test
from evaluation.evaluate import evaluate_recommender
from evaluation.tuning import benchmark_baseline_models, benchmark_hybrid_models
from models.base_knn import CropKNNRecommender
from models.kmeanSearch_knn import KMeansSearchKNNRecommender


def main() -> None:
	root = Path(__file__).resolve().parents[1]
	data_path = root / "data" / "raw" / "Crop_recommendation.csv"
	models_dir = root / "models"
	reports_dir = root / "reports" / "tables"
	models_dir.mkdir(parents=True, exist_ok=True)
	reports_dir.mkdir(parents=True, exist_ok=True)

	df = load_crop_dataset(data_path)
	split_data = split_train_val_test(df)
	imputer, scaler, X_train_scaled = fit_preprocessor(split_data.X_train)
	X_val_scaled = pd.DataFrame(
		scaler.transform(imputer.transform(split_data.X_val)),
		columns=split_data.X_val.columns,
		index=split_data.X_val.index,
	)
	X_test_scaled = pd.DataFrame(
		scaler.transform(imputer.transform(split_data.X_test)),
		columns=split_data.X_test.columns,
		index=split_data.X_test.index,
	)

	baseline_validation = benchmark_baseline_models(
		X_train_scaled, split_data.y_train, X_val_scaled, split_data.y_val
	)
	hybrid_validation = benchmark_hybrid_models(
		X_train_scaled, split_data.y_train, X_val_scaled, split_data.y_val
	)

	baseline_validation.to_csv(reports_dir / "baseline_validation.csv", index=False)
	hybrid_validation.to_csv(reports_dir / "hybrid_validation.csv", index=False)

	best_baseline_row = baseline_validation.iloc[0]
	best_hybrid_row = hybrid_validation.iloc[0]

	X_trainval = pd.concat([split_data.X_train, split_data.X_val], axis=0)
	y_trainval = pd.concat([split_data.y_train, split_data.y_val], axis=0)
	final_imputer, final_scaler, X_trainval_scaled = fit_preprocessor(X_trainval)
	X_test_final_scaled = pd.DataFrame(
		final_scaler.transform(final_imputer.transform(split_data.X_test)),
		columns=split_data.X_test.columns,
		index=split_data.X_test.index,
	)

	baseline_model = CropKNNRecommender(
		metric=best_baseline_row["metric"],
		neighbors_per_crop=int(best_baseline_row["neighbors_per_crop"]),
	).fit(X_trainval_scaled, y_trainval)
	hybrid_model = KMeansSearchKNNRecommender(
		metric=best_hybrid_row["metric"],
		neighbors_per_crop=int(best_hybrid_row["neighbors_per_crop"]),
		n_clusters=int(best_hybrid_row["n_clusters"]),
		n_probe=int(best_hybrid_row["n_probe"]),
	).fit(X_trainval_scaled, y_trainval)

	baseline_test_metrics, _, _, _ = evaluate_recommender(
		baseline_model, X_test_final_scaled, split_data.y_test, top_k=TOP_K
	)
	hybrid_test_metrics, _, _, _ = evaluate_recommender(
		hybrid_model, X_test_final_scaled, split_data.y_test, top_k=TOP_K
	)

	pd.DataFrame([baseline_test_metrics]).to_csv(reports_dir / "baseline_test_metrics.csv", index=False)
	pd.DataFrame([hybrid_test_metrics]).to_csv(reports_dir / "hybrid_test_metrics.csv", index=False)

	artifact_payload = {
		"feature_columns": list(split_data.X_train.columns),
		"top_k": TOP_K,
		"best_baseline": {
			"metric": best_baseline_row["metric"],
			"neighbors_per_crop": int(best_baseline_row["neighbors_per_crop"]),
		},
		"best_hybrid": {
			"metric": best_hybrid_row["metric"],
			"neighbors_per_crop": int(best_hybrid_row["neighbors_per_crop"]),
			"n_clusters": int(best_hybrid_row["n_clusters"]),
			"n_probe": int(best_hybrid_row["n_probe"]),
		},
	}

	with open(models_dir / "final_imputer.pkl", "wb") as file_handle:
		pickle.dump(final_imputer, file_handle)

	with open(models_dir / "final_scaler.pkl", "wb") as file_handle:
		pickle.dump(final_scaler, file_handle)

	with open(models_dir / "baseline_best_model.pkl", "wb") as file_handle:
		pickle.dump(baseline_model, file_handle)

	with open(models_dir / "hybrid_best_model.pkl", "wb") as file_handle:
		pickle.dump(hybrid_model, file_handle)

	with open(models_dir / "artifact_metadata.json", "w", encoding="utf-8") as file_handle:
		json.dump(artifact_payload, file_handle, indent=2)


if __name__ == "__main__":
	main()
