from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from config import (
    BOOTSTRAP_RESAMPLES,
    CONFIDENCE_LEVEL,
    LATENCY_BENCHMARK_RUNS,
    LATENCY_THRESHOLD_MS,
    LATENCY_WARMUP_RUNS,
    RANDOM_STATE,
    TOP_K,
)
from data.loader import load_crop_dataset
from data.preprocessing import fit_preprocessor, transform_features
from data.split import split_train_val_test
from evaluation.bootstrap import bootstrap_metric_ci, paired_bootstrap_difference
from evaluation.evaluate import evaluate_recommender
from evaluation.ground_truth import (
    build_weak_multilabel_ground_truth,
    fit_crop_suitability_profiles,
)
from evaluation.latency import benchmark_latency
from evaluation.tuning import benchmark_baseline_models, benchmark_hybrid_models
from models.base_knn import CropKNNRecommender
from models.kmeanSearch_knn import KMeansSearchKNNRecommender


def _best_per_metric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("metric", as_index=False, sort=False).first()


def _build_model(row: pd.Series, hybrid: bool):
    common = {
        "metric": row["metric"],
        "neighbors_per_crop": int(row["neighbors_per_crop"]),
    }
    if hybrid:
        return KMeansSearchKNNRecommender(
            **common,
            n_clusters=int(row["n_clusters"]),
            n_probe=int(row["n_probe"]),
        )
    return CropKNNRecommender(**common)


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
    X_val_scaled = transform_features(split_data.X_val, imputer, scaler)

    baseline_validation = benchmark_baseline_models(
        X_train_scaled, split_data.y_train, X_val_scaled, split_data.y_val
    )
    hybrid_validation = benchmark_hybrid_models(
        X_train_scaled, split_data.y_train, X_val_scaled, split_data.y_val
    )
    baseline_validation.to_csv(reports_dir / "baseline_validation.csv", index=False)
    hybrid_validation.to_csv(reports_dir / "hybrid_validation.csv", index=False)

    selected_baseline = _best_per_metric(baseline_validation)
    selected_hybrid = _best_per_metric(hybrid_validation)
    selected_baseline.to_csv(reports_dir / "selected_baseline_configs.csv", index=False)
    selected_hybrid.to_csv(reports_dir / "selected_hybrid_configs.csv", index=False)

    X_trainval = pd.concat([split_data.X_train, split_data.X_val], axis=0)
    y_trainval = pd.concat([split_data.y_train, split_data.y_val], axis=0)
    final_imputer, final_scaler, X_trainval_scaled = fit_preprocessor(X_trainval)
    X_test_scaled = transform_features(split_data.X_test, final_imputer, final_scaler)

    # Weak multi-label ground truth is learned only from train+validation features.
    profiles = fit_crop_suitability_profiles(X_trainval, y_trainval)
    weak_ground_truth = build_weak_multilabel_ground_truth(
        split_data.X_test,
        split_data.y_test,
        profiles,
        max_labels=3,
        min_features_in_range=5,
    )
    pd.DataFrame({
        "original_label": split_data.y_test.to_numpy(),
        "weak_relevant_crops": ["|".join(labels) for labels in weak_ground_truth],
        "relevant_count": [len(labels) for labels in weak_ground_truth],
    }).to_csv(reports_dir / "weak_multilabel_ground_truth.csv", index=False)

    model_details: dict[str, dict] = {}
    test_rows = []
    latency_rows = []

    for hybrid, selected in ((False, selected_baseline), (True, selected_hybrid)):
        for _, row in selected.iterrows():
            model_type = "KNN + K-Means++" if hybrid else "Exact KNN"
            model_name = f"{model_type} | {row['metric']}"
            model = _build_model(row, hybrid=hybrid).fit(X_trainval_scaled, y_trainval)

            single_metrics, single_per_query, recs, diagnostics = evaluate_recommender(
                model, X_test_scaled, split_data.y_test, top_k=TOP_K
            )
            multi_metrics, multi_per_query, _, _ = evaluate_recommender(
                model, X_test_scaled, weak_ground_truth, top_k=TOP_K
            )

            result = {
                "model": model_type,
                "metric": row["metric"],
                "ground_truth": "single_label",
                **single_metrics,
            }
            test_rows.append(result)
            test_rows.append({
                "model": model_type,
                "metric": row["metric"],
                "ground_truth": "weak_multilabel",
                **multi_metrics,
            })

            model_details[model_name] = {
                "model": model,
                "single_per_query": single_per_query,
                "multi_per_query": multi_per_query,
                "recommendations": recs,
                "diagnostics": diagnostics,
            }

            _, latency_summary = benchmark_latency(
                model,
                X_test_scaled.iloc[:100],
                top_k=TOP_K,
                threshold_ms=LATENCY_THRESHOLD_MS,
                warmup_runs=LATENCY_WARMUP_RUNS,
                measured_runs=LATENCY_BENCHMARK_RUNS,
            )
            latency_rows.append({"model": model_type, "metric": row["metric"], **latency_summary})

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(reports_dir / "test_metrics_single_and_multilabel.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(reports_dir / "latency_profile.csv", index=False)

    ci_rows = []
    paired_rows = []
    for ground_truth_key, per_query_key in (
        ("single_label", "single_per_query"),
        ("weak_multilabel", "multi_per_query"),
    ):
        for model_name, detail in model_details.items():
            for metric_column, metric_name in (("AP@K", "MAP@K"), ("NDCG@K", "NDCG@K")):
                ci = bootstrap_metric_ci(
                    detail[per_query_key][metric_column].to_numpy(),
                    confidence=CONFIDENCE_LEVEL,
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    random_state=RANDOM_STATE,
                )
                ci_rows.append({
                    "ground_truth": ground_truth_key,
                    "model": model_name,
                    "metric": metric_name,
                    **ci,
                })

        for distance_metric in ("euclidean", "manhattan", "cosine"):
            exact = model_details[f"Exact KNN | {distance_metric}"][per_query_key]
            hybrid = model_details[f"KNN + K-Means++ | {distance_metric}"][per_query_key]
            for metric_column, metric_name in (("AP@K", "MAP@K"), ("NDCG@K", "NDCG@K")):
                comparison = paired_bootstrap_difference(
                    exact[metric_column].to_numpy(),
                    hybrid[metric_column].to_numpy(),
                    confidence=CONFIDENCE_LEVEL,
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    random_state=RANDOM_STATE,
                )
                paired_rows.append({
                    "ground_truth": ground_truth_key,
                    "distance_metric": distance_metric,
                    "metric": metric_name,
                    "difference": "hybrid_minus_exact",
                    **comparison,
                })

    pd.DataFrame(ci_rows).to_csv(reports_dir / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(reports_dir / "paired_bootstrap_model_comparison.csv", index=False)

    # Persist the overall best weak-multilabel models for the application.
    weak_results = test_results[test_results["ground_truth"] == "weak_multilabel"]
    best_exact = weak_results[weak_results["model"] == "Exact KNN"].sort_values(
        ["MAP@K", "NDCG@K"], ascending=False
    ).iloc[0]
    best_hybrid = weak_results[weak_results["model"] == "KNN + K-Means++"].sort_values(
        ["MAP@K", "NDCG@K"], ascending=False
    ).iloc[0]
    baseline_model = model_details[f"Exact KNN | {best_exact['metric']}"]["model"]
    hybrid_model = model_details[f"KNN + K-Means++ | {best_hybrid['metric']}"]["model"]

    with open(models_dir / "final_imputer.pkl", "wb") as file_handle:
        pickle.dump(final_imputer, file_handle)
    with open(models_dir / "final_scaler.pkl", "wb") as file_handle:
        pickle.dump(final_scaler, file_handle)
    with open(models_dir / "baseline_best_model.pkl", "wb") as file_handle:
        pickle.dump(baseline_model, file_handle)
    with open(models_dir / "hybrid_best_model.pkl", "wb") as file_handle:
        pickle.dump(hybrid_model, file_handle)

    artifact_payload = {
        "feature_columns": list(split_data.X_train.columns),
        "top_k": TOP_K,
        "random_state": RANDOM_STATE,
        "latency_threshold_ms": LATENCY_THRESHOLD_MS,
        "ground_truth_note": (
            "Weak multi-label labels retain the observed crop and add up to two alternatives "
            "using train-only crop-specific 10--90% feature envelopes. They are pseudo labels, "
            "not expert-validated agronomic annotations."
        ),
        "best_baseline_metric": best_exact["metric"],
        "best_hybrid_metric": best_hybrid["metric"],
    }
    with open(models_dir / "artifact_metadata.json", "w", encoding="utf-8") as file_handle:
        json.dump(artifact_payload, file_handle, indent=2)

    print("Experiment complete. Generated tables in reports/tables/.")


if __name__ == "__main__":
    main()
