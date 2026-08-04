from pathlib import Path
from data.loader import load_crop_dataset
from data.split import split_train_val_test
from data.preprocessing import fit_preprocessor, transform_features
from models.base_knn import CropKNNRecommender
from models.kmeanSearch_knn import KMeansSearchKNNRecommender
from evaluation.evaluate import evaluate_recommender


def test_end_to_end_baseline_and_hybrid():
    root = Path(__file__).resolve().parents[1]
    df = load_crop_dataset(root / "data/raw/Crop_recommendation.csv")
    split = split_train_val_test(df)
    _, scaler, X_train = fit_preprocessor(split.X_train)
    # Refit helper returns imputer too; use explicit second call to keep test readable.
    imputer, scaler, X_train = fit_preprocessor(split.X_train)
    X_val = transform_features(split.X_val, imputer, scaler)

    models = [
        CropKNNRecommender(metric="euclidean", neighbors_per_crop=3).fit(X_train, split.y_train),
        KMeansSearchKNNRecommender(metric="euclidean", neighbors_per_crop=3,
                                   n_clusters=8, n_probe=1).fit(X_train, split.y_train),
    ]
    for model in models:
        metrics, _, recs, _ = evaluate_recommender(model, X_val.iloc[:20], split.y_val.iloc[:20])
        assert recs.shape == (20, 5)
        assert 0 <= metrics["MAP@K"] <= 1
        assert 0 <= metrics["NDCG@K"] <= 1
