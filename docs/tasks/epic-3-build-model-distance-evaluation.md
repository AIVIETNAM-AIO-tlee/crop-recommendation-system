# Epic 3: Model Development & Evaluation

## Objective

Develop and evaluate retrieval-based crop recommendation models by implementing an exact KNN baseline and a K-Means++ accelerated KNN variant. The objective is to compare recommendation quality, retrieval efficiency, and the impact of different distance metrics under a unified evaluation framework.

---

## Description

This epic focuses on implementing the core recommendation algorithms and conducting experimental evaluations. Two retrieval-based methods are developed: an exact class-wise KNN baseline and a K-Means++ accelerated KNN model that reduces the search space through cluster-based candidate selection.

The models are evaluated using multiple distance metrics and ranking-oriented evaluation metrics to analyze both recommendation quality and inference efficiency.

---

## Tasks

### Baseline Model

- [x] Implement the exact class-wise KNN recommender.
- [x] Support Top-K recommendation.
- [x] Implement Euclidean distance.
- [x] Implement Manhattan distance.
- [x] Implement Cosine similarity.

### Search Optimization

- [x] Train the K-Means++ clustering model.
- [x] Build the centroid index.
- [x] Implement cluster-based candidate selection.
- [x] Apply class-wise KNN within selected clusters.
- [x] Tune the number of clusters (`n_clusters`).
- [x] Tune the number of probed clusters (`n_probe`).

### Hyperparameter Tuning

- [x] Tune the number of neighbors (`k`).
- [x] Select the best distance metric.
- [x] Compare different search configurations.
- [x] Retrain the final model using the best configuration.

### Evaluation

- [x] Compute Precision@K.
- [x] Compute Recall@K.
- [x] Compute MAP@K.
- [x] Compute NDCG@K.
- [x] Compute HitRate@K.
- [x] Measure inference latency.
- [x] Measure candidate reduction ratio.
- [x] Compare Exact KNN and K-Means++ KNN.

### Result Analysis

- [x] Analyze recommendation quality.
- [x] Analyze retrieval efficiency.
- [x] Compare the impact of different distance metrics.
- [x] Summarize experimental findings.

### Quality Assurance

- [x] Verify model correctness.
- [x] Review experimental results.
- [x] Validate reproducibility of experiments.

---

## Deliverables

- Exact KNN implementation
- K-Means++ accelerated KNN implementation
- Hyperparameter tuning results
- Experimental results
- Performance comparison
- Evaluation report

---

## Acceptance Criteria

- Both retrieval models are fully implemented.
- Hyperparameters have been tuned using the validation set.
- Models are evaluated on the held-out test set.
- Recommendation quality is reported using ranking metrics.
- Retrieval efficiency is analyzed using latency and candidate reduction.
- Experimental results are reproducible and documented.

---

## Status

**Completed**