# Experiment Update: Latency, Confidence Intervals, Multi-label Ranking, and Tests

## Reproducible evaluation protocol

The dataset is split with `random_state=42` into 70% training, 10% validation, and 20% test partitions. Hyperparameters are selected on validation data. The final preprocessing objects and recommenders are refitted on train+validation, while the test partition is used only for final reporting.

## Latency threshold and component profiling

Latency is measured on a fixed workload of 100 test queries after three warm-up runs, followed by ten measured runs. The service-level threshold is configurable in `src/config.py`; the current target is **p95 <= 1.0 ms/query**. Reporting p95 rather than only the mean makes the conclusion less sensitive to unusually fast runs.

The exact KNN profiler separates:

- pairwise distance computation;
- class-wise crop scoring and ranking.

The KNN + K-Means++ profiler separates:

- query-to-centroid distance computation;
- cluster selection and candidate collection;
- query-to-candidate distance computation;
- class-wise crop scoring and ranking.

All six selected configurations pass the current 1.0 ms/query p95 threshold. Exact KNN is faster on this small dataset. In the hybrid model, candidate-distance computation is the largest component, followed by class-wise ranking and candidate/cluster selection. Therefore, the current bottleneck is not centroid-distance computation. Vectorizing candidate scoring or batching queries would be more useful than optimizing centroid calculation alone.

See `reports/tables/latency_profile.csv` for the measured p50, p95, threshold result, and component breakdown.

## Bootstrap confidence intervals

MAP@5 and NDCG@5 are bootstrapped over test queries with 2,000 resamples and a 95% percentile confidence interval. A paired bootstrap is also performed on the per-query difference:

`KNN + K-Means++ - Exact KNN`.

A difference interval entirely above zero supports the hybrid model; an interval entirely below zero supports exact KNN; an interval containing zero indicates that the observed difference is not clear at the selected confidence level.

Under the weak multi-label evaluation, Exact KNN with Manhattan distance has the highest point estimates: MAP@5 = 0.9201 (95% CI 0.9078--0.9317) and NDCG@5 = 0.9554 (95% CI 0.9472--0.9628). The paired intervals show no clear difference between exact and hybrid search for Euclidean or Manhattan. For cosine MAP@5, the paired interval is below zero, so Exact KNN is better for that comparison. Consequently, the overall recommendation is Exact KNN with Manhattan distance when ranking quality is the main objective. The Manhattan hybrid remains useful when reducing the candidate set is more important, because it examines about 21.2% of train+validation samples while its paired MAP@5 and NDCG@5 intervals still include zero.

See `bootstrap_confidence_intervals.csv` and `paired_bootstrap_model_comparison.csv`.

## Weak multi-label ground truth

The original dataset provides one crop label per observation even though several crops may be agronomically plausible under similar soil and climate conditions. The updated evaluation therefore adds a reproducible weak multi-label target containing two or three crops:

1. The observed crop label is always retained.
2. Crop-specific suitability profiles are learned only from train+validation data.
3. For every crop and feature, the profile stores the median and the 10th--90th percentile interval for nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.
4. An alternative crop must place at least five of seven query features inside its crop-specific interval.
5. Eligible alternatives are ranked by standardized distance to the crop median, and at most two are added.
6. When no alternative passes the five-feature rule, the nearest crop profile is used to ensure the evaluation contains at least two relevant labels.

These additional labels are **weak/pseudo labels**, not expert-validated agronomic annotations. They reduce the known limitation of treating every non-original crop as definitely wrong, but they must not be presented as a replacement for expert annotation. The generated labels are stored in `weak_multilabel_ground_truth.csv` for auditability.

## Automated tests

The repository now includes unit and integration tests executed with `pytest`:

- known-value Euclidean, Manhattan, and cosine distances;
- zero distance for identical vectors;
- nearest crop appears first;
- Top-K crop labels are unique;
- K larger than the number of available crop labels is safely clipped;
- hybrid K-Means++ recommendation returns valid diagnostics;
- correct multi-label Precision@K, Recall@K, MAP@K, and NDCG@K;
- bootstrap CI and paired-bootstrap conclusions;
- weak multi-label generation retains the original label and adds alternatives;
- end-to-end data loading, preprocessing, exact KNN, hybrid KNN, and evaluation.

Current result: **12 tests passed**.

Run:

```bash
python -m pip install -r requirements.txt
pytest
PYTHONPATH=src python src/main.py
```
