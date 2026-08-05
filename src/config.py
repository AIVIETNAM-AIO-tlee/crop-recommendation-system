from __future__ import annotations

RANDOM_STATE = 42
TOP_K = 5
DISTANCE_METRICS = ["euclidean", "manhattan", "cosine"]
TARGET_COLUMN = "label"

# Service-level objective used by the reproducible latency benchmark.
# Override in experiments when the deployment target has a different budget.
LATENCY_THRESHOLD_MS = 1.0
LATENCY_WARMUP_RUNS = 3
LATENCY_BENCHMARK_RUNS = 10
BOOTSTRAP_RESAMPLES = 2000
CONFIDENCE_LEVEL = 0.95
