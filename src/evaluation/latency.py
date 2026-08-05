from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd


def benchmark_latency(
    model,
    X_query,
    top_k: int = 5,
    threshold_ms: float = 1.0,
    warmup_runs: int = 5,
    measured_runs: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Benchmark fixed-workload latency and report p50/p95 against a threshold."""
    if len(X_query) == 0:
        raise ValueError("X_query must contain at least one query")
    for _ in range(warmup_runs):
        model.recommend(X_query, top_k=top_k, return_diagnostics=True)

    run_rows = []
    component_rows = []
    for run in range(measured_runs):
        start = perf_counter()
        _, diagnostics = model.recommend(X_query, top_k=top_k, return_diagnostics=True)
        total_ms = (perf_counter() - start) * 1000 / len(X_query)
        run_rows.append({"run": run + 1, "total_ms_per_query": total_ms})
        timings = diagnostics.get("timing_ms", {})
        component_rows.append({key: float(np.mean(value)) for key, value in timings.items()})

    run_df = pd.DataFrame(run_rows)
    components = pd.DataFrame(component_rows).mean().to_dict() if component_rows else {}
    summary = {
        "workload_queries": len(X_query),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "threshold_ms": threshold_ms,
        "mean_ms": float(run_df["total_ms_per_query"].mean()),
        "p50_ms": float(run_df["total_ms_per_query"].quantile(0.50)),
        "p95_ms": float(run_df["total_ms_per_query"].quantile(0.95)),
        "passes_threshold": bool(run_df["total_ms_per_query"].quantile(0.95) <= threshold_ms),
        **{f"component_{key}": value for key, value in components.items()},
    }
    return run_df, summary
