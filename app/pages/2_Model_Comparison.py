from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick_best_validation_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None

    sort_columns = [col for col in ["MAP@K", "NDCG@K", "Candidate_ratio", "Latency_ms_per_query"] if col in df.columns]
    if sort_columns:
        ascending = [False, False, True, True][: len(sort_columns)]
        return df.sort_values(sort_columns, ascending=ascending).iloc[0]
    return df.iloc[0]


def render_best_validation_block(title: str, df: pd.DataFrame) -> None:
    best_row = pick_best_validation_row(df)
    if best_row is None:
        return

    st.markdown(f"**Best model after {title}:**")
    summary = {k: best_row[k] for k in best_row.index if k not in {"model"} and pd.notna(best_row[k])}
    st.write(f"- Model: {best_row.get('model', 'N/A')}")
    for key in ["metric", "neighbors_per_crop", "n_clusters", "n_probe", "MAP@K", "NDCG@K", "HitRate@K", "Latency_ms_per_query", "Candidate_ratio"]:
        if key in best_row.index and pd.notna(best_row[key]):
            st.write(f"- {key}: {best_row[key]}")


st.title("Model Comparison")
st.caption("Validation and test results exported by the training script.")

reports_dir = ROOT / "reports" / "tables"
frames = {
    "KNN Baseline Validation": read_csv_if_exists(reports_dir / "baseline_validation.csv"),
    "KNN Search optimizer Validation": read_csv_if_exists(reports_dir / "hybrid_validation.csv"),
    "KNN Baseline Test": read_csv_if_exists(reports_dir / "baseline_test_metrics.csv"),
    "KNN Search optimizer Test": read_csv_if_exists(reports_dir / "hybrid_test_metrics.csv"),
}

available = {name: df for name, df in frames.items() if not df.empty}
if not available:
    st.warning("No reports found yet. Run `python main.py` from `src/` to generate validation/test CSVs.")
    st.stop()

for name, df in available.items():
    st.subheader(name)
    st.dataframe(df, use_container_width=True)
    if "Validation" in name:
        render_best_validation_block(name, df)

st.markdown("### Snapshot")
summary_rows = []
for name, df in available.items():
    row = {"report": name}
    for col in ["MAP@K", "NDCG@K", "HitRate@K", "Latency_ms_per_query", "Candidate_ratio"]:
        if col in df.columns:
            row[col] = float(df.iloc[0][col]) if len(df) else None
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
if not summary.empty:
    st.dataframe(summary, use_container_width=True)
    metric_cols = [c for c in ["MAP@K", "NDCG@K", "Latency_ms_per_query", "Candidate_ratio"] if c in summary.columns]
    # if metric_cols:
    #     st.bar_chart(summary.set_index("report")[metric_cols])
