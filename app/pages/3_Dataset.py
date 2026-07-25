from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.loader import load_crop_dataset

st.title("Dataset Overview")
st.caption("Inspect the raw crop recommendation dataset used by the app.")

csv_path = ROOT / "data" / "raw" / "Crop_recommendation.csv"
if not csv_path.exists():
    st.error("Dataset not found in data/raw/Crop_recommendation.csv")
    st.stop()

df = load_crop_dataset(csv_path)
feature_columns = [column for column in df.columns if column != "label"]

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", len(feature_columns))
c3.metric("Classes", df["label"].nunique())

st.subheader("Sample Rows")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Class Distribution")
class_counts = df["label"].value_counts().sort_values(ascending=False)
st.bar_chart(class_counts)

st.subheader("Feature Summary")
st.dataframe(df[feature_columns].describe().T, use_container_width=True)
