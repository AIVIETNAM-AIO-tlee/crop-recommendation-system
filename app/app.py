from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_metadata() -> dict:
    metadata_path = ROOT / "models" / "artifact_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}

metadata = load_metadata()
feature_count = len(metadata.get("feature_columns", [])) or 7

analysis_page = st.Page("pages/1_Recommendation.py", title="Crop Recommendation", icon=":material/analytics:",)
model_page = st.Page("pages/2_Model_Comparison.py", title="Model Comparison", icon=":material/compare:",)
dataset_page = st.Page("pages/3_Dataset.py", title="Dataset", icon=":material/dataset:",)

pg = st.navigation({
    "Analysis": [analysis_page],
    "Model Comparison": [model_page],
    "Dataset": [dataset_page]
}, position="sidebar",)

pg.run()

st.caption("Tip: launch with `streamlit run app/app.py` from the project root.")