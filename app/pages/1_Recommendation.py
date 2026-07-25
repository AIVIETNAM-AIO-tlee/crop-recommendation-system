from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from data.loader import load_crop_dataset
from data.preprocessing import recommend_from_raw_features

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_artifacts():
    models_dir = ROOT / "models"
    metadata = json.loads((models_dir / "artifact_metadata.json").read_text(encoding="utf-8"))
    with open(models_dir / "final_imputer.pkl", "rb") as fh:
        imputer = pickle.load(fh)
    with open(models_dir / "final_scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(models_dir / "baseline_best_model.pkl", "rb") as fh:
        baseline_model = pickle.load(fh)
    with open(models_dir / "hybrid_best_model.pkl", "rb") as fh:
        hybrid_model = pickle.load(fh)
    return metadata, imputer, scaler, baseline_model, hybrid_model


st.title("Crop Recommendation")
st.caption("Enter soil and weather values to get ranked crop suggestions.")

models_dir = ROOT / "models"
if not (models_dir / "artifact_metadata.json").exists():
    st.error("Missing model artifacts. Run `python main.py` from `src/` first.")
    st.stop()

metadata, imputer, scaler, baseline_model, hybrid_model = load_artifacts()
feature_columns = metadata.get("feature_columns", ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"])

left, right = st.columns([1.1, 0.9])
with left:
    model_choice = st.radio(
        "Choose model",
        ["Hybrid KMeans + KNN", "Baseline KNN"],
        horizontal=True,
    )
    top_k = st.slider("Top-K recommendations", 1, 10, int(metadata.get("top_k", 5)))

    defaults = {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "temperature": 20.5,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        nitrogen = st.number_input("Nitrogen", value=defaults["nitrogen"], step=1.0)
        potassium = st.number_input("Potassium", value=defaults["potassium"], step=1.0)
    with c2:
        phosphorus = st.number_input("Phosphorus", value=defaults["phosphorus"], step=1.0)
        temperature = st.number_input("Temperature", value=defaults["temperature"], step=0.1)
    with c3:
        humidity = st.number_input("Humidity", value=defaults["humidity"], step=1.0)
        ph = st.number_input("pH", value=defaults["ph"], step=0.1)
        rainfall = st.number_input("Rainfall", value=defaults["rainfall"], step=1.0)

    query = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    if st.button("Generate recommendations", type="primary"):
        model = hybrid_model if model_choice.startswith("Hybrid") else baseline_model
        recommendations, diagnostics = recommend_from_raw_features(
            raw_features=query,
            model=model,
            imputer=imputer,
            scaler=scaler,
            feature_columns=feature_columns,
            top_k=top_k,
        )
        recs = list(recommendations[0]) if len(recommendations) else []
        diag = diagnostics["candidate_count"][0]
        st.session_state["recommendation_result"] = {
            "recommendations": recs,
            "diagnostics": {
                "candidate_count": int(diag),
                "candidate_ratio": float(diagnostics["candidate_ratio"][0]),
                "clusters_scanned": float(diagnostics["clusters_scanned"][0]) if str(diagnostics["clusters_scanned"][0]) != "nan" else None,
            },
            "query": query,
            "model_choice": model_choice,
        }

with right:
    st.markdown("#### Input Summary")
    st.dataframe(pd.DataFrame([query]), use_container_width=True)
    st.markdown("#### Notes")
    st.info("The app uses the saved final imputer and scaler, so user inputs are transformed exactly like during training.")

result = st.session_state.get("recommendation_result")
if result:
    st.markdown("### Recommended Crops")
    for idx, crop in enumerate(result["recommendations"], start=1):
        st.markdown(
            f"<div style='padding:0.9rem 1rem;margin-bottom:0.6rem;border-radius:0.9rem;background:#0f172a;color:white;'><strong>#{idx}</strong> {crop}</div>",
            unsafe_allow_html=True,
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Candidates Scanned", result["diagnostics"]["candidate_count"])
    with col_b:
        st.metric("Candidate Ratio", f"{result['diagnostics']['candidate_ratio']:.3f}")
    with col_c:
        clusters = result["diagnostics"]["clusters_scanned"]
        st.metric("Clusters Scanned", "-" if clusters is None else f"{clusters:.0f}")

    st.markdown("### Model Pick")
    st.write(f"Selected model: **{result['model_choice']}**")
