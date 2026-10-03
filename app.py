"""
Streamlit web app for the Phishing URL Detection System.
Lets a user enter a URL, pick a model, see the prediction,
and view the full model comparison report.
"""

import os
import pandas as pd
import streamlit as st

from src.predict import predict_url, list_available_models

METRICS_PATH = "results/metrics.csv"
PLOTS_DIR = "results/plots"

st.set_page_config(page_title="Phishing URL Detector", page_icon="🎣", layout="centered")

st.title("🎣 Phishing URL Detection System")
st.caption("A Comparative Study of Machine-Learning Techniques for Phishing URL Detection")

# ---------- Prediction section ----------
st.header("Check a URL")

available_models = list_available_models()

if not available_models:
    st.warning("No trained models found. Run `python -m src.train` first.")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input("Enter a URL to check", placeholder="https://example.com")
    with col2:
        model_choice = st.selectbox("Model", available_models,
                                     index=available_models.index("RandomForest")
                                     if "RandomForest" in available_models else 0)

    if st.button("Predict", type="primary"):
        if not url_input.strip():
            st.error("Please enter a URL.")
        else:
            try:
                label, confidence = predict_url(url_input, model_choice)
                if label == "Phishing":
                    st.error(f"⚠️ **{label}** — confidence: {confidence:.2%}")
                else:
                    st.success(f"✅ **{label}** — confidence: {confidence:.2%}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()

# ---------- Comparison section ----------
st.header("Model Comparison")

if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=metrics_df.columns[1:], color="lightgreen"))

    best_model = metrics_df.iloc[0]["Model"]
    st.caption(f"Best performing model by F1-score: **{best_model}**")
else:
    st.info("No metrics found yet. Run `python -m src.evaluate` first.")

# ---------- Plots section ----------
st.header("Visualizations")

roc_path = os.path.join(PLOTS_DIR, "roc_comparison.png")
if os.path.exists(roc_path):
    st.subheader("ROC Curve Comparison")
    st.image(roc_path)

if os.path.exists(PLOTS_DIR):
    cm_files = [f for f in os.listdir(PLOTS_DIR) if f.startswith("confusion_")]
    if cm_files:
        st.subheader("Confusion Matrices")
        selected_cm = st.selectbox("Select a model's confusion matrix", cm_files)
        st.image(os.path.join(PLOTS_DIR, selected_cm))