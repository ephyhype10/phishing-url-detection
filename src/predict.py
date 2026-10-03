"""
Single-URL prediction module.
Loads a trained model and classifies one URL as phishing or legitimate.
Used by both the CLI and the Streamlit app.
"""

import os
import joblib
import pandas as pd

from src.feature_extractor import extract_features

MODELS_DIR = "results/models"


def list_available_models(models_dir: str = MODELS_DIR) -> list:
    """Returns the names of all trained models saved on disk."""
    if not os.path.isdir(models_dir):
        return []
    return [
        f.replace(".joblib", "")
        for f in os.listdir(models_dir)
        if f.endswith(".joblib")
    ]


def load_model(model_name: str, models_dir: str = MODELS_DIR):
    """Loads a single trained model by name. Raises a clear error if missing."""
    path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(path):
        available = list_available_models(models_dir)
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {path}. "
            f"Available models: {available}"
        )
    return joblib.load(path)


def predict_url(url: str, model_name: str = "RandomForest", models_dir: str = MODELS_DIR):
    """
    Predicts whether a single URL is phishing or legitimate.
    Returns (label: str, confidence: float).
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")

    model = load_model(model_name, models_dir)
    features = extract_features(url)
    features_df = pd.DataFrame([features])

    prediction = model.predict(features_df)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(features_df)[0][int(prediction)]
    else:
        confidence = 1.0  # model has no probability estimate

    label = "Phishing" if prediction == 1 else "Legitimate"
    return label, float(confidence)


if __name__ == "__main__":
    print("Available models:", list_available_models())
    model_choice = input("Enter model name to use (e.g. RandomForest): ").strip() or "RandomForest"
    url = input("Enter a URL to check: ").strip()

    try:
        label, confidence = predict_url(url, model_choice)
        print(f"\nPrediction: {label}  (confidence: {confidence:.2%})")
    except Exception as e:
        print(f"\nError: {e}")