"""
Integration tests for the full pipeline: raw data -> features -> train -> predict.
These check that the pieces work correctly TOGETHER, not in isolation.
Run with: pytest tests/test_integration.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_extractor import extract_features
from src.data_loader import load_raw_data, clean_data, build_feature_matrix, split_data


# ---------- Small synthetic dataset shared across tests ----------

SAMPLE_URLS = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "http://192.168.1.1/login",
    "http://paypal-verify-account.com",
    "https://github.com",
    "http://secure-bank-update.net/login",
]
SAMPLE_LABELS = [0, 0, 1, 1, 0, 1]


def _make_sample_df():
    return pd.DataFrame({"url": SAMPLE_URLS, "label": SAMPLE_LABELS})


# ---------- Data loading + cleaning integration ----------

def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "url": ["http://a.com", "http://a.com", "http://b.com"],
        "label": [0, 0, 1],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2


def test_clean_data_drops_missing_values():
    df = pd.DataFrame({
        "url": ["http://a.com", None, "http://b.com"],
        "label": [0, 1, None],
    })
    cleaned = clean_data(df)
    assert cleaned["url"].isnull().sum() == 0
    assert cleaned["label"].isnull().sum() == 0


def test_clean_data_keeps_only_binary_labels():
    df = pd.DataFrame({
        "url": ["http://a.com", "http://b.com", "http://c.com"],
        "label": [0, 1, 5],  # 5 is invalid
    })
    cleaned = clean_data(df)
    assert set(cleaned["label"].unique()).issubset({0, 1})


# ---------- Feature matrix integration ----------

def test_build_feature_matrix_matches_row_count():
    df = _make_sample_df()
    features_df = build_feature_matrix(df)
    assert len(features_df) == len(df)


def test_build_feature_matrix_has_no_missing_values():
    df = _make_sample_df()
    features_df = build_feature_matrix(df)
    assert not features_df.isnull().values.any()


def test_build_feature_matrix_label_column_matches_input():
    df = _make_sample_df()
    features_df = build_feature_matrix(df)
    assert list(features_df["label"]) == SAMPLE_LABELS


# ---------- Train/test split integration ----------

def test_split_data_produces_correct_shapes():
    df = _make_sample_df()
    features_df = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(features_df, test_size=0.33, random_state=42)

    assert len(X_train) + len(X_test) == len(features_df)
    assert len(y_train) + len(y_test) == len(features_df)
    assert "label" not in X_train.columns  # label must not leak into features


# ---------- Full pipeline: raw data -> train -> predict ----------

def test_full_pipeline_runs_without_error():
    df = _make_sample_df()
    df = clean_data(df)
    features_df = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(features_df, test_size=0.34, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_model_save_load_predict_roundtrip(tmp_path):
    """Simulates the real app flow: train -> save -> load -> predict on a new URL."""
    df = _make_sample_df()
    df = clean_data(df)
    features_df = build_feature_matrix(df)
    X = features_df.drop(columns=["label"])
    y = features_df["label"]

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    loaded_model = joblib.load(model_path)

    new_url_features = pd.DataFrame([extract_features("http://login-verify-secure.net")])
    prediction = loaded_model.predict(new_url_features)

    assert prediction[0] in [0, 1]


def test_feature_columns_consistent_between_train_and_predict():
    """
    Catches the #1 real bug in these pipelines: the columns used at
    training time must exactly match the columns produced at prediction time.
    """
    df = _make_sample_df()
    df = clean_data(df)
    features_df = build_feature_matrix(df)
    train_columns = set(features_df.drop(columns=["label"]).columns)

    predict_features = extract_features("https://some-new-url.com")
    predict_columns = set(predict_features.keys())

    assert train_columns == predict_columns


# ---------- Error handling integration ----------

def test_load_raw_data_raises_on_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("not_a_url,not_a_label\na,b\n")

    with pytest.raises(ValueError):
        load_raw_data(str(bad_csv))


def test_pipeline_handles_single_class_gracefully():
    """If a sample has only one class, split should still work with stratify disabled."""
    df = pd.DataFrame({
        "url": ["http://a.com", "http://b.com", "http://c.com", "http://d.com"],
        "label": [0, 0, 0, 0],
    })
    df = clean_data(df)
    features_df = build_feature_matrix(df)

    X = features_df.drop(columns=["label"])
    y = features_df["label"]
    assert len(X) == 4
    assert y.nunique() == 1