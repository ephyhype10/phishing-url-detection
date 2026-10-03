"""
Dataset loading and cleaning module.
Reads the raw CSV, removes duplicates/nulls, and splits into
train/test sets ready for feature extraction and model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_extractor import extract_features


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'url' and 'label' columns. Found: {list(df.columns)}"
        )

    # Normalize label column to 0/1, regardless of pandas' internal string dtype
    label_map = {"good": 0, "bad": 1, "legitimate": 0, "phishing": 1,
                 "0": 0, "1": 1}
    normalized = df["label"].astype(str).str.strip().str.lower()

    if normalized.isin(label_map.keys()).any():
        df["label"] = normalized.map(label_map)
    # else: leave as-is, assume it's already numeric 0/1

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate URLs, drops rows with missing url/label,
    and ensures label is an integer (0 or 1).
    """
    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["url"])

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies extract_features() to every URL and returns a DataFrame
    of features plus the label column.
    """
    feature_rows = [extract_features(u) for u in df["url"]]
    features_df = pd.DataFrame(feature_rows)
    features_df["label"] = df["label"].values
    return features_df


def split_data(features_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features/labels into stratified train and test sets.
    Stratified so both classes are proportionally represented in each split.
    """
    X = features_df.drop(columns=["label"])
    y = features_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_and_prepare(csv_path: str, test_size: float = 0.2, random_state: int = 42, sample_size: int = None):
    """
    Convenience function: runs the full pipeline from raw CSV to
    train/test feature splits in one call. sample_size optionally
    limits how many rows are used (useful for large datasets).
    """
    df = load_raw_data(csv_path)
    df = clean_data(df)

    if sample_size and len(df) > sample_size:
        per_class = sample_size // 2
        legit = df[df["label"] == 0]
        phish = df[df["label"] == 1]

        legit_sample = legit.sample(min(len(legit), per_class), random_state=random_state)
        phish_sample = phish.sample(min(len(phish), per_class), random_state=random_state)

        df = pd.concat([legit_sample, phish_sample]).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)

    features_df = build_feature_matrix(df)
    return split_data(features_df, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare("data/phishing_urls.csv", sample_size=20000)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Class balance (train):\n", y_train.value_counts())