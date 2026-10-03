"""
Dataset loading and cleaning module.
Reads the raw CSV, removes duplicates/nulls, and splits into
train/test sets ready for feature extraction and model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_extractor import extract_features


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the dataset CSV. Expects at least two columns: 'url' and 'label'
    (label: 1 = phishing, 0 = legitimate). Raises a clear error if columns
    are missing, instead of failing deep inside the pipeline.
    """
    df = pd.read_csv(csv_path)

    df.columns = [c.strip().lower() for c in df.columns]

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'url' and 'label' columns. Found: {list(df.columns)}"
        )

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


def load_and_prepare(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Convenience function: runs the full pipeline from raw CSV to
    train/test feature splits in one call.
    """
    df = load_raw_data(csv_path)
    df = clean_data(df)
    features_df = build_feature_matrix(df)
    return split_data(features_df, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare("data/phishing_urls.csv")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Class balance (train):\n", y_train.value_counts())