"""
Model training module.
Trains multiple classifiers on the same data and saves each one to disk,
so evaluate.py can load and compare them fairly.
"""

import os
import time
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_and_prepare

MODELS_DIR = "results/models"

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}


def train_all_models(X_train, y_train, models: dict = None) -> dict:
    """
    Trains every model in the given dict (defaults to MODELS) on X_train/y_train.
    Returns a dict of {model_name: trained_model} and prints training time for each.
    """
    if models is None:
        models = MODELS

    trained = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"[{name}] trained in {elapsed:.3f}s")
        trained[name] = model

    return trained


def save_models(trained_models: dict, out_dir: str = MODELS_DIR) -> None:
    """Saves each trained model to disk as a .joblib file."""
    os.makedirs(out_dir, exist_ok=True)
    for name, model in trained_models.items():
        path = os.path.join(out_dir, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"Saved {name} -> {path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare("data/phishing_urls.csv")

    print(f"\nTraining on {X_train.shape[0]} samples, {X_train.shape[1]} features...\n")
    trained_models = train_all_models(X_train, y_train)

    print("\nSaving models...\n")
    save_models(trained_models)

    print("\nDone. Run evaluate.py next to compare performance.")