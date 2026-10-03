"""
Model evaluation module.
Loads every trained model, evaluates it on the same test set,
and produces a comparison table plus confusion matrices and ROC curves.
"""

import os
import glob
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_loader import load_and_prepare

MODELS_DIR = "results/models"
PLOTS_DIR = "results/plots"
METRICS_PATH = "results/metrics.csv"


def load_all_models(models_dir: str = MODELS_DIR) -> dict:
    """Loads every .joblib file in the models directory into a dict."""
    models = {}
    for path in glob.glob(os.path.join(models_dir, "*.joblib")):
        name = os.path.splitext(os.path.basename(path))[0]
        models[name] = joblib.load(path)
    return models


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluates every model on the same test set.
    Returns a DataFrame sorted by F1-score, best model first.
    """
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = preds  # fallback for models without predict_proba

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan"),
        })

    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return results


def plot_confusion_matrices(models: dict, X_test, y_test, out_dir: str = PLOTS_DIR) -> None:
    """Saves one confusion matrix image per model."""
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Phishing"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_{name}.png"))
        plt.close()


def plot_roc_curves(models: dict, X_test, y_test, out_dir: str = PLOTS_DIR) -> None:
    """Saves one combined ROC curve plot with all models overlaid."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        probs = model.predict_proba(X_test)[:, 1]
        if len(set(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_comparison.png"))
    plt.close()


if __name__ == "__main__":
    _, X_test, _,y_test = load_and_prepare("data/phishing_urls.csv", sample_size=20000)
    print("Loading trained models...")
    models = load_all_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}\n")

    print("Evaluating...\n")
    results = evaluate_all(models, X_test, y_test)
    print(results.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    results.to_csv(METRICS_PATH, index=False)
    print(f"\nSaved metrics to {METRICS_PATH}")

    print("\nGenerating plots...")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    print(f"Saved plots to {PLOTS_DIR}/")