"""
evaluate_model.py
Evaluates trained model pipelines: accuracy, weighted precision/recall/F1,
classification report, and confusion matrix. Saves reports and confusion
matrix images to outputs/.
Run from project root: python3 src/evaluate_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import load_and_prepare, SEVERITY_LABELS
from train_model import train_all_models

CM_OUTPUT_DIR = "outputs/confusion_matrices"
REPORT_OUTPUT_DIR = "outputs"
MODEL_OUTPUT_DIR = "models"


def evaluate_model(name, pipeline, X_test, y_test):
    """Compute all required metrics for a single trained pipeline."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    labels_sorted = sorted(SEVERITY_LABELS.keys())
    target_names = [SEVERITY_LABELS[l] for l in labels_sorted]

    report_text = classification_report(
        y_test, y_pred, labels=labels_sorted, target_names=target_names, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    return metrics, report_text, cm, target_names


def save_classification_report(name, report_text):
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    path = f"{REPORT_OUTPUT_DIR}/classification_report_{safe_name}.txt"
    with open(path, "w") as f:
        f.write(f"Classification Report - {name}\n")
        f.write("=" * 50 + "\n")
        f.write(report_text)
    return path


def save_confusion_matrix_plot(name, cm, target_names):
    os.makedirs(CM_OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    path = f"{CM_OUTPUT_DIR}/confusion_matrix_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def evaluate_all(trained_models, X_test, y_test):
    """Evaluate every trained model, save reports/confusion matrices, return metrics list."""
    all_metrics = []
    for name, pipeline in trained_models.items():
        metrics, report_text, cm, target_names = evaluate_model(name, pipeline, X_test, y_test)
        all_metrics.append(metrics)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"Weighted Precision: {metrics['Precision (weighted)']:.4f}")
        print(f"Weighted Recall: {metrics['Recall (weighted)']:.4f}")
        print(f"Weighted F1-score: {metrics['F1-score (weighted)']:.4f}")
        print(report_text)

        save_classification_report(name, report_text)
        save_confusion_matrix_plot(name, cm, target_names)

    return all_metrics


def build_comparison_table(all_metrics):
    """Build a DataFrame comparing all models, sorted by weighted F1-score (descending)."""
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("F1-score (weighted)", ascending=False).reset_index(drop=True)
    return df


def save_comparison_table(df):
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    path = f"{REPORT_OUTPUT_DIR}/model_comparison.csv"
    df.to_csv(path, index=False)
    return path


def select_best_model(comparison_df, trained_models):
    """Select the model with the highest weighted F1-score."""
    best_name = comparison_df.iloc[0]["Model"]
    return best_name, trained_models[best_name]


def save_best_model(name, pipeline):
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    path = f"{MODEL_OUTPUT_DIR}/best_model.pkl"
    joblib.dump(pipeline, path)
    with open(f"{MODEL_OUTPUT_DIR}/model_info.txt", "w") as f:
        f.write(f"Best model: {name}\nSelection criterion: highest weighted F1-score\n")
    return path


def plot_feature_importance(name, pipeline):
    """Generate feature importance plot if the model supports it (tree-based models)."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        print(f"{name} does not support feature_importances_; skipping.")
        return None

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = classifier.feature_importances_

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=fi_df, x="importance", y="feature", hue="feature", legend=False, palette="crest")
    plt.title(f"Top 15 Feature Importances - {name}")
    plt.tight_layout()
    path = f"{REPORT_OUTPUT_DIR}/feature_importance_{name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare()
    trained_models = train_all_models(X_train, y_train)
    all_metrics = evaluate_all(trained_models, X_test, y_test)

    comparison_df = build_comparison_table(all_metrics)
    print("\n=== Model Comparison (sorted by weighted F1-score) ===")
    print(comparison_df.to_string(index=False))
    save_comparison_table(comparison_df)

    best_name, best_pipeline = select_best_model(comparison_df, trained_models)
    print(f"\nBest model selected: {best_name}")
    save_best_model(best_name, best_pipeline)

    fi_path = plot_feature_importance(best_name, best_pipeline)
    if fi_path:
        print(f"Feature importance plot saved to {fi_path}")
