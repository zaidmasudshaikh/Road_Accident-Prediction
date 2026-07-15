"""
eda.py
Exploratory Data Analysis for the Road Traffic Accident dataset.
Generates and saves plots to outputs/eda/.
Run from project root: python3 src/eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_data, clean_data, CATEGORICAL_FEATURES, TARGET_COL, SEVERITY_LABELS

OUTPUT_DIR = "outputs/eda"
sns.set_style("whitegrid")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame):
    """Accident severity class distribution (count + %)."""
    counts = df[TARGET_COL].map(SEVERITY_LABELS).value_counts()
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Accident Severity Distribution")
    plt.ylabel("Count")
    plt.xlabel("Severity")
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v + total * 0.01, f"{v} ({v/total*100:.1f}%)", ha="center")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=150)
    plt.close()


def plot_missing_values(df: pd.DataFrame):
    """Missing value check (dataset is pre-cleaned, so this confirms zero missing)."""
    missing = df.isnull().sum()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=missing.values, y=missing.index, color="steelblue")
    plt.title("Missing Values per Column (post-cleaning)")
    plt.xlabel("Missing Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=150)
    plt.close()


def plot_categorical_vs_severity(df: pd.DataFrame, columns):
    """For each selected categorical feature, show its relationship with severity."""
    df_plot = df.copy()
    df_plot[TARGET_COL] = df_plot[TARGET_COL].map(SEVERITY_LABELS)

    for col in columns:
        plt.figure(figsize=(10, 5))
        order = df_plot[col].value_counts().index
        sns.countplot(
            data=df_plot, y=col, hue=TARGET_COL, order=order,
            hue_order=["Fatal injury", "Serious Injury", "Slight Injury"],
        )
        plt.title(f"{col} vs Accident Severity")
        plt.xlabel("Count")
        plt.legend(title="Severity", loc="lower right")
        plt.tight_layout()
        safe_name = col.lower()
        plt.savefig(f"{OUTPUT_DIR}/{safe_name}_vs_severity.png", dpi=150)
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    All features are categorical, so we label-encode a copy purely for a
    relative-association heatmap. This is an approximation, not a true
    Pearson correlation (noted here and in the README).
    """
    from sklearn.preprocessing import OrdinalEncoder

    df_enc = df.copy()
    encoder = OrdinalEncoder()
    df_enc[CATEGORICAL_FEATURES] = encoder.fit_transform(df_enc[CATEGORICAL_FEATURES])

    corr = df_enc[CATEGORICAL_FEATURES + [TARGET_COL]].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Label-Encoded Feature Correlation (approximation)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()


def run_eda():
    ensure_output_dir()
    df = load_data()
    df = clean_data(df)

    plot_target_distribution(df)
    plot_missing_values(df)

    # Most relevant categorical features for accident severity analysis
    key_features = [
        "Cause_of_accident",
        "Type_of_collision",
        "Light_conditions",
        "Weather_conditions",
        "Driving_experience",
        "Age_band_of_driver",
    ]
    plot_categorical_vs_severity(df, key_features)

    plot_correlation_heatmap(df)

    print(f"EDA complete. Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_eda()
