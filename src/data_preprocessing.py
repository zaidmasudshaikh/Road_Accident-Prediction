"""
data_preprocessing.py
Loads, cleans, and prepares the Road Traffic Accident dataset for modeling.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "data/cleaned.csv"
TARGET_COL = "Accident_severity"

# All 14 predictor columns are categorical (verified against actual dataset)
CATEGORICAL_FEATURES = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Educational_level",
    "Vehicle_driver_relation",
    "Driving_experience",
    "Lanes_or_Medians",
    "Types_of_Junction",
    "Road_surface_type",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Vehicle_movement",
    "Pedestrian_movement",
    "Cause_of_accident",
]

# Target class mapping (confirmed in Phase 1)
SEVERITY_LABELS = {0: "Fatal injury", 1: "Serious Injury", 2: "Slight Injury"}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows (390 confirmed present) and reset index."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df


def get_feature_target(df: pd.DataFrame):
    """Split dataframe into features (X) and target (y)."""
    X = df[CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that one-hot encodes all categorical features.
    handle_unknown='ignore' ensures unseen categories at prediction time
    (e.g. from the Streamlit app) don't break the pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ]
    )
    return preprocessor


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split (target classes are imbalanced)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_and_prepare(path: str = DATA_PATH):
    """Convenience function: load -> clean -> split into X_train/X_test/y_train/y_test."""
    df = load_data(path)
    df = clean_data(df)
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print("Train target distribution:")
    print(y_train.value_counts(normalize=True).round(3))
    print("Test target distribution:")
    print(y_test.value_counts(normalize=True).round(3))
