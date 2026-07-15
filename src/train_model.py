"""
train_model.py
Trains three classifiers (Logistic Regression, Decision Tree, Random Forest)
on the Road Traffic Accident dataset using a shared preprocessing pipeline.
Run from project root: python3 src/train_model.py
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_and_prepare, build_preprocessor

RANDOM_STATE = 42


def get_models():
    """Return a dict of {model_name: sklearn estimator}. class_weight='balanced'
    is used on all three to address the severe class imbalance found in Phase 1."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, max_depth=20,
            random_state=RANDOM_STATE
        ),
    }


def build_pipeline(estimator):
    """Combine the shared preprocessor with a given estimator."""
    preprocessor = build_preprocessor()
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


def train_all_models(X_train, y_train):
    """Train all three models and return a dict of {name: fitted pipeline}."""
    trained = {}
    for name, estimator in get_models().items():
        print(f"Training {name}...")
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
    return trained


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare()
    trained_models = train_all_models(X_train, y_train)
    print(f"\nTrained {len(trained_models)} models: {list(trained_models.keys())}")
