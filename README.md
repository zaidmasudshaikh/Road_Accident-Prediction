# Road Accident Severity Prediction System

## Overview

A machine learning system that predicts the severity of road traffic accidents (**Slight Injury**, **Serious Injury**, or **Fatal injury**) based on driver, road, and environmental conditions. Built as a data science internship project, it includes a full ML pipeline, an interactive Streamlit web application, and an optional AI-powered explanation feature.

## Problem Statement

Road traffic accidents vary widely in severity depending on conditions such as driver experience, road type, weather, and cause of accident. Predicting severity in advance — using factors already known at or near the time of an incident — can help identify high-risk conditions and inform road safety interventions.

## Objectives

- Analyze a real-world Road Traffic Accident dataset to understand patterns behind accident severity
- Build and compare multiple classification models
- Select and deploy the best-performing model
- Provide a simple, usable interface for predictions
- Add an optional AI-generated explanation of predictions for interpretability

## Dataset Information

- **Source**: Kaggle Road Traffic Accidents dataset
- **File used**: `data/cleaned.csv` (a pre-cleaned version of the original `RTA_Dataset.csv`, with missing values already filled as `"Unknown"`)
- **Rows**: 12,316 (11,926 after removing 390 duplicate rows found during preprocessing)
- **Features used**: 14 categorical columns — `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`, `Vehicle_driver_relation`, `Driving_experience`, `Lanes_or_Medians`, `Types_of_Junction`, `Road_surface_type`, `Light_conditions`, `Weather_conditions`, `Type_of_collision`, `Vehicle_movement`, `Pedestrian_movement`, `Cause_of_accident`
- **Target**: `Accident_severity` — `0` = Fatal injury (158 records, 1.3%), `1` = Serious Injury (1,743 records, 14.6%), `2` = Slight Injury (10,415 records, 84.1%)
- **Class imbalance**: severe (~66:1 between Slight Injury and Fatal injury) — addressed using `class_weight='balanced'`

## Technologies Used

- Python 3
- pandas, scikit-learn (ColumnTransformer, Pipeline, OneHotEncoder)
- matplotlib, seaborn (visualization)
- joblib (model serialization)
- Streamlit (web application)
- python-dotenv, requests (AI API integration)
- Anthropic API (optional explanation feature)

## Project Architecture

```
road-accident-prediction/
├── data/
│   └── cleaned.csv
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # loading, cleaning, splitting, ColumnTransformer
│   ├── eda.py                  # exploratory data analysis, saves plots
│   ├── train_model.py          # trains Logistic Regression, Decision Tree, Random Forest
│   ├── evaluate_model.py       # evaluates models, comparison table, best model selection/saving
│   └── ai_explanation.py       # optional AI-generated explanation (Anthropic API)
├── models/
│   ├── best_model.pkl
│   └── model_info.txt
├── outputs/
│   ├── eda/                              (9 EDA plots)
│   ├── confusion_matrices/               (3 confusion matrix images)
│   ├── classification_report_*.txt       (per-model reports)
│   ├── model_comparison.csv
│   └── feature_importance_random_forest.png
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Data Preprocessing

1. Load `data/cleaned.csv`
2. Drop 390 duplicate rows (confirmed present via `.duplicated().sum()`)
3. Split into 14 categorical features (`X`) and target (`y`)
4. `ColumnTransformer` applies `OneHotEncoder(handle_unknown='ignore')` to all 14 features (no numeric scaling needed — no numeric predictor columns exist in this dataset)
5. Stratified 80/20 train/test split (preserves class proportions in both sets)

## Exploratory Data Analysis

Generated and saved to `outputs/eda/`:
- Accident severity class distribution
- Missing value check (confirms 0 missing post-cleaning)
- 6 categorical-feature-vs-severity plots (Cause of accident, Type of collision, Light conditions, Weather conditions, Driving experience, Age band of driver)
- Label-encoded correlation heatmap (an approximation, since all features are nominal categories — not a true Pearson correlation)

## Machine Learning Models

Three classifiers trained with `class_weight='balanced'` to address class imbalance:
1. **Logistic Regression** (`max_iter=1000`)
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (`n_estimators=100, max_depth=20` — tuned to keep the saved model file under 30 MB)

## Evaluation Metrics

Each model evaluated using: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-score, full Classification Report, and Confusion Matrix.

## Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---|---|
| **Random Forest** | 0.8277 | 0.7364 | 0.8277 | **0.7682** |
| Decision Tree | 0.7054 | 0.7242 | 0.7054 | 0.7145 |
| Logistic Regression | 0.4380 | 0.7571 | 0.4380 | 0.5317 |

*(exact values also saved in `outputs/model_comparison.csv`)*

**Important caveat (for viva discussion):** Random Forest achieves the highest weighted F1-score, but its per-class performance is heavily skewed toward the majority class ("Slight Injury"). Its precision/recall on "Fatal injury" and "Serious Injury" are close to 0, despite `class_weight='balanced'`. This is a known limitation of optimizing for weighted F1 on a severely imbalanced dataset — the metric rewards majority-class accuracy. Logistic Regression, by contrast, detects Fatal injury cases much better (recall 0.62) at the cost of overall accuracy. This tradeoff is worth explaining honestly rather than hiding.

## Best Model

**Random Forest**, selected by highest weighted F1-score (as specified in the project plan). Saved as `models/best_model.pkl` (full pipeline: preprocessing + classifier).

Top contributing features (see `outputs/feature_importance_random_forest.png`): Types of Junction, Educational level, Lanes/Medians type, Age band of driver, Vehicle movement, among others — all with fairly evenly distributed importance (no single dominant feature).

## Installation Instructions (macOS)

### 1. Set up the virtual environment
```bash
cd road-accident-prediction
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set up environment variables (optional, for AI explanation feature)
```bash
cp .env.example .env
# then edit .env and add your real ANTHROPIC_API_KEY
```

## How to Train the Model

```bash
python3 src/eda.py              # generates EDA plots in outputs/eda/
python3 src/evaluate_model.py   # trains all 3 models, evaluates, compares, saves best model
```

This regenerates: classification reports, confusion matrices, `outputs/model_comparison.csv`, `models/best_model.pkl`, `models/model_info.txt`, and the feature importance plot.

## How to Run the Streamlit Application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`) in your browser.

## API Configuration

The optional AI explanation feature (`src/ai_explanation.py`) calls the Anthropic API to generate a short, human-readable explanation of contributing risk factors **after** the local ML model has already made its prediction. The API never performs the prediction itself.

- Set `ANTHROPIC_API_KEY` in `.env` (see `.env.example`)
- If the key is missing or the request fails, the app displays a clear message and continues working normally — no crash

## Project Folder Structure

See [Project Architecture](#project-architecture) above.
