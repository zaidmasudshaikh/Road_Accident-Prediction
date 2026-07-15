"""
app.py
Streamlit application for Road Accident Severity Prediction.
Run from project root: streamlit run app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_preprocessing import SEVERITY_LABELS  # noqa: E402

MODEL_PATH = "models/best_model.pkl"
MODEL_INFO_PATH = "models/model_info.txt"
COMPARISON_PATH = "outputs/model_comparison.csv"

# Valid category values, extracted directly from the actual training data (Phase 1/6)
FEATURE_OPTIONS = {
    "Age_band_of_driver": ["18-30", "31-50", "Over 51", "Under 18", "Unknown"],
    "Sex_of_driver": ["Male", "Female", "Unknown"],
    "Educational_level": [
        "Above high school", "Junior high school", "Elementary school",
        "High school", "Illiterate", "Writing & reading", "Unknown",
    ],
    "Vehicle_driver_relation": ["Employee", "Owner", "Other", "Unknown"],
    "Driving_experience": [
        "Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "No Licence", "Unknown",
    ],
    "Lanes_or_Medians": [
        "Undivided Two way", "One way", "Double carriageway (median)",
        "Two-way (divided with solid lines road marking)",
        "Two-way (divided with broken lines road marking)", "other", "Unknown",
    ],
    "Types_of_Junction": [
        "No junction", "Y Shape", "Crossing", "O Shape", "T Shape", "X Shape", "Other", "Unknown",
    ],
    "Road_surface_type": [
        "Asphalt roads", "Asphalt roads with some distress", "Earth roads",
        "Gravel roads", "Other", "Unknown",
    ],
    "Light_conditions": [
        "Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting",
    ],
    "Weather_conditions": [
        "Normal", "Raining", "Raining and Windy", "Cloudy", "Windy", "Fog or mist", "Snow", "Other", "Unknown",
    ],
    "Type_of_collision": [
        "Vehicle with vehicle collision", "Collision with roadside-parked vehicles",
        "Collision with pedestrians", "Rollover", "Collision with animals",
        "Collision with roadside objects", "Fall from vehicles", "With Train", "Other", "Unknown",
    ],
    "Vehicle_movement": [
        "Going straight", "Overtaking", "Turnover", "U-Turn", "Waiting to go",
        "Entering a junction", "Stopping", "Reversing", "Moving Backward",
        "Getting off", "Parked", "Other", "Unknown",
    ],
    "Pedestrian_movement": [
        "Not a Pedestrian",
        "Crossing from driver's nearside",
        "Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle",
        "Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle",
        "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)",
        "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle",
        "Walking along in carriageway, facing traffic",
        "Walking along in carriageway, back to traffic",
        "Unknown or other",
    ],
    "Cause_of_accident": [
        "No distancing", "Changing lane to the right", "Changing lane to the left",
        "Driving carelessly", "No priority to vehicle", "Moving Backward",
        "No priority to pedestrian", "Overtaking", "Driving under the influence of drugs",
        "Driving to the left", "Getting off the vehicle improperly", "Driving at high speed",
        "Overturning", "Turnover", "Overspeed", "Overloading", "Drunk driving",
        "Improper parking", "Other", "Unknown",
    ],
}

SEVERITY_EXPLANATIONS = {
    "Fatal injury": "This combination of factors is associated with the highest-risk accident patterns in the training data (loss of life).",
    "Serious Injury": "This combination of factors is associated with accidents causing significant but non-fatal injuries in the training data.",
    "Slight Injury": "This combination of factors is associated with minor-injury accident patterns in the training data.",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_model_info():
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            return f.read()
    return "Model info not available."


def get_comparison_table():
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH)
    return None


def main():
    st.set_page_config(page_title="Road Accident Severity Prediction", page_icon="🚗", layout="centered")

    st.title("🚗 Road Accident Severity Prediction System")
    st.write(
        "This application predicts the likely **severity** of a road accident "
        "(Slight Injury, Serious Injury, or Fatal injury) based on driver, road, "
        "and environmental conditions, using a machine learning model trained on "
        "real road traffic accident records."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Trained model not found at models/best_model.pkl. "
            "Please run `python3 src/evaluate_model.py` first to train and save the model."
        )
        return

    model = load_model()

    st.header("Enter Accident-Related Details")

    col1, col2 = st.columns(2)
    with col1:
        age_band = st.selectbox("Age band of driver", FEATURE_OPTIONS["Age_band_of_driver"])
        sex = st.selectbox("Sex of driver", FEATURE_OPTIONS["Sex_of_driver"])
        education = st.selectbox("Educational level", FEATURE_OPTIONS["Educational_level"])
        relation = st.selectbox("Vehicle driver relation", FEATURE_OPTIONS["Vehicle_driver_relation"])
        experience = st.selectbox("Driving experience", FEATURE_OPTIONS["Driving_experience"])
        lanes = st.selectbox("Lanes or medians", FEATURE_OPTIONS["Lanes_or_Medians"])
        junction = st.selectbox("Type of junction", FEATURE_OPTIONS["Types_of_Junction"])

    with col2:
        road_surface = st.selectbox("Road surface type", FEATURE_OPTIONS["Road_surface_type"])
        light = st.selectbox("Light conditions", FEATURE_OPTIONS["Light_conditions"])
        weather = st.selectbox("Weather conditions", FEATURE_OPTIONS["Weather_conditions"])
        collision = st.selectbox("Type of collision", FEATURE_OPTIONS["Type_of_collision"])
        movement = st.selectbox("Vehicle movement", FEATURE_OPTIONS["Vehicle_movement"])
        pedestrian = st.selectbox("Pedestrian movement", FEATURE_OPTIONS["Pedestrian_movement"])
        cause = st.selectbox("Cause of accident", FEATURE_OPTIONS["Cause_of_accident"])

    if st.button("Predict Accident Severity", type="primary"):
        input_data = pd.DataFrame([{
            "Age_band_of_driver": age_band,
            "Sex_of_driver": sex,
            "Educational_level": education,
            "Vehicle_driver_relation": relation,
            "Driving_experience": experience,
            "Lanes_or_Medians": lanes,
            "Types_of_Junction": junction,
            "Road_surface_type": road_surface,
            "Light_conditions": light,
            "Weather_conditions": weather,
            "Type_of_collision": collision,
            "Vehicle_movement": movement,
            "Pedestrian_movement": pedestrian,
            "Cause_of_accident": cause,
        }])

        prediction = model.predict(input_data)[0]
        severity_label = SEVERITY_LABELS[prediction]

        st.subheader("Prediction Result")
        if severity_label == "Fatal injury":
            st.error(f"Predicted Severity: **{severity_label}**")
        elif severity_label == "Serious Injury":
            st.warning(f"Predicted Severity: **{severity_label}**")
        else:
            st.success(f"Predicted Severity: **{severity_label}**")

        st.write(SEVERITY_EXPLANATIONS[severity_label])

        # Show prediction probabilities if supported
        if hasattr(model.named_steps["classifier"], "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            proba_df = pd.DataFrame({
                "Severity": [SEVERITY_LABELS[c] for c in model.named_steps["classifier"].classes_],
                "Probability": proba,
            }).sort_values("Probability", ascending=False)
            st.write("Model confidence:")
            st.dataframe(proba_df, hide_index=True, use_container_width=True)

        # Optional AI explanation (Phase 7) - never affects the prediction itself
        try:
            from ai_explanation import generate_explanation
            with st.spinner("Generating AI risk factor explanation..."):
                result = generate_explanation(input_data.iloc[0].to_dict(), severity_label)
            st.subheader("AI-Generated Risk Factor Explanation")
            if result["success"]:
                st.write(result["text"])
            else:
                st.info(f"AI explanation unavailable: {result['reason']}")
        except ImportError:
            pass
        except Exception as e:
            st.info(f"AI explanation unavailable: {e}")

    st.divider()
    st.header("Model Information & Evaluation Results")
    st.text(get_model_info())

    comparison_df = get_comparison_table()
    if comparison_df is not None:
        st.write("Model comparison (test set results):")
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    else:
        st.info("Model comparison table not found. Run src/evaluate_model.py to generate it.")


if __name__ == "__main__":
    main()
