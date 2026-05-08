
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "Output/model.pkl"
DATA_PATH = "Dataset/dataset_ready_for_modeling.csv"
SHAP_PATH = "Output/shap_top_features.json"
PHASE2_PATH = "Output/phase2_results.json"

st.set_page_config(page_title="CVD Risk Prediction App", layout="wide")
st.title("CVD Risk Prediction App")
st.write("This app estimates cardiovascular disease risk using the top 9 most important variables from SHAP analysis.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["any_cvd"], errors="ignore")
    defaults = X.median(numeric_only=True).to_dict()
    for col in X.columns:
        if col not in defaults:
            defaults[col] = X[col].mode(dropna=True).iloc[0]
    return X, defaults

@st.cache_data
def load_shap_data():
    with open(SHAP_PATH, "r") as f:
        shap_data = json.load(f)
    return shap_data

@st.cache_data
def load_threshold():
    try:
        with open(PHASE2_PATH, "r") as f:
            phase2 = json.load(f)
        return float(phase2["Logistic Regression"].get("threshold", 0.55))
    except Exception:
        return 0.55

model = load_model()
X_reference, defaults = load_data()
shap_data = load_shap_data()
threshold = load_threshold()

TOP_9_OVERALL = shap_data.get("TOP_9_OVERALL")
if TOP_9_OVERALL is None:
    TOP_9_OVERALL = list(shap_data.get("all_features_ranked", {}).keys())[:9]

display_names = {
    "age_at_baseline": "Age at baseline",
    "sex": "Sex",
    "taking_bp_medication": "Taking blood pressure medication",
    "smoking_status": "Smoking status",
    "hdl_cholesterol": "HDL cholesterol",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "apnea_events_per_hour": "Apnea events per hour",
    "waist_circumference_cm": "Waist circumference (cm)",
    "history_of_diabetes": "History of diabetes",
    "self_reported_hypertension": "Self-reported hypertension",
    "race": "Race",
    "cholesterol": "Total cholesterol",
    "bmi": "BMI",
    "min_oxygen_saturation": "Minimum oxygen saturation",
    "avg_oxygen_saturation": "Average oxygen saturation",
    "sleep_efficiency_pct": "Sleep efficiency (%)",
    "wake_after_sleep_onset_min": "Wake after sleep onset (min)",
}

category_map = {
    "apnea_events_per_hour": "Sleep",
    "avg_oxygen_saturation": "Sleep",
    "min_oxygen_saturation": "Sleep",
    "sleep_efficiency_pct": "Sleep",
    "wake_after_sleep_onset_min": "Sleep",
}

categorical_options = {
    "sex": {"Male": 1, "Female": 2},
    "race": {"White": 1, "Black": 2, "Other": 3},
    "smoking_status": {"Never": 0, "Former": 1, "Current": 2},
    "self_reported_hypertension": {"No": 0, "Yes": 1},
    "history_of_diabetes": {"No": 0, "Yes": 1},
    "taking_bp_medication": {"No": 0, "Yes": 1},
}

st.subheader("Patient Information")
st.caption("Only the top 9 SHAP-ranked variables are shown. Other model variables use population median or default values.")

user_input = defaults.copy()
cols = st.columns(3)

for i, feature in enumerate(TOP_9_OVERALL):
    label = display_names.get(feature, feature)
    with cols[i % 3]:
        if feature in categorical_options:
            options = list(categorical_options[feature].keys())
            default_value = defaults.get(feature, list(categorical_options[feature].values())[0])
            default_label = next((k for k, v in categorical_options[feature].items() if v == default_value), options[0])
            user_choice = st.selectbox(label, options, index=options.index(default_label))
            user_input[feature] = categorical_options[feature][user_choice]
        else:
            default_value = float(defaults.get(feature, 0))
            min_value = 0.0
            max_value = max(default_value * 3, 300.0)
            if "oxygen_saturation" in feature or "sleep_efficiency" in feature:
                max_value = 100.0
            elif feature == "age_at_baseline":
                max_value = 100.0
            user_input[feature] = st.number_input(label, min_value=min_value, max_value=max_value, value=default_value, step=1.0)

input_df = pd.DataFrame([user_input], columns=X_reference.columns)
probability = model.predict_proba(input_df)[0, 1]

if probability >= threshold:
    risk_level = "High Risk"
    badge = "High"
elif probability >= 0.30:
    risk_level = "Medium Risk"
    badge = "Medium"
else:
    risk_level = "Low Risk"
    badge = "Low"

st.divider()
st.subheader("Prediction Result")
col1, col2, col3 = st.columns(3)
col1.metric("Estimated CVD Risk", f"{probability * 100:.1f}%")
col2.metric("Risk Level", f"{badge}: {risk_level}")
col3.metric("Model Threshold", f"{threshold:.2f}")

st.write(
    "This result is a risk estimate based on the information entered. "
    "It is not a medical diagnosis. It should be used to support early screening and health awareness."
)

st.subheader("Top 9 Variables Used in This App")
top9_table = pd.DataFrame({
    "Rank": range(1, len(TOP_9_OVERALL) + 1),
    "Variable": [display_names.get(f, f) for f in TOP_9_OVERALL],
    "Category": [category_map.get(f, "Health") for f in TOP_9_OVERALL],
    "Mean |SHAP|": [shap_data.get("top_9_shap_values", {}).get(f, shap_data.get("all_features_ranked", {}).get(f, np.nan)) for f in TOP_9_OVERALL],
})
st.dataframe(top9_table, use_container_width=True, hide_index=True)

st.subheader("Suggested Interpretation")
if risk_level == "High Risk":
    st.write("This patient may benefit from further medical review, especially for blood pressure, sleep apnea, and cardiovascular screening.")
elif risk_level == "Medium Risk":
    st.write("This patient may benefit from monitoring and lifestyle review, especially if symptoms or other risk factors are present.")
else:
    st.write("This patient has a lower estimated risk based on the current input values, but regular health checkups are still important.")

st.caption("Educational project only. This app does not replace advice from a doctor or healthcare professional.")
