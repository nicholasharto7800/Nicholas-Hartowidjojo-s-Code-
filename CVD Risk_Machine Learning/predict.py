"""
predict.py — CVD Risk Prediction CLI

Loads the trained Logistic Regression model and predicts CVD risk
for one or more patients supplied as a CSV file.

Usage:
    python src/predict.py --input Dataset/sample_input_patient.csv

Input CSV must contain all 18 feature columns:
    apnea_events_per_hour, avg_oxygen_saturation, min_oxygen_saturation,
    sleep_efficiency_pct, wake_after_sleep_onset_min, bmi,
    waist_circumference_cm, systolic_bp, diastolic_bp, cholesterol,
    hdl_cholesterol, self_reported_hypertension, history_of_diabetes,
    taking_bp_medication, age_at_baseline, sex, race, smoking_status

    sex coding: 1 = Male, 2 = Female
    race coding: 1 = White, 2 = Black/African American, 3 = Other
    smoking_status: 0.0 = Never, 1.0 = Former, 2.0 = Current
    self_reported_hypertension: 0.0 = No, 1.0 = Yes
    history_of_diabetes: 0.0 = No, 1.0 = Yes
    taking_bp_medication: 0.0 = No, 1.0 = Yes

Output: prediction label and CVD probability for each patient row.
"""

import argparse
import sys
import os
import pandas as pd
import joblib

ROOT       = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(ROOT, 'Output', 'model.pkl')
THRESHOLD  = 0.55   # tuned in Step 9 to achieve Recall >= 0.70

REQUIRED_FEATURES = [
    'apnea_events_per_hour', 'avg_oxygen_saturation', 'min_oxygen_saturation',
    'sleep_efficiency_pct', 'wake_after_sleep_onset_min', 'bmi',
    'waist_circumference_cm', 'systolic_bp', 'diastolic_bp', 'cholesterol',
    'hdl_cholesterol', 'self_reported_hypertension', 'history_of_diabetes',
    'taking_bp_medication', 'age_at_baseline', 'sex', 'race', 'smoking_status',
]


def predict_cvd_risk(input_path):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    try:
        df_input = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    missing_cols = [f for f in REQUIRED_FEATURES if f not in df_input.columns]
    if missing_cols:
        print(f"ERROR: missing columns in input CSV: {missing_cols}")
        sys.exit(1)

    X = df_input[REQUIRED_FEATURES]
    probabilities = model.predict_proba(X)[:, 1]
    predictions   = (probabilities >= THRESHOLD).astype(int)

    print("=" * 58)
    print("  CVD Risk Prediction Results")
    print("=" * 58)
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = "CVD Risk Detected" if pred == 1 else "No CVD Risk Detected"
        print(f"  Patient {i + 1:>2}: {label}  ({prob * 100:.1f}% probability)")
    print("=" * 58)
    print("  NOTE: For educational use only.")
    print("        Not a substitute for medical advice.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict CVD risk from patient sleep and health data.')
    parser.add_argument('--input', required=True,
                        help='Path to input CSV with patient data')
    args = parser.parse_args()
    predict_cvd_risk(args.input)
