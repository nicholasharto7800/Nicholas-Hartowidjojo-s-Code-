# CVD Risk Prediction Using Sleep and Health Data

**Course:** BSAN6070 Introduction to Machine Learning | Loyola Marymount University
**Authors:** Jerry Hsu, Nicholson, Nicholas
**Date:** April 2026
**Streamlit App:** [Launch App](https://shopsmart-jerry.streamlit.app)
**GitHub:** [JerryHsu1117/cvd-risk-prediction-shhs](https://github.com/JerryHsu1117/cvd-risk-prediction-shhs)

---

## Project Summary

This project applies machine learning to predict cardiovascular disease (CVD) risk using sleep and health data from 5,042 patients in the Sleep Heart Health Study (1995 to 1998). Three models were trained and evaluated. Logistic Regression achieved the best performance with AUC = 0.8018 and Recall = 0.7071, meeting both success criteria. The model correctly identifies 7 out of 10 CVD patients on unseen data.

The final model is deployed as an interactive Streamlit application where users input health data and receive a personalized CVD risk score.

---

## Research Questions

**Primary:** Can we predict CVD risk using sleep patterns and baseline health data?

**Secondary 1:** Which sleep metrics and health indicators drive CVD risk most? Do sleep metrics add meaningful predictive value?

**Secondary 2:** What are the top 9 most influential variables overall based on SHAP analysis?

---

## Key Results

| Model | Test AUC | Recall | F1 | Train Time |
|---|---|---|---|---|
| **Logistic Regression** | **0.8018** | **0.7071** | **0.5624** | 0.8s |
| Random Forest | 0.7905 | 0.7071 | 0.5391 | 13s |
| Gradient Boosting | 0.7893 | 0.7113 | 0.5304 | 82s |

**Selected model:** Logistic Regression. Highest AUC and F1. Most interpretable. Trains 100x faster than Gradient Boosting for a 0.004 Recall difference.

**Top 3 predictors by SHAP:** Age at baseline (0.7336), Sex (0.4002), Taking BP medication (0.3408)

**Sleep vs Health contribution:** Sleep metrics account for 5.6% of total SHAP importance. Health indicators account for 94.4%.

---

## Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| shhs1 (polysomnography) | 5,804 | 1,279 | Overnight sleep study data, 1995 to 1998 |
| CVD summary | 5,802 | 41 | 15-year CVD outcome follow-up, 1995 to 2011 |
| Merged (inner join) | 5,802 | 1,280 | Joined on patient ID (nsrrid) |
| Final modeling dataset | 5,042 | 19 | After cleaning: 18 features + target |

**Target variable:** any_cvd (0 = No CVD, 1 = CVD)
**Class distribution:** 76.3% No CVD / 23.7% CVD

**Data source:** [National Sleep Research Resource, Zhang et al. (2018)](https://sleepdata.org/datasets/shhs)

---

## Features Used

| Category | Features |
|---|---|
| Sleep (5) | Apnea events per hour, avg oxygen saturation, min oxygen saturation, sleep efficiency, wake after sleep onset |
| Health (13) | Age, sex, race, BMI, waist circumference, systolic BP, diastolic BP, cholesterol, HDL cholesterol, smoking status, self-reported hypertension, history of diabetes, taking BP medication |

---

## Project Structure

```
cvd-risk-prediction-shhs/
├── README.md
├── requirements.txt
├── notebooks/
│   └── cvd_risk_prediction.ipynb       # Full pipeline, 14 steps, all cells executed
├── src/
│   ├── app.py                          # Streamlit web application
│   └── predict.py                      # CLI prediction tool
├── Dataset/
│   └── sample_input_patient.csv        # Demo patient for live prediction
├── Output/
│   ├── model.pkl                       # Trained Logistic Regression pipeline
│   ├── shap_top_features.json          # SHAP rankings for Streamlit app
│   ├── phase2_results.json             # Tuning results
│   ├── eda_charts/                     # EDA visualizations
│   ├── evaluation_charts/             # ROC curves, confusion matrices
│   └── shap_waterfall/                 # Individual patient SHAP explanations
└── .gitignore
```

---

## Pipeline Steps

1. **Data Loading and Merging** — Inner join on patient ID, 5,802 rows x 1,280 columns
2. **Feature Selection** — Remove leakage, domain knowledge review, select 18 features
3. **Data Cleaning** — Drop 760 missing target rows, median imputation, IQR outlier capping
4. **Rename Features** — SHHS codes to descriptive English names
5. **EDA** — Class balance, t-tests, sex analysis, correlation heatmap
6. **Save Modeling Dataset** — Persist clean dataset before modeling
7. **Preprocessing** — sklearn pipeline, 80/20 train/test split, undersampling to 50/50
8. **Model Assumption Validation** — VIF check (max 4.50), leakage check, class balance check
9. **K Selection and Training** — K=3,5,10 stability test, RandomizedSearchCV tuning
10. **Evaluation** — AUC, Recall, confusion matrices, threshold tuning, subgroup analysis
11. **SHAP Interpretation** — LinearExplainer, top 9 features, beeswarm and waterfall plots
12. **Final Model Selection** — LR selected: highest AUC, F1, interpretability, fastest training
13. **Results and Conclusion** — Research questions answered, limitations, recommendations
14. **Deployment** — model.pkl, predict.py CLI, Streamlit app

---

## Performance Criteria

| Criteria | Threshold | Result |
|---|---|---|
| AUC | > 0.75 (target 0.80) | 0.8018 PASS |
| Recall | > 0.70 | 0.7071 PASS |
| Beats baseline accuracy | > 76.3% | YES |

---

## Subgroup Analysis (Logistic Regression)

| Group | n | AUC | Recall |
|---|---|---|---|
| Age < 65 | 513 | 0.7289 | 0.245 |
| Age >= 65 | 496 | 0.7208 | 0.826 |
| Male | 448 | 0.7579 | 0.790 |
| Female | 561 | 0.8422 | 0.617 |

The model is most reliable for patients aged 65 and older. Recall drops to 24.5% for patients under 65. Do not apply this model to younger patients without retraining.

---

## Streamlit App

The app reads the top 9 SHAP-ranked features from `Output/shap_top_features.json` and displays an input form for clinicians. Remaining features are filled with population medians.

**Launch locally:**
```bash
streamlit run src/app.py
```

**CLI predictions:**
```bash
python src/predict.py --input Dataset/sample_input_patient.csv
```

**Demo patient:** 68-year-old male, AHI = 20.5, hypertension, on BP medication. Predicted CVD probability: 85.1%.

---

## Limitations

- Dataset covers 1995 to 1998. Risk profiles may differ from modern populations.
- Recall = 24.5% for patients under 65. Model is unreliable for younger patients.
- For educational purposes only. Not validated for clinical diagnosis.

---

## Future Work

- Validate on post-2010 polysomnography datasets
- Test LightGBM and CatBoost for AUC improvement beyond 0.80
- Apply survival analysis to model time-to-CVD-event
- Collect features targeted at younger patients to close the subgroup gap

---

## Tools and Libraries

Python 3.10+, pandas, numpy, scikit-learn, matplotlib, seaborn, shap, joblib, streamlit

```bash
pip install -r requirements.txt
```
