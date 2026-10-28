# Predictive Maintenance System for Milling and Turning (Mechanical + AI)

A software-only system that predicts machine failure, identifies the likely
failure type, gives a risk level and recommends maintenance actions.
No hardware or sensors are needed.

## Problem
Machine breakdowns cause downtime, high repair cost and safety risk.
Predictive maintenance uses data to find problems before the machine fails.

## Dataset
Augmented version of the AI4I 2020 Predictive Maintenance Dataset
(synthetic, 10,000 rows, 16 columns). The original rows are kept, and these
columns were added or changed (see `scripts/augment_dataset.py`):

| Column | Unit | Status |
|---|---|---|
| Machine Type (Milling / Turning) | none | New |
| Cutting force | N | New |
| Tool flank wear VB | mm | Replaces tool wear in minutes |
| Quality Type (L / M / H) | none | Renamed from Type |

Failure labels: HDF, PWF and RNF are unchanged. TWF keeps its original rows
(all with flank wear of at least 0.30 mm). OSF is recalculated from
flank wear x cutting force. Machine failure is 1 when any failure type is 1
(355 failures, 3.55%).

All new values are simulated, so results show the method and not real
machine performance.

## Features
- Failure prediction with a tuned XGBoost model (class imbalance handled
  with class weights and a tuned threshold of 0.25)
- Failure type diagnosis: TWF, HDF, PWF and OSF (one model per type)
- Risk level: Low, Medium or High
- Rule-based maintenance advice
- Streamlit dashboard: single machine and bulk CSV prediction

## Engineered mechanical features
| Feature | Formula | Related failure |
|---|---|---|
| temp_diff | process temp - air temp | Heat dissipation (HDF) |
| power | torque x rpm x 2pi / 60 | Power failure (PWF) |
| wear_force | flank wear x cutting force | Overstrain (OSF) |

## Project modules
1. Setup and data loading
2. Exploratory data analysis
3. Preprocessing and feature engineering
4. Model training
5. Evaluation and tuning
6. Failure type diagnosis
7. Risk scoring and recommendations
8. Streamlit dashboard
9. Packaging and documentation

## Results (test set, 2,000 rows, 71 failures)
| Metric | Value |
|---|---|
| Precision | [your value] |
| Recall | [your value] |
| F1-score | [your value] |
| ROC-AUC | 0.975 |
| PR-AUC | 0.902 |

Accuracy alone is misleading because only about 3.5% of rows are failures.

## How to run
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app/app.py

## Limitations
- The dataset is synthetic, and the new columns are simulated
- Only 71 failures in the test set, so scores can vary
- No time-series data, so remaining useful life is not predicted
- Probabilities are very close to 0 or 1 because the data follows exact rules

## Future scope
Real machine data, remaining useful life, SHAP explanations, an API.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit