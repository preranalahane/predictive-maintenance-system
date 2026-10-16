# Predictive Maintenance System (Mechanical + AI)

A software-only system that predicts machine failure, identifies the likely
failure type and recommends maintenance actions. Built on the AI4I 2020
Predictive Maintenance Dataset. No hardware or sensors are needed.

## Problem
Machine breakdowns cause downtime, high repair cost and safety risk.
Predictive maintenance uses data to find problems before the machine fails.

## Dataset
- AI4I 2020 Predictive Maintenance Dataset (synthetic, 10,000 rows)
- Inputs: product type, air temperature, process temperature, rotational
  speed, torque, tool wear
- Target: machine failure (about 3.4% of rows are failures)
- Failure types: TWF, HDF, PWF, OSF (RNF skipped because it is random)

## Features
- Failure prediction with an XGBoost model (imbalance handled with class weights)
- Failure type diagnosis (one model per failure type)
- Risk level: Low, Medium or High
- Rule-based maintenance recommendations from mechanical limits
- Streamlit dashboard with single machine and bulk CSV prediction

## Engineered mechanical features
| Feature | Formula | Related failure |
|---|---|---|
| temp_diff | process temp - air temp | Heat dissipation (HDF) |
| power | torque x rpm x 2π / 60 | Power failure (PWF) |
| wear_torque | tool wear x torque | Overstrain (OSF) |

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



## How to run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Limitations
- The dataset is synthetic, so real machines may behave differently.
- The test set has only a few dozen failures, so scores can vary.
- There is no time-series data, so remaining useful life is not predicted.
- Risk bands (30% and 60%) are fixed and not calibrated.

## Future scope
- Train on real machine data
- Add remaining useful life prediction with time-series data
- Add SHAP explanations
- Deploy as an API

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit