from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

scaler = joblib.load(MODELS_DIR / "scaler.joblib")
final_model = joblib.load(MODELS_DIR / "final_model.joblib")
threshold = joblib.load(MODELS_DIR / "threshold.joblib")
diagnosis_models = joblib.load(MODELS_DIR / "diagnosis_models.joblib")

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

feature_cols = [
    "Type", "air_temp", "process_temp", "rpm", "torque",
    "tool_wear", "temp_diff", "power", "wear_torque"
]

def build_features(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    row = {
        "Type": TYPE_MAP[machine_type],
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rpm": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
        "temp_diff": process_temp - air_temp,
        "power": torque * rpm * 2 * np.pi / 60,
        "wear_torque": tool_wear * torque,
    }
    return pd.DataFrame([row])[feature_cols]

def get_risk_level(prob):
    if prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"

def get_recommendations(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    advice = []

    temp_diff = process_temp - air_temp
    power = torque * rpm * 2 * np.pi / 60
    wear_torque = tool_wear * torque
    overstrain_limit = {"L": 11000, "M": 12000, "H": 13000}[machine_type]

    if tool_wear >= 200:
        advice.append("Tool wear is high: replace or inspect the tool.")

    if temp_diff < 8.6 and rpm < 1380:
        advice.append("Low temperature difference at low speed: check the cooling system.")

    if power < 3500:
        advice.append("Power is too low: check for slipping or an unstable load.")
    elif power > 9000:
        advice.append("Power is too high: reduce the load or speed.")

    if wear_torque > overstrain_limit:
        advice.append("Overstrain risk: reduce torque on this worn tool.")

    if not advice:
        advice.append("All readings are in the normal range. Continue regular monitoring.")

    return advice

def predict_machine(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    features = build_features(machine_type, air_temp, process_temp, rpm, torque, tool_wear)
    scaled = pd.DataFrame(scaler.transform(features), columns=feature_cols)

    prob = float(final_model.predict_proba(scaled)[0, 1])

    type_probs = {
        t: float(diagnosis_models[t].predict_proba(scaled)[0, 1])
        for t in diagnosis_models
    }

    return {
        "failure_probability": round(prob, 3),
        "failure_alert": prob >= threshold,
        "risk_level": get_risk_level(prob),
        "likely_cause": max(type_probs, key=type_probs.get),
        "cause_probabilities": {k: round(v, 3) for k, v in type_probs.items()},
        "advice": get_recommendations(machine_type, air_temp, process_temp, rpm, torque, tool_wear),
    }

RENAME_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}

REQUIRED_COLS = ["Type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]


def predict_batch(df):
    df = df.rename(columns=RENAME_MAP)

    feats = pd.DataFrame({
        "Type": df["Type"].map(TYPE_MAP),
        "air_temp": df["air_temp"],
        "process_temp": df["process_temp"],
        "rpm": df["rpm"],
        "torque": df["torque"],
        "tool_wear": df["tool_wear"],
        "temp_diff": df["process_temp"] - df["air_temp"],
        "power": df["torque"] * df["rpm"] * 2 * np.pi / 60,
        "wear_torque": df["tool_wear"] * df["torque"],
    })[feature_cols]

    scaled = pd.DataFrame(scaler.transform(feats), columns=feature_cols)
    probs = final_model.predict_proba(scaled)[:, 1]

    type_probs = pd.DataFrame({
        t: m.predict_proba(scaled)[:, 1] for t, m in diagnosis_models.items()
    })

    out = df.copy()
    out["failure_probability"] = probs.round(3)
    out["risk_level"] = [get_risk_level(p) for p in probs]
    out["failure_alert"] = probs >= threshold
    out["likely_cause"] = type_probs.idxmax(axis=1).values
    return out   