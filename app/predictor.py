from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------
# Load the saved files
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

scaler = joblib.load(MODELS_DIR / "scaler.joblib")
final_model = joblib.load(MODELS_DIR / "final_model.joblib")
threshold = joblib.load(MODELS_DIR / "threshold.joblib")
diagnosis_models = joblib.load(MODELS_DIR / "diagnosis_models.joblib")

QUALITY_MAP = {"L": 0, "M": 1, "H": 2}
MACHINE_MAP = {"Milling": 0, "Turning": 1}

# Must be the same columns and the same order as in training
feature_cols = [
    "quality_type", "machine_type", "air_temp", "process_temp", "rpm",
    "torque", "cutting_force", "flank_wear", "temp_diff", "power", "wear_force"
]

OSF_LIMIT = {"L": 1200, "M": 1320, "H": 1600}


# ---------------------------------------------------------
# Build the features for one machine
# ---------------------------------------------------------
def build_features(quality_type, machine_type, air_temp, process_temp,
                   rpm, torque, cutting_force, flank_wear):
    row = {
        "quality_type": QUALITY_MAP[quality_type],
        "machine_type": MACHINE_MAP[machine_type],
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rpm": rpm,
        "torque": torque,
        "cutting_force": cutting_force,
        "flank_wear": flank_wear,
        "temp_diff": process_temp - air_temp,
        "power": torque * rpm * 2 * np.pi / 60,
        "wear_force": flank_wear * cutting_force,
    }
    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------
# Risk level and maintenance advice
# ---------------------------------------------------------
def get_risk_level(prob):
    if prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    else:
        return "Low"


def get_recommendations(quality_type, machine_type, air_temp, process_temp,
                        rpm, torque, cutting_force, flank_wear):
    advice = []

    temp_diff = process_temp - air_temp
    power = torque * rpm * 2 * np.pi / 60
    wear_force = flank_wear * cutting_force

    if flank_wear >= 0.30:
        advice.append("Flank wear has reached the 0.30 mm limit: replace the tool.")

    if temp_diff < 8.6 and rpm < 1380:
        advice.append("Low temperature difference at low speed: check the cooling system.")

    if power < 3500:
        advice.append("Power is too low: check for slipping or an unstable load.")
    elif power > 9000:
        advice.append("Power is too high: reduce the load or speed.")

    if wear_force > OSF_LIMIT[quality_type]:
        advice.append("Overstrain risk: reduce the cutting force (lower feed or depth of cut) or replace the worn tool.")

    if not advice:
        advice.append("All readings are in the normal range. Continue regular monitoring.")

    return advice


# ---------------------------------------------------------
# Predict one machine
# ---------------------------------------------------------
def predict_machine(quality_type, machine_type, air_temp, process_temp,
                    rpm, torque, cutting_force, flank_wear):
    features = build_features(quality_type, machine_type, air_temp, process_temp,
                              rpm, torque, cutting_force, flank_wear)
    scaled = pd.DataFrame(scaler.transform(features), columns=feature_cols)

    prob = float(final_model.predict_proba(scaled)[0, 1])

    type_probs = {
        t: float(diagnosis_models[t].predict_proba(scaled)[0, 1])
        for t in diagnosis_models
    }

    return {
        "failure_probability": round(prob, 3),
        "failure_alert": bool(prob >= threshold),
        "risk_level": get_risk_level(prob),
        "likely_cause": max(type_probs, key=type_probs.get),
        "cause_probabilities": {k: round(v, 3) for k, v in type_probs.items()},
        "calculated": {
            "temp_diff": round(float(features["temp_diff"].iloc[0]), 2),
            "power": round(float(features["power"].iloc[0]), 1),
            "wear_force": round(float(features["wear_force"].iloc[0]), 1),
        },
        "advice": get_recommendations(quality_type, machine_type, air_temp, process_temp,
                                      rpm, torque, cutting_force, flank_wear),
    }


# ---------------------------------------------------------
# Predict many machines (CSV page)
# ---------------------------------------------------------
RENAME_MAP = {
    "Quality Type": "quality_type",
    "Machine Type": "machine_type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Cutting force [N]": "cutting_force",
    "Tool flank wear VB [mm]": "flank_wear",
}

REQUIRED_COLS = [
    "quality_type", "machine_type", "air_temp", "process_temp",
    "rpm", "torque", "cutting_force", "flank_wear"
]


def predict_batch(df):
    df = df.rename(columns=RENAME_MAP)

    feats = pd.DataFrame({
        "quality_type": df["quality_type"].map(QUALITY_MAP),
        "machine_type": df["machine_type"].map(MACHINE_MAP),
        "air_temp": df["air_temp"],
        "process_temp": df["process_temp"],
        "rpm": df["rpm"],
        "torque": df["torque"],
        "cutting_force": df["cutting_force"],
        "flank_wear": df["flank_wear"],
        "temp_diff": df["process_temp"] - df["air_temp"],
        "power": df["torque"] * df["rpm"] * 2 * np.pi / 60,
        "wear_force": df["flank_wear"] * df["cutting_force"],
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
    out.loc[out["risk_level"] == "Low", "likely_cause"] = "-"
    return out