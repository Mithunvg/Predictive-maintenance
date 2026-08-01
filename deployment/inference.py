"""Shared inference logic used by both the Streamlit app (app.py) and the
notebook's deployment smoke-test. Loads the registered model + scaler
(from the HF Model Hub if available, else a local fallback) and exposes a
single `predict_from_raw_inputs` function that takes the same raw sensor
readings a technician would read off a machine.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import CONTINUOUS_FEATURES, FEATURE_COLUMNS, TYPE_MAP

HF_NAMESPACE = os.environ.get("HF_USERNAME", "Mitzzzz")
HF_MODEL_REPO = f"{HF_NAMESPACE}/predictive-maintenance-model"
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_MODEL = None
_SCALER = None
_FEATURE_NAMES = None


def _resolve_artifact_dir() -> str:
    """Prefer artifacts downloaded from the HF Model Hub; fall back to the
    local models/ directory produced by src/train.py (e.g. when running
    inside this repo without network access to Hugging Face).
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_id=HF_MODEL_REPO, repo_type="model", token=token)
        except Exception as exc:  # pragma: no cover - network/CI dependent
            print(f"[inference] Could not fetch model from HF Hub ({exc}); using local artifacts")
    return LOCAL_MODELS_DIR


def load_artifacts():
    global _MODEL, _SCALER, _FEATURE_NAMES
    if _MODEL is not None:
        return _MODEL, _SCALER, _FEATURE_NAMES

    artifact_dir = _resolve_artifact_dir()
    _MODEL = joblib.load(os.path.join(artifact_dir, "best_model.pkl"))
    _SCALER = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
    with open(os.path.join(artifact_dir, "feature_names.json")) as f:
        _FEATURE_NAMES = json.load(f)
    return _MODEL, _SCALER, _FEATURE_NAMES


def build_feature_row(
    machine_type: str,
    air_temp_k: float,
    process_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> pd.DataFrame:
    row = {
        "Air_Temperature_K": air_temp_k,
        "Process_Temperature_K": process_temp_k,
        "Rotational_Speed_rpm": rotational_speed_rpm,
        "Torque_Nm": torque_nm,
        "Tool_Wear_min": tool_wear_min,
        "Type": TYPE_MAP.get(machine_type, machine_type),
    }
    row["Power_W"] = row["Rotational_Speed_rpm"] * row["Torque_Nm"] * (2 * np.pi / 60)
    row["Delta_Temp_K"] = row["Process_Temperature_K"] - row["Air_Temperature_K"]
    row["Wear_Torque"] = row["Tool_Wear_min"] * row["Torque_Nm"]
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_from_raw_inputs(
    machine_type: str,
    air_temp_k: float,
    process_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> dict:
    model, scaler, feature_names = load_artifacts()
    df_row = build_feature_row(
        machine_type, air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min
    )
    df_row = df_row[feature_names]
    df_scaled = df_row.copy()
    df_scaled[CONTINUOUS_FEATURES] = scaler.transform(df_row[CONTINUOUS_FEATURES])

    proba = float(model.predict_proba(df_scaled)[0, 1])
    prediction = int(proba >= 0.5)
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.5:
        risk = "Moderate"
    elif proba < 0.75:
        risk = "High"
    else:
        risk = "Critical"

    return {
        "failure_probability": round(proba, 4),
        "prediction": "Failure" if prediction else "No Failure",
        "risk_level": risk,
    }
