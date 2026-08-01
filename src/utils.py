"""Shared data-cleaning and feature-engineering logic used by the notebook,
data_prep.py, train.py and the Streamlit inference app so all four stay
in sync on the exact same transformation.
"""
import numpy as np
import pandas as pd

RAW_COLUMN_MAP = {
    "UDI": "UDI",
    "Product ID": "Product_ID",
    "Type": "Type",
    "Air temperature [K]": "Air_Temperature_K",
    "Process temperature [K]": "Process_Temperature_K",
    "Rotational speed [rpm]": "Rotational_Speed_rpm",
    "Torque [Nm]": "Torque_Nm",
    "Tool wear [min]": "Tool_Wear_min",
    "Machine failure": "Machine_Failure",
    "TWF": "TWF",
    "HDF": "HDF",
    "PWF": "PWF",
    "OSF": "OSF",
    "RNF": "RNF",
}

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

LEAKAGE_COLUMNS = ["UDI", "Product_ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

CONTINUOUS_FEATURES = [
    "Air_Temperature_K",
    "Process_Temperature_K",
    "Rotational_Speed_rpm",
    "Torque_Nm",
    "Tool_Wear_min",
    "Power_W",
    "Delta_Temp_K",
    "Wear_Torque",
]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + ["Type"]
TARGET_COLUMN = "Machine_Failure"


def standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw UCI/HF column names to a consistent snake_case schema."""
    df = df.rename(columns=RAW_COLUMN_MAP)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-derived features and encode the categorical Type column.

    Power_W captures the mechanical power delivered to the spindle, which is
    the physical driver behind Power Failures (PWF). Delta_Temp_K captures the
    heat-dissipation margin (driver of HDF) instead of two highly-correlated
    raw temperatures. Wear_Torque captures the compounding stress that leads
    to Overstrain Failures (OSF).
    """
    df = df.copy()
    df["Power_W"] = df["Rotational_Speed_rpm"] * df["Torque_Nm"] * (2 * np.pi / 60)
    df["Delta_Temp_K"] = df["Process_Temperature_K"] - df["Air_Temperature_K"]
    df["Wear_Torque"] = df["Tool_Wear_min"] * df["Torque_Nm"]
    df["Type"] = df["Type"].map(TYPE_MAP)
    return df


def clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: standardize columns, engineer features, drop
    leakage / identifier columns. Does NOT scale or split — that is a
    modelling-stage concern kept separate to avoid train/test leakage.
    """
    df = standardize_raw_columns(df_raw)
    df = engineer_features(df)
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    return df
