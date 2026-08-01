"""Streamlit inference app for the Predictive Maintenance model.

Deployed to a Hugging Face Space (see src/deploy_to_space.py). Loads the
registered best model + scaler from the HF Model Hub (mithunvg/predictive-
maintenance-model), takes the five raw sensor readings + product type a
technician would read off a machine, and returns a failure probability and
risk tier.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from inference import predict_from_raw_inputs, load_artifacts

st.set_page_config(page_title="Predictive Maintenance — Failure Risk Estimator", page_icon="🛠️", layout="centered")

st.title("🛠️ Predictive Maintenance — Machine Failure Risk Estimator")
st.markdown(
    """
    Enter live sensor readings for a machine to estimate the probability of
    an imminent failure. The model was trained on the **AI4I 2020 Predictive
    Maintenance Dataset** and is served from the Hugging Face Model Hub.
    """
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Project:** MLOps — Predictive Maintenance
        **Model source:** `mithunvg/predictive-maintenance-model`
        **Dataset source:** `mithunvg/predictive-maintenance-ai4i`

        This Space is deployed automatically by the GitHub Actions pipeline
        (`.github/workflows/pipeline.yml`) whenever the repository's `main`
        branch is updated.
        """
    )
    try:
        _, _, feature_names = load_artifacts()
        st.success("Model artifacts loaded ✅")
        with st.expander("Model input features"):
            st.write(feature_names)
    except Exception as exc:
        st.warning(f"Model artifacts not yet available: {exc}")

st.subheader("Machine Sensor Readings")
col1, col2 = st.columns(2)
with col1:
    machine_type = st.selectbox("Product Quality Type", options=["L", "M", "H"], index=0)
    air_temp = st.number_input("Air Temperature (K)", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
    process_temp = st.number_input("Process Temperature (K)", min_value=295.0, max_value=315.0, value=310.0, step=0.1)
with col2:
    rpm = st.number_input("Rotational Speed (rpm)", min_value=1000, max_value=3000, value=1500, step=10)
    torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=80.0, value=40.0, step=0.1)
    tool_wear = st.number_input("Tool Wear (min)", min_value=0, max_value=260, value=100, step=1)

if st.button("Predict Failure Risk", type="primary"):
    try:
        result = predict_from_raw_inputs(
            machine_type=machine_type,
            air_temp_k=air_temp,
            process_temp_k=process_temp,
            rotational_speed_rpm=rpm,
            torque_nm=torque,
            tool_wear_min=tool_wear,
        )
        st.metric("Failure Probability", f"{result['failure_probability'] * 100:.2f}%")
        risk_colors = {"Low": "green", "Moderate": "orange", "High": "red", "Critical": "red"}
        st.markdown(
            f"### Prediction: **{result['prediction']}**  |  Risk level: "
            f":{risk_colors[result['risk_level']]}[{result['risk_level']}]"
        )
        if result["prediction"] == "Failure":
            st.error("⚠️ Recommend scheduling maintenance inspection for this machine.")
        else:
            st.success("✅ No immediate maintenance action indicated.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Source: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning "
    "Repository (Matzka, 2020). Built for the MLOps Predictive Maintenance capstone project."
)
