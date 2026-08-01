---
title: Predictive Maintenance
emoji: 🛠️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.60.0"
app_file: app.py
pinned: false
license: mit
---

# Predictive Maintenance — Failure Risk Estimator

Streamlit app serving the best model from the MLOps Predictive Maintenance
capstone project (`mithunvg/predictive-maintenance-model`), trained on the
AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository).

Enter live sensor readings (air/process temperature, rotational speed,
torque, tool wear, product quality type) to get an estimated failure
probability and risk tier.

Deployed automatically by the GitHub Actions pipeline in
[mithunvg/predictive-maintenance](https://github.com/mithunvg/predictive-maintenance)
whenever `main` is updated.

This Space runs on HF's native Streamlit SDK (free on `cpu-basic`). A
`Dockerfile` is also provided in this folder for containerized/self-hosted
deployment — HF Spaces' Docker and Gradio SDKs require a PRO subscription
on `cpu-basic` as of 2026, so the free-tier live Space uses the Streamlit
SDK instead while the Docker path remains fully functional and documented.
