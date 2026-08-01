---
title: Predictive Maintenance
emoji: 🛠️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
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
