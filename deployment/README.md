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

**Note on hosting cost (2026):** as of this writing, Hugging Face Spaces
requires a **PRO subscription** to run any interactive compute backend
(Gradio or Docker SDK) on `cpu-basic` — only `static` Spaces (no Python
backend) are free. This app therefore cannot be hosted live on a free HF
account; `src/deploy_to_space.py` fails soft on this (logs it, does not
crash the CI pipeline) so dataset/model registration still complete. The
Dockerfile and app.py in this folder are fully functional and ready to
deploy the moment PRO is available (or run locally via
`streamlit run app.py`, or on any free Docker-friendly host).
