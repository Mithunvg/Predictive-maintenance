# MLOps — Predictive Maintenance

Capstone project: an end-to-end MLOps pipeline that predicts machine failure
from live sensor telemetry, trained on the [AI4I 2020 Predictive Maintenance
Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(UCI ML Repository, Matzka 2020). Data and model registration run through the
Hugging Face Hub, experiments are tracked with MLflow, and the whole pipeline
— data → training → deployment — is automated with GitHub Actions,
deploying a Streamlit app to a Hugging Face Space.

## Deliverables

- **`MithunVG_PredictiveMaintenance_FinalReport.pdf`** — the business report (primary evaluation artefact).
- **`MithunVG_PredictiveMaintenance_Notebook.html`** — the fully executed notebook (code verification artefact).

## Repository Structure

```
data/raw, data/processed   Raw and cleaned/split datasets
notebooks/                 Executed end-to-end notebook (source: final_notebook.py, percent format)
src/                       Reusable pipeline modules
  utils.py                 Shared feature engineering (single source of truth)
  data_registration.py     Step 1 — push raw dataset to HF Dataset Hub
  data_prep.py              Step 2 — clean, engineer features, split, push splits
  train.py                  Step 3 — GridSearchCV x 6 models, MLflow tracking, register best model
  deploy_to_space.py        Step 4 — push Streamlit app to a HF Space
models/                    Serialized model artifacts + experiment results
mlruns/                    Local MLflow tracking store (SQLite)
deployment/                Dockerfile, Streamlit app, inference logic, HF Space README
reports/figures/           Chart images used in the PDF report
.github/workflows/         pipeline.yml — CI/CD automation
```

## Running the pipeline locally

```bash
pip install -r requirements.txt
python src/data_registration.py   # add HF_TOKEN env var to actually push to the Hub
python src/data_prep.py
python src/train.py
python src/deploy_to_space.py
```

Without an `HF_TOKEN` environment variable, every stage still runs fully
using local files under `data/` and `models/` — only the live Hugging Face
push is skipped (and clearly logged as skipped).

## CI/CD: GitHub Actions

`.github/workflows/pipeline.yml` runs the four steps above on every push to
`main`, using a repository secret `HF_TOKEN` (a Hugging Face token with
write access). **One-time setup required to go live:**

1. Create a Hugging Face access token (write scope): https://huggingface.co/settings/tokens
2. In this GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**, name it `HF_TOKEN`.
3. Merge/push to `main` (or run the workflow manually via **Actions → Predictive Maintenance MLOps Pipeline → Run workflow**).

Once that runs successfully, the following go live:

- Dataset: `https://huggingface.co/datasets/mithunvg/predictive-maintenance-ai4i`
- Model: `https://huggingface.co/mithunvg/predictive-maintenance-model`
- App: `https://huggingface.co/spaces/mithunvg/predictive-maintenance-app`

## Best model

Selected by F1-score across 6 tuned model families (Decision Tree, Bagging,
Random Forest, AdaBoost, Gradient Boosting, XGBoost) — see the report,
Section 4, for the full comparison and rationale.
