"""Step 4 — Model Building with Experimentation Tracking.

Trains six tree-based / ensemble classifiers (Decision Tree, Bagging, Random
Forest, AdaBoost, Gradient Boosting, XGBoost) with GridSearchCV, logs every
run (params + metrics + model artifact) to a local MLflow tracking store
under ./mlruns, picks the best model by F1-score (primary — balances catching
real failures against flooding maintenance crews with false alarms) with
Recall reported as the key secondary business metric, saves it under
models/, and registers it on the HF Model Hub.
"""
import json
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from utils import TARGET_COLUMN

HF_NAMESPACE = os.environ.get("HF_USERNAME", "Mitzzzz")
HF_MODEL_REPO = f"{HF_NAMESPACE}/predictive-maintenance-model"
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")
EXPERIMENT_NAME = "predictive_maintenance_final"

# Real class ratio in AI4I 2020 is ~96.61% / 3.39% -> ~28.5:1
SCALE_POS_WEIGHT = 28

MODEL_GRID = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [5, 10, 20, None], "criterion": ["gini", "entropy"]},
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_samples": [0.7, 0.9, 1.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None], "min_samples_split": [2, 5]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="logloss"),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
            "scale_pos_weight": [1, SCALE_POS_WEIGHT],
        },
    ),
}


def load_splits():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def run_all_experiments(X_train, X_test, y_train, y_test, cv: int = 5, scoring: str = "f1"):
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(MLRUNS_DIR, 'mlflow.db')}")
    # Explicit absolute artifact_location so logged model artifacts always
    # land under MLRUNS_DIR regardless of the caller's working directory
    # (MLflow's default artifact root is otherwise relative to cwd).
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifact_dir = os.path.abspath(os.path.join(MLRUNS_DIR, "artifacts"))
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"file:{artifact_dir}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    fitted_models = {}

    for name, (estimator, grid) in MODEL_GRID.items():
        with mlflow.start_run(run_name=name.replace(" ", "_")):
            gs = GridSearchCV(estimator, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            metrics = evaluate(best, X_test, y_test)

            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_family", name)
            if name == "XGBoost":
                mlflow.xgboost.log_model(best, name="model")
            else:
                mlflow.sklearn.log_model(best, name="model")

            results.append({"model": name, "best_params": gs.best_params_, **metrics})
            fitted_models[name] = best
            print(f"[train] {name}: recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}")

    results_df = pd.DataFrame(results).sort_values(["f1", "recall"], ascending=False).reset_index(drop=True)
    return results_df, fitted_models


def select_and_save_best(results_df: pd.DataFrame, fitted_models: dict):
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(
            {
                "model_family": best_name,
                "metrics": results_df.iloc[0].drop(["model", "best_params"]).to_dict(),
                "best_params": results_df.iloc[0]["best_params"],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[train] Best model: {best_name} -> saved to {MODELS_DIR}/best_model.pkl")
    return best_name, best_model


def register_on_hub(repo_id: str = HF_MODEL_REPO) -> bool:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[train] HF_TOKEN not set — skipping live push of the best model "
            f"to https://huggingface.co/{repo_id}."
        )
        return False

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    for fname in ["best_model.pkl", "best_model_info.json", "scaler.pkl", "feature_names.json"]:
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            api.upload_file(path_or_fileobj=fpath, path_in_repo=fname, repo_id=repo_id, repo_type="model")
    print(f"[train] Registered model artifacts at https://huggingface.co/{repo_id}")
    return True


def main():
    X_train, X_test, y_train, y_test = load_splits()
    results_df, fitted_models = run_all_experiments(X_train, X_test, y_train, y_test)
    results_df.to_csv(os.path.join(MODELS_DIR, "experiment_results.csv"), index=False)
    select_and_save_best(results_df, fitted_models)
    register_on_hub()
    return 0


if __name__ == "__main__":
    sys.exit(main())
