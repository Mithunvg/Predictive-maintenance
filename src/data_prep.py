"""Step 3 — Data Preparation.

Loads the raw dataset (from the HF Hub if HF_TOKEN is available, otherwise
from the local mirror), cleans it, engineers features, scales continuous
features, performs a stratified train/test split, saves both splits locally
under data/processed/, and pushes them back to the HF Dataset Hub as
separate configs (train_split_v1 / test_split_v1).
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from utils import CONTINUOUS_FEATURES, TARGET_COLUMN, clean_and_engineer

HF_NAMESPACE = os.environ.get("HF_USERNAME", "mithunvg")
HF_DATASET_REPO = f"{HF_NAMESPACE}/predictive-maintenance-ai4i"
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ai4i2020.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_dataset_from_hub_or_local() -> pd.DataFrame:
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from datasets import load_dataset

            ds = load_dataset(HF_DATASET_REPO, token=token)
            df = ds["train"].to_pandas()
            print(f"[data_prep] Loaded {len(df)} rows from HF Hub: {HF_DATASET_REPO}")
            return df
        except Exception as exc:  # pragma: no cover - network/CI dependent
            print(f"[data_prep] Could not load from HF Hub ({exc}); falling back to local CSV")
    df = pd.read_csv(RAW_CSV_PATH)
    print(f"[data_prep] Loaded {len(df)} rows from local mirror: {RAW_CSV_PATH}")
    return df


def prepare(df_raw: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df = clean_and_engineer(df_raw)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    X_test[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    return train_df, test_df, scaler


def save_local(train_df: pd.DataFrame, test_df: pd.DataFrame, scaler: StandardScaler):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(list(train_df.drop(columns=[TARGET_COLUMN]).columns), f, indent=2)
    print(f"[data_prep] Saved train ({len(train_df)}) / test ({len(test_df)}) splits to {PROCESSED_DIR}")


def push_splits_to_hub(train_df: pd.DataFrame, test_df: pd.DataFrame, repo_id: str = HF_DATASET_REPO) -> bool:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[data_prep] HF_TOKEN not set — skipping live push of train/test "
            f"splits to https://huggingface.co/datasets/{repo_id}."
        )
        return False

    from datasets import Dataset

    Dataset.from_pandas(train_df, preserve_index=False).push_to_hub(
        repo_id, config_name="train_split_v1", token=token
    )
    Dataset.from_pandas(test_df, preserve_index=False).push_to_hub(
        repo_id, config_name="test_split_v1", token=token
    )
    print(f"[data_prep] Pushed train_split_v1 / test_split_v1 to https://huggingface.co/datasets/{repo_id}")
    return True


def main():
    df_raw = load_dataset_from_hub_or_local()
    train_df, test_df, scaler = prepare(df_raw)
    save_local(train_df, test_df, scaler)
    push_splits_to_hub(train_df, test_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
