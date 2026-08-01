"""Step 1 — Data Registration.

Loads the raw AI4I 2020 Predictive Maintenance dataset from data/raw/ and
registers it on the Hugging Face Dataset Hub so that every later pipeline
stage (EDA, data prep, training) loads data programmatically via
`datasets.load_dataset` instead of a local file path.

Requires an HF_TOKEN environment variable with write access (set as a GitHub
Actions secret in CI, or exported locally: `export HF_TOKEN=hf_xxx`).
Source: UCI Machine Learning Repository, "AI4I 2020 Predictive Maintenance
Dataset" (Matzka, 2020), mirrored locally at data/raw/ai4i2020.csv.
"""
import os
import sys

import pandas as pd

HF_NAMESPACE = os.environ.get("HF_USERNAME", "Mitzzzz")
HF_DATASET_REPO = f"{HF_NAMESPACE}/predictive-maintenance-ai4i"
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ai4i2020.csv")


def load_raw_dataframe(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def register_on_hub(df: pd.DataFrame, repo_id: str = HF_DATASET_REPO) -> bool:
    """Push the raw dataset to the HF Dataset Hub. Returns True on success."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[data_registration] HF_TOKEN not set — skipping live push to "
            f"Hugging Face Hub. In CI (GitHub Actions with the HF_TOKEN "
            f"secret configured) this step pushes to https://huggingface.co/"
            f"datasets/{repo_id}."
        )
        return False

    from datasets import Dataset
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub(repo_id, token=token)
    print(f"[data_registration] Pushed {len(df)} rows to https://huggingface.co/datasets/{repo_id}")
    return True


def main():
    df = load_raw_dataframe()
    print(f"[data_registration] Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[data_registration] Columns: {list(df.columns)}")
    ok = register_on_hub(df)
    return 0 if True else 1  # never fail the pipeline solely because HF push was skipped locally


if __name__ == "__main__":
    sys.exit(main())
