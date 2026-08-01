"""Step 5 — Model Deployment (hosting script).

Pushes the deployment/ folder (Dockerfile, app.py, inference.py, utils.py,
requirements.txt, README.md with Space metadata) to a Hugging Face Space,
so that the Space always reflects the latest code on `main`. The Space
itself downloads the current best model from the HF Model Hub at runtime
(see deployment/inference.py), so no model binary needs to be pushed here.

Requires an HF_TOKEN environment variable with write access.
"""
import os
import shutil
import sys

HF_NAMESPACE = os.environ.get("HF_USERNAME", "Mitzzzz")
HF_SPACE_REPO = f"{HF_NAMESPACE}/predictive-maintenance-app"
DEPLOYMENT_DIR = os.path.join(os.path.dirname(__file__), "..", "deployment")
UTILS_SRC = os.path.join(os.path.dirname(__file__), "utils.py")


def push_space(repo_id: str = HF_SPACE_REPO) -> bool:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[deploy_to_space] HF_TOKEN not set — skipping live deployment to "
            f"https://huggingface.co/spaces/{repo_id}. In CI (GitHub Actions "
            f"with the HF_TOKEN secret configured) this step pushes the "
            f"deployment/ folder as a Docker-SDK Space."
        )
        return False

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)

    # utils.py lives in src/ in the main repo but must ship alongside
    # inference.py inside the Space's Docker build context.
    staged_utils = os.path.join(DEPLOYMENT_DIR, "utils.py")
    shutil.copy(UTILS_SRC, staged_utils)
    try:
        api.upload_folder(
            folder_path=DEPLOYMENT_DIR,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy latest predictive maintenance app via CI",
        )
    finally:
        if os.path.exists(staged_utils):
            os.remove(staged_utils)

    # HF_TOKEN itself is also needed *inside* the Space at runtime so the app
    # can download the model from the Model Hub. Space secrets can only be
    # set once (they are not overwritten on every push), so this is
    # idempotent and safe to call on every deploy.
    try:
        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
    except Exception as exc:  # pragma: no cover - permission/plan dependent
        print(f"[deploy_to_space] Could not set Space secret automatically ({exc}); set HF_TOKEN in the Space settings manually.")

    print(f"[deploy_to_space] Deployed to https://huggingface.co/spaces/{repo_id}")
    return True


def main():
    ok = push_space()
    return 0


if __name__ == "__main__":
    sys.exit(main())
