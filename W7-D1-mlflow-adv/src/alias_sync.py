# src/alias_sync.py
import os
from mlflow import MlflowClient

def main():
    name = os.environ.get("MODEL_NAME", "w7d1_cancer_classifier")
    c = MlflowClient()

    # Pick current Production and Staging by stage
    mvs = list(c.search_model_versions(f"name='{name}'"))
    prod = max((mv for mv in mvs if mv.current_stage == "Production"),
               key=lambda mv: int(mv.version), default=None)
    staging = max((mv for mv in mvs if mv.current_stage == "Staging"),
                  key=lambda mv: int(mv.version), default=None)

    # Sync aliases
    if prod:
        c.set_registered_model_alias(name=name, alias="production", version=prod.version)
        print(f"[alias] production -> v{prod.version}")
    else:
        print("[alias] production: no version in Production; skipping")

    if staging:
        c.set_registered_model_alias(name=name, alias="staging", version=staging.version)
        print(f"[alias] staging -> v{staging.version}")
    else:
        print("[alias] staging: no version in Staging; skipping")

    # Optional: latest-candidate alias = highest version number overall
    if mvs:
        latest = max(mvs, key=lambda mv: int(mv.version))
        c.set_registered_model_alias(name=name, alias="latest-candidate", version=latest.version)
        print(f"[alias] latest-candidate -> v{latest.version}")

if __name__ == "__main__":
    main()

