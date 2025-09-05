import os, mlflow
os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI", "http://54:147.138.39:8081"))
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
from mlflow.tracking import MlflowClient
import yaml, sys

with open("params.yaml") as f:
    name = yaml.safe_load(f)["registered_model_name"]

c = MlflowClient()
versions = sorted(c.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
if not versions:
    print("", end="")
    sys.exit(1)
print(versions[-1].version)

