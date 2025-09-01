import os
import mlflow

os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI", "http://54.147.151.249:8081"))
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.MlflowClient()

name = "w7d1_cancer_classifier"
print(f"Registered Model: {name}")
for mv in client.search_model_versions(f"name='{name}'"):
    # mv fields: version, current_stage, run_id, creation_timestamp, last_updated_timestamp, tags
    print(f"  v{mv.version}  stage={mv.current_stage or 'None'}  run_id={mv.run_id}")

