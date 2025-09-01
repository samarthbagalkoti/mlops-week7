import os, argparse, sys, yaml
import mlflow
from mlflow import MlflowClient

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_metric_from_run(client: MlflowClient, run_id: str, metric_key: str):
    if not run_id:
        return None
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_key)

def stage_to_alias(stage: str) -> str:
    mapping = {"Production": "production", "Staging": "staging"}
    return mapping.get(stage, stage.lower())

def main(candidate_version: int):
    # URIs
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_REGISTRY_URI", registry_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()
    policy = load_yaml("policy.yaml")
    params = load_yaml("params.yaml")

    model_name = params["registered_model_name"]
    metric_key = policy["primary_metric"]
    allowed_regression = float(policy["allowed_regression"])
    allow_first = bool(policy.get("allow_first_promotion", True))

    # Candidate
    cand = client.get_model_version(name=model_name, version=str(candidate_version))
    cand_metric = get_metric_from_run(client, cand.run_id, metric_key)

    # Current "production" via alias (no stages)
    prod_alias = stage_to_alias("Production")
    try:
        prod = client.get_model_version_by_alias(model_name, prod_alias)
    except Exception:
        prod = None

    if not prod:
        if allow_first:
            print(f"[GATE] No current alias '{prod_alias}'. Allowing first promotion. "
                  f"Candidate v{candidate_version} {metric_key}={cand_metric}.")
            sys.exit(0)
        else:
            print(f"[GATE] No version bound to alias '{prod_alias}' and first promotion not allowed.")
            sys.exit(1)

    prod_metric = get_metric_from_run(client, prod.run_id, metric_key)

    if cand_metric is None or prod_metric is None:
        print(f"[GATE] Missing metric '{metric_key}' on candidate or production. "
              f"Candidate={cand_metric}, Production={prod_metric}")
        sys.exit(1)

    delta = float(cand_metric) - float(prod_metric)
    print(f"[GATE] Candidate v{candidate_version} {metric_key}={cand_metric:.6f} | "
          f"Current '{prod_alias}' v{prod.version} {metric_key}={prod_metric:.6f} | Δ={delta:.6f}")

    if delta >= -allowed_regression:
        print("[GATE] PASS")
        sys.exit(0)
    else:
        print("[GATE] FAIL (exceeds allowed regression)")
        sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-version", type=int, required=True)
    args = ap.parse_args()
    main(args.candidate_version)

