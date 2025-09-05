import os, argparse, yaml, sys
import mlflow
from mlflow.tracking import MlflowClient

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def latest_by_stage(client, name, stage):
    versions = client.search_model_versions(f"name='{name}'")
    by_stage = [v for v in versions if (v.current_stage or "").lower() == stage.lower()]
    return by_stage[0] if by_stage else None

def main(stage: str):
    os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI","http://54.147.138.39:8081"))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    params = load_yaml("params.yaml")
    policy = load_yaml("governance.tags.yaml")
    req = policy.get("required", [])
    constraints = policy.get("constraints", {})

    mv = latest_by_stage(client, params["registered_model_name"], stage)
    if not mv:
        print(f"[GOV] No version in stage {stage}")
        sys.exit(1)

    tags = mv.tags or {}
    missing = [item["key"] if isinstance(item, dict) else item for item in req if (tags.get(item if isinstance(item,str) else item["key"]) is None)]
    problems = []

    # constraints
    for key, allowed in constraints.items():
        val = tags.get(key)
        if val is None:
            problems.append(f"missing constrained tag: {key}")
        elif val not in allowed:
            problems.append(f"invalid {key}={val} (allowed: {allowed})")

    if missing or problems:
        print(f"[GOV] Stage={stage} version={mv.version} FAILED")
        if missing:  print("Missing:", missing)
        if problems: print("Problems:", problems)
        sys.exit(2)

    print(f"[GOV] Stage={stage} version={mv.version} PASSED")
    sys.exit(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["Staging","Production"])
    args = ap.parse_args()
    main(args.stage)

