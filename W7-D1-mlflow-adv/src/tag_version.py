import os, argparse, yaml, json, subprocess
import mlflow
from mlflow.tracking import MlflowClient

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def git_sha():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def infer_features_count(client, run_id):
    try:
        run = client.get_run(run_id)
        # Try to read n_features from params if logged; else fallback via model
        nf = run.data.params.get("n_features_in_", None)
        if nf: return nf
    except Exception:
        pass
    return None

def parse_kv_list(kv: str):
    out = {}
    if not kv: return out
    for pair in kv.split(","):
        if not pair.strip(): continue
        k,v = pair.split("=",1)
        out[k.strip()] = v.strip()
    return out

def main(version: int, extra: str, stage: str):
    os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI","http://54.147.138.39:8081"))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    params = load_yaml("params.yaml")
    name = params["registered_model_name"]
    mv = client.get_model_version(name=name, version=str(version))

    # Build default tags
    defaults = {
        "owner": "samarth",
        "use.case": "classification:breast_cancer_demo",
        "git.sha": git_sha(),
        "serve.slo.p95_ms": "200",
        "risk.tier": "low",
        "pii": "none",
    }
    # try to infer features count
    nf = infer_features_count(client, mv.run_id)
    if nf: defaults["data.schema.features"] = str(nf)

    # caller extras override defaults
    extras = parse_kv_list(extra)
    payload = {**defaults, **extras}
    if stage:
        payload["stage.intent"] = stage  # optional hint

    # apply tags
    for k,v in payload.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
        print(f"[tag] {k}={v}")

    print(f"[OK] Tagged {name} v{mv.version}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--set", type=str, default="", help='Comma list: k=v,k2=v2')
    ap.add_argument("--stage", type=str, default="", help="Optional intended stage")
    args = ap.parse_args()
    main(args.version, args.set, args.stage)

