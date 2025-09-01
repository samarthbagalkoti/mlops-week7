import os, argparse, json, sys, subprocess, yaml
from datetime import datetime, timezone
import mlflow
from mlflow import MlflowClient

AUDIT_PATH = "logs/audit.jsonl"

def append_audit(entry: dict):
    os.makedirs("logs", exist_ok=True)
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")

def stage_to_alias(stage: str) -> str:
    mapping = {"Production": "production", "Staging": "staging"}
    return mapping.get(stage, stage.lower())

def gate(candidate_version: int) -> bool:
    cmd = [sys.executable, "src/compare_and_gate.py", "--candidate-version", str(candidate_version)]
    return subprocess.call(cmd, env=os.environ.copy()) == 0

def main(candidate_version: int, stage: str, dry_run: bool, reason: str, promoted_by: str):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_REGISTRY_URI", registry_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()
    model_name = load_yaml("params.yaml")["registered_model_name"]

    # Candidate version
    mv = client.get_model_version(name=model_name, version=str(candidate_version))

    # Gate
    if not gate(candidate_version):
        print("[PROMOTE] Gate failed. Aborting.")
        sys.exit(1)

    alias = stage_to_alias(stage)

    # Previous holder of the alias (if any)
    try:
        prev = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        prev = None

    print(f"[PROMOTE] Candidate v{mv.version} -> alias='{alias}' dry_run={dry_run}")
    if dry_run:
        print("[PROMOTE] Dry-run: no changes will be made.")
        sys.exit(0)

    # Bind alias to the candidate (this replaces previous binding automatically)
    client.set_registered_model_alias(name=model_name, alias=alias, version=str(mv.version))

    # Tag bookkeeping
    client.set_model_version_tag(name=model_name, version=mv.version, key="promoted_by", value=promoted_by)
    client.set_model_version_tag(name=model_name, version=mv.version, key="promote_reason", value=reason)
    if alias == "production":
        client.set_model_version_tag(name=model_name, version=mv.version, key="was_production", value="true")
        if prev:
            # Mark outgoing production
            client.set_model_version_tag(name=model_name, version=prev.version, key="was_production", value="true")

    # Audit
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": "PROMOTE",
        "model": model_name,
        "alias": alias,
        "candidate_version": int(mv.version),
        "previous_alias_version": int(prev.version) if prev else None,
        "reason": reason,
        "promoted_by": promoted_by,
        "tracking_uri": tracking_uri,
        "registry_uri": registry_uri,
    }
    append_audit(entry)
    print("[PROMOTE] Done and audited.")
    sys.exit(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-version", type=int, required=True)
    ap.add_argument("--to", dest="stage", required=True, choices=["Staging", "Production"])
    ap.add_argument("--dry-run", type=str, default="true", help="true|false")
    ap.add_argument("--reason", type=str, default="routine promotion")
    ap.add_argument("--promoted-by", type=str, default="samarth")
    args = ap.parse_args()

    main(
        candidate_version=args.candidate_version,
        stage=args.stage,
        dry_run=str2bool(args.dry_run),
        reason=args.reason,
        promoted_by=args.promoted_by,
    )

