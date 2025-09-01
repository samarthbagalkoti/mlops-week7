import os, argparse, json, sys, yaml
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


def last_prev_from_audit(model_name: str):
    """Return the most recent previous_alias_version from audit for production, if any."""
    if not os.path.exists(AUDIT_PATH):
        return None
    try:
        with open(AUDIT_PATH, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("action") == "PROMOTE" and ev.get("model") == model_name and ev.get("alias") == "production":
                prev_ver = ev.get("previous_alias_version")
                if prev_ver is not None:
                    return str(prev_ver)
    except Exception:
        pass
    return None


def main(reason: str, dry_run: bool, requested_by: str):
    # URIs (default to local if not set)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    os.environ.setdefault("MLFLOW_REGISTRY_URI", registry_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient()

    model_name = load_yaml("params.yaml")["registered_model_name"]

    # Current production holder via alias (no deprecated stages)
    try:
        current_prod = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        current_prod = None

    if not current_prod:
        msg = "[ROLLBACK] No current 'production' alias. Nothing to rollback."
        print(msg)
        # Dry-run should not fail CI
        sys.exit(0 if dry_run else 1)

    # --- Find rollback target ---
    target = None

    # 1) From audit log (most reliable)
    prev_ver = last_prev_from_audit(model_name)
    if prev_ver and str(prev_ver) != str(current_prod.version):
        try:
            target = client.get_model_version(name=model_name, version=str(prev_ver))
        except Exception:
            target = None  # continue to next strategy

    # 2) From tags: was_production=true (excluding current)
    if target is None:
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        prior = []
        for v in versions:
            if str(v.version) == str(current_prod.version):
                continue
            mv = client.get_model_version(name=model_name, version=str(v.version))
            tags = getattr(mv, "tags", {}) or {}
            if str(tags.get("was_production", "")).strip().lower() == "true":
                prior.append(mv)
        if prior:
            target = sorted(prior, key=lambda x: int(x.version))[-1]

    if target is None:
        msg = "[ROLLBACK] No prior production candidate found (audit/tags)."
        print(msg)
        # Don't fail dry-run checks; fail only for real rollback
        sys.exit(0 if dry_run else 1)

    print(f"[ROLLBACK] Will set alias 'production' -> v{target.version} (replacing v{current_prod.version}). dry_run={dry_run}")

    if dry_run:
        print("[ROLLBACK] Dry-run: no changes will be made.")
        sys.exit(0)

    # Re-point alias to target (non-deprecated)
    try:
        client.set_registered_model_alias(name=model_name, alias="production", version=str(target.version))

        # Optional bookkeeping tags
        client.set_model_version_tag(name=model_name, version=target.version, key="rollback_to", value=str(target.version))
        client.set_model_version_tag(name=model_name, version=target.version, key="rollback_reason", value=reason)
        client.set_model_version_tag(name=model_name, version=current_prod.version, key="rollback_from", value=str(current_prod.version))

        # Audit
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "ROLLBACK",
            "model": model_name,
            "from_version": int(current_prod.version),
            "to_version": int(target.version),
            "alias": "production",
            "reason": reason,
            "requested_by": requested_by,
            "tracking_uri": tracking_uri,
            "registry_uri": registry_uri,
        }
        append_audit(entry)
        print("[ROLLBACK] Completed and audited.")
        sys.exit(0)

    except Exception as e:
        print(f"[ROLLBACK] ERROR during alias re-point/tagging/audit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # accept 1+ tokens for reason, join to a single string
    ap.add_argument("--reason", nargs="+", default=["incident/drift"])
    ap.add_argument("--dry-run", type=str, default="true")  # true|false
    ap.add_argument("--requested-by", type=str, default="samarth")
    args = ap.parse_args()

    reason_str = " ".join(args.reason) if isinstance(args.reason, list) else args.reason
    main(
        reason=reason_str,
        dry_run=str2bool(args.dry_run),
        requested_by=args.requested_by,
    )

