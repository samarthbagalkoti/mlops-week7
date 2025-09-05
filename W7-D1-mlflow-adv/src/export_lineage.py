import os, json, yaml
import mlflow
from mlflow.tracking import MlflowClient

def load_yaml(p):
    with open(p,"r") as f: 
        import yaml; return yaml.safe_load(f)

def main():
    os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI","http://54.147.138.39:8081"))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    c = MlflowClient()
    name = load_yaml("params.yaml")["registered_model_name"]
    versions = sorted(c.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))

    data = []
    for v in versions:
        run = c.get_run(v.run_id)
        entry = {
            "version": int(v.version),
            "stage": v.current_stage or "None",
            "run_id": v.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": v.tags
        }
        data.append(entry)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/registry.json","w") as f:
        json.dump({"model": name, "versions": data}, f, indent=2)

    # Mermaid graph: versions with stage
    lines = ["flowchart LR"]
    for e in data:
        label = f"v{e['version']}\\nstage:{e['stage']}"
        lines.append(f"  V{e['version']}[{label}]")
        lines.append(f"  RUN_{e['version']}((run))")
        lines.append(f"  RUN_{e['version']} --> V{e['version']}")
    # alias nodes
    for alias in ["production","staging","latest-candidate"]:
        try:
            mv = c.get_model_version_by_alias(name, alias)
            lines.append(f"  A_{alias}{{{alias}}}")
            lines.append(f"  A_{alias} --> V{mv.version}")
        except Exception:
            pass

    with open("outputs/lineage.mmd","w") as f:
        f.write("\n".join(lines))

    print("[OK] Wrote outputs/registry.json and outputs/lineage.mmd")

if __name__ == "__main__":
    main()

