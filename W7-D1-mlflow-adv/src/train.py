import os, json, argparse, yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score

def load_cfg():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main(seed: int):
    cfg = load_cfg()
    os.environ.setdefault("MLFLOW_TRACKING_URI", os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(cfg["experiment_name"])

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=seed, stratify=y
    )

    # Scale features + give the solver more iterations to avoid ConvergenceWarning
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs", n_jobs=None)
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        # Log metrics and params
        mlflow.log_params({
            "random_state": seed,
            "model_type": "logistic_regression",
            "pipeline": "StandardScaler->LogisticRegression",
            "max_iter": 1000,
            "solver": "lbfgs",
            "scikit_learn": sklearn.__version__,
        })
        mlflow.log_metrics({"auc": float(auc), "f1": float(f1)})

        # Save a metrics.json artifact (handy for audits)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"auc": float(auc), "f1": float(f1)}, f, indent=2)
        mlflow.log_artifact("outputs/metrics.json", artifact_path="eval")

        # Use 'name' instead of deprecated 'artifact_path'
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=cfg["registered_model_name"],
            input_example=X_test[:5],
            signature=None
        )

        # Tag run for traceability
        mlflow.set_tags({
            "data.name": "sklearn_breast_cancer",
            "purpose": "W7D1-mlflow-adv",
            "owner": "samarth",
            "candidate": "true"
        })

        print(f"[OK] Run {run.info.run_id} logged. AUC={auc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    main(args.seed)

