import os, json, numpy as np
from typing import List
from collections import deque

REF = "outputs/reference_bins.json"
REQS = "logs/requests.jsonl"
OUT = "outputs/drift_report.json"

def read_recent_rows(max_rows: int = 500) -> np.ndarray:
    if not os.path.exists(REQS):
        return np.empty((0,0))
    rows: List[List[float]] = []
    # read last ~max_rows payloads (cheap)
    with open(REQS, "r") as f:
        dq = deque(f, maxlen=1000)  # cap file memory
    for line in dq:
        try:
            d = json.loads(line)
            rows.extend(d.get("rows", []))
        except Exception:
            pass
    if not rows:
        return np.empty((0,0))
    X = np.array(rows, dtype=float)
    if X.shape[0] > max_rows:
        X = X[-max_rows:]
    return X

def psi(ref_p, cur_p, eps=1e-6):
    ref_p = np.asarray(ref_p, float) + eps
    cur_p = np.asarray(cur_p, float) + eps
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

def main():
    if not os.path.exists(REF):
        raise SystemExit("[ERR] build reference first: python src/build_reference.py")
    with open(REF, "r") as f:
        ref = json.load(f)
    X = read_recent_rows()
    if X.size == 0:
        print("[DRIFT] No recent requests to evaluate.")
        report = {"status": "no_data", "avg_psi": None, "features": []}
        os.makedirs("outputs", exist_ok=True)
        with open(OUT, "w") as f: json.dump(report, f, indent=2)
        return

    n_feats = int(ref["n_features"])
    if X.shape[1] != n_feats:
        print(f"[DRIFT] Shape mismatch: expected {n_feats}, got {X.shape[1]}")
        raise SystemExit(2)

    feat_reports = []
    psis = []
    for c in range(n_feats):
        edges = np.array(ref["features"][c]["edges"])
        ref_p = np.array(ref["features"][c]["ref_p"])
        # compute current histogram on same edges
        counts, _ = np.histogram(X[:, c], bins=edges)
        cur_p = counts / counts.sum() if counts.sum() > 0 else counts
        v = psi(ref_p, cur_p)
        psis.append(v)
        feat_reports.append({"feature": c, "psi": round(v, 4)})

    avg_psi = float(np.mean(psis))
    status = "ok"
    severity = "none"
    if avg_psi >= 0.2 or max(psis) >= 0.2:
        status, severity = "drift", "high"
    elif avg_psi >= 0.1 or max(psis) >= 0.1:
        status, severity = "drift", "moderate"

    report = {"status": status, "severity": severity, "avg_psi": round(avg_psi, 4),
              "max_psi": round(float(np.max(psis)), 4), "features": feat_reports}
    os.makedirs("outputs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(report, f, indent=2)
    print("[DRIFT]", json.dumps(report))
    # non-zero exit on drift so CI/alerts can hook into it
    if status == "drift": raise SystemExit(3)

if __name__ == "__main__":
    main()

