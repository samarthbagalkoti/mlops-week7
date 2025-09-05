import os, json, numpy as np
from sklearn.datasets import load_breast_cancer

os.makedirs("outputs", exist_ok=True)
data = load_breast_cancer()
X = data.data  # (n, 30)

ref = {"n_features": X.shape[1], "bins": 10, "features": []}
for c in range(X.shape[1]):
    col = X[:, c]
    edges = np.quantile(col, np.linspace(0, 1, 11))  # 10 bins → 11 edges
    # to avoid duplicate edges if constant segments
    edges = np.unique(edges)
    # histogram on these edges
    counts, edges_used = np.histogram(col, bins=edges)
    # normalize
    p = counts / counts.sum() if counts.sum() > 0 else counts
    ref["features"].append({"edges": edges_used.tolist(), "ref_p": p.tolist()})

with open("outputs/reference_bins.json", "w") as f:
    json.dump(ref, f, indent=2)
print("[OK] wrote outputs/reference_bins.json")

