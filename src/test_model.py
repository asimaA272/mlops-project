"""Model test: call the deployed REST endpoint and check accuracy on the hold-out set."""
import argparse
import json
import sys
import urllib.request

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score

from common import DATA_DIR, DATASETS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--threshold", type=float, default=0.85)
    a = p.parse_args()

    test = pd.read_csv(DATA_DIR / f"{a.dataset}_test.csv")
    X, y = test.drop(columns="target"), test["target"]

    payload = {"dataframe_split": {"columns": list(X.columns), "data": X.values.tolist()}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{a.port}/invocations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        preds = json.load(resp)["predictions"]

    acc = accuracy_score(y, preds)
    mlflow.set_experiment(f"{a.dataset}-endpoint-tests")
    with mlflow.start_run(run_name=f"port-{a.port}"):
        mlflow.log_metric("endpoint_accuracy", acc)
        mlflow.log_param("threshold", a.threshold)

    print(f"[test] {a.dataset} port={a.port} accuracy={acc:.4f} threshold={a.threshold}")
    if acc < a.threshold:
        sys.exit("[test] FAILED: accuracy below threshold")
    print("[test] PASSED")


if __name__ == "__main__":
    main()
