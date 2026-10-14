"""Train Random Forest + Logistic Regression, log to MLflow, register the best one."""
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ART_DIR, DATA_DIR, DATASETS, model_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--env", default="dev")
    args = p.parse_args()

    train = pd.read_csv(DATA_DIR / f"{args.dataset}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{args.dataset}_test.csv")
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    candidates = {
        "random_forest": (
            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
            {"n_estimators": 200, "max_depth": 8},
        ),
        "logistic_regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            {"C": 1.0, "max_iter": 2000},
        ),
    }

    mlflow.set_experiment(f"{args.dataset}-{args.env}")
    best = {"acc": -1.0, "run_id": None, "name": None}

    for name, (model, params) in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", args.dataset)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=infer_signature(X_train, model.predict(X_train)),
                input_example=X_train.head(3),
            )
            print(f"[train] {name}: {metrics}")
            if metrics["accuracy"] > best["acc"]:
                best = {"acc": metrics["accuracy"], "run_id": run.info.run_id, "name": name}

    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", model_name(args.dataset))
    mlflow.tracking.MlflowClient().set_model_version_tag(
        model_name(args.dataset), mv.version, "model_type", best["name"]
    )
    (ART_DIR / f"{args.dataset}_version.txt").write_text(str(mv.version))
    print(f"[train] best={best['name']} acc={best['acc']:.4f} -> registered "
          f"{model_name(args.dataset)} v{mv.version}")


if __name__ == "__main__":
    main()
