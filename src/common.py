"""Shared helpers for all pipeline scripts."""
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

DATASETS = ("disease", "boston")


def model_name(dataset: str) -> str:
    return f"{dataset}-classifier"


def client() -> MlflowClient:
    return MlflowClient(tracking_uri=TRACKING_URI)


def version_for_alias(dataset: str, alias: str) -> int:
    mv = client().get_model_version_by_alias(model_name(dataset), alias)
    return int(mv.version)
