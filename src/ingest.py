"""Data ingest: load dataset, build binary target, save deterministic train/test CSVs.

disease -> sklearn breast-cancer dataset (offline, binary: malignant/benign)
boston  -> Boston house prices (OpenML). Target = 1 if MEDV > median else 0,
           so Random Forest / Logistic Regression *classifiers* can be used.
           (sklearn removed load_boston, so we fetch from OpenML. If OpenML is
           unreachable we fall back to California housing, same treatment.)
"""
import argparse

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

from common import DATA_DIR, DATASETS


def load(dataset: str) -> pd.DataFrame:
    if dataset == "disease":
        df = load_breast_cancer(as_frame=True).frame.copy()
        df["target"] = df["target"].astype(int)
        return df

    if dataset == "boston":
        try:
            df = fetch_openml(name="boston", version=1, as_frame=True, parser="auto").frame.copy()
            price_col = "MEDV"
        except Exception as exc:  # network / OpenML problem
            print(f"[ingest] OpenML boston failed ({exc}); falling back to California housing")
            df = fetch_california_housing(as_frame=True).frame.copy()
            price_col = "MedHouseVal"
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str))
        df["target"] = (df[price_col] > df[price_col].median()).astype(int)
        return df.drop(columns=[price_col])

    raise SystemExit(f"unknown dataset {dataset}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    args = p.parse_args()

    df = load(args.dataset)
    feature_cols = [c for c in df.columns if c != "target"]
    df[feature_cols] = df[feature_cols].astype("float64")

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    train.to_csv(DATA_DIR / f"{args.dataset}_train.csv", index=False)
    test.to_csv(DATA_DIR / f"{args.dataset}_test.csv", index=False)
    print(f"[ingest] {args.dataset}: train={train.shape} test={test.shape}")


if __name__ == "__main__":
    main()
