"""Registry operations.

alias : put an alias on an explicit version            (dev:     -> Challenger)
copy  : load model by alias, log + register as a NEW version, give it new alias
                                                        (preprod: Challenger -> Challenger-pre-test,
                                                         prod:    Challenger-post-test -> Champion)
retag : move a new alias onto the version an alias points to
                                                        (preprod: Challenger-pre-test -> Challenger-post-test)
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from common import ART_DIR, DATA_DIR, DATASETS, client, model_name, version_for_alias


def cmd_alias(a):
    version = a.version if a.version is not None else int(Path(a.version_file).read_text().strip())
    client().set_registered_model_alias(model_name(a.dataset), a.alias, str(version))
    print(f"[promote] {model_name(a.dataset)} v{version} <- alias '{a.alias}'")


def cmd_copy(a):
    name = model_name(a.dataset)
    src_version = version_for_alias(a.dataset, a.src_alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{a.src_alias}")

    sample = pd.read_csv(DATA_DIR / f"{a.dataset}_test.csv").drop(columns="target").head(20)
    signature = infer_signature(sample, model.predict(sample))

    mlflow.set_experiment(f"{a.dataset}-{a.env}")
    with mlflow.start_run(run_name=f"{a.env}: {a.src_alias} -> {a.new_alias}"):
        mlflow.log_param("source_alias", a.src_alias)
        mlflow.log_param("source_version", src_version)
        info = mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                        registered_model_name=name)
    new_version = getattr(info, "registered_model_version", None)
    if new_version is None:  # older mlflow fallback
        new_version = max(int(v.version) for v in client().search_model_versions(f"name='{name}'"))
    new_version = int(new_version)

    client().set_registered_model_alias(name, a.new_alias, str(new_version))
    (ART_DIR / f"{a.dataset}_{a.env}_version.txt").write_text(str(new_version))
    print(f"[promote] {name} v{src_version} ('{a.src_alias}') re-registered as v{new_version} "
          f"<- alias '{a.new_alias}'")


def cmd_retag(a):
    version = version_for_alias(a.dataset, a.src_alias)
    client().set_registered_model_alias(model_name(a.dataset), a.new_alias, str(version))
    print(f"[promote] {model_name(a.dataset)} v{version} ('{a.src_alias}') <- alias '{a.new_alias}'")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("alias")
    s.add_argument("--dataset", required=True, choices=DATASETS)
    s.add_argument("--alias", required=True)
    s.add_argument("--version", type=int)
    s.add_argument("--version-file")
    s.set_defaults(fn=cmd_alias)

    s = sub.add_parser("copy")
    s.add_argument("--dataset", required=True, choices=DATASETS)
    s.add_argument("--env", required=True)
    s.add_argument("--src-alias", required=True)
    s.add_argument("--new-alias", required=True)
    s.set_defaults(fn=cmd_copy)

    s = sub.add_parser("retag")
    s.add_argument("--dataset", required=True, choices=DATASETS)
    s.add_argument("--src-alias", required=True)
    s.add_argument("--new-alias", required=True)
    s.set_defaults(fn=cmd_retag)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
