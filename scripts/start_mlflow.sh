#!/bin/bash
# Start the MLflow tracking server + model registry (run inside WSL Ubuntu).
set -e
mkdir -p ~/mlflow-server && cd ~/mlflow-server
if [ ! -d venv ]; then
  python3 -m venv venv
  . venv/bin/activate
  pip install "mlflow>=2.14,<3" "sqlalchemy>=2.0,<2.1" "starlette<1.0"
else
  . venv/bin/activate
fi
exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///$HOME/mlflow-server/mlflow.db \
  --artifacts-destination $HOME/mlflow-server/mlartifacts
