#!/bin/bash
# Run dev -> preprod -> prod flow locally (no Jenkins) to verify everything works.
# Usage: bash scripts/local_demo.sh [disease|boston]
set -e
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}
DS=${1:-disease}
TH=${THRESHOLD:-0.85}
OFF=$([ "$DS" = boston ] && echo 1 || echo 0)

echo "===== DEV ====="
python src/ingest.py --dataset $DS
python src/train.py  --dataset $DS --env dev
python src/deploy.py --dataset $DS --port $((5100+OFF)) --version-file artifacts/${DS}_version.txt
python src/test_model.py --dataset $DS --port $((5100+OFF)) --threshold $TH
python src/promote.py alias --dataset $DS --alias Challenger --version-file artifacts/${DS}_version.txt

echo "===== PRE-PROD ====="
python src/promote.py copy --dataset $DS --env preprod --src-alias Challenger --new-alias Challenger-pre-test
python src/deploy.py --dataset $DS --port $((5200+OFF)) --alias Challenger-pre-test
python src/test_model.py --dataset $DS --port $((5200+OFF)) --threshold $TH
python src/promote.py retag --dataset $DS --src-alias Challenger-pre-test --new-alias Challenger-post-test

echo "===== PROD ====="
python src/promote.py copy --dataset $DS --env prod --src-alias Challenger-post-test --new-alias Champion
python src/deploy.py --dataset $DS --port $((5300+OFF)) --alias Champion
echo "DONE. Prod endpoint: http://127.0.0.1:$((5300+OFF))/invocations"
