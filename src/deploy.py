"""Deploy a registered model with `mlflow models serve` (REST endpoint on a port).

Pass either --alias (e.g. Champion) or --version / --version-file.
The server is started in its own process group so it keeps running after the
Jenkins build finishes; the next deploy to the same port stops the old one.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import DATASETS, TRACKING_URI, model_name, version_for_alias

SERVE_DIR = Path(os.getenv("SERVE_DIR", Path.home() / "mlops-serving"))
SERVE_DIR.mkdir(parents=True, exist_ok=True)


def stop_old(port: int) -> None:
    pid_file = SERVE_DIR / f"{port}.pid"
    if pid_file.exists():
        try:
            os.killpg(int(pid_file.read_text().strip()), signal.SIGTERM)
            print(f"[deploy] stopped old server on port {port}")
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pid_file.unlink(missing_ok=True)
        time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--alias")
    p.add_argument("--version", type=int)
    p.add_argument("--version-file")
    p.add_argument("--timeout", type=int, default=180)
    a = p.parse_args()

    name = model_name(a.dataset)
    if a.alias:
        version = version_for_alias(a.dataset, a.alias)
    elif a.version is not None:
        version = a.version
    elif a.version_file:
        version = int(Path(a.version_file).read_text().strip())
    else:
        sys.exit("give --alias or --version or --version-file")

    uri = f"models:/{name}/{version}"
    stop_old(a.port)

    log_path = SERVE_DIR / f"{a.dataset}-{a.port}.log"
    env = {**os.environ, "MLFLOW_TRACKING_URI": TRACKING_URI}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "models", "serve", "-m", uri,
             "-p", str(a.port), "-h", "127.0.0.1", "--env-manager", "local"],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
    (SERVE_DIR / f"{a.port}.pid").write_text(str(proc.pid))
    print(f"[deploy] serving {uri} on port {a.port} (pid {proc.pid}, log {log_path})")

    deadline = time.time() + a.timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(log_path.read_text()[-3000:])
            sys.exit("[deploy] server exited early")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/ping", timeout=3):
                print("[deploy] endpoint is healthy")
                return
        except Exception:
            time.sleep(3)
    sys.exit("[deploy] timeout waiting for endpoint")


if __name__ == "__main__":
    main()
