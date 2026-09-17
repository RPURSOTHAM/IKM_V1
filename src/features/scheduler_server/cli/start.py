#!/usr/bin/env python3
"""Start the Scheduler Server background process."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PID_FILE = REPO_ROOT / ".scheduler.pid"


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Scheduler Daemon")
    parser.add_argument("--foreground", action="store_true", help="Run daemon in foreground")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for health readiness")
    parser.add_argument("--timeout", type=int, default=60, help="Readiness timeout in seconds")
    args = parser.parse_args()

    if PID_FILE.exists():
        print(f"PID file exists at {PID_FILE}. Scheduler may already be running.")

    cmd = [sys.executable, "-m", "src.features.scheduler_server.main"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    if args.foreground:
        print("Starting Scheduler Daemon in foreground...")
        return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)

    print("Spawning Scheduler Daemon in background...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"Started Scheduler Daemon (PID: {proc.pid}). PID file written to {PID_FILE}")

    if args.no_wait:
        return 0

    print(f"Waiting for Scheduler readiness (timeout: {args.timeout}s)...")
    from src.features.scheduler_server.client.scheduler_client import SchedulerClient

    client = SchedulerClient(timeout=3.0)
    start_time = time.time()
    while time.time() - start_time < args.timeout:
        res = client.healthcheck()
        if res.get("status") == "Healthy":
            print("Scheduler is Healthy and ready!")
            return 0
        time.sleep(2)

    print("Timeout waiting for Scheduler readiness.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
