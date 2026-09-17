#!/usr/bin/env python3
"""Kill or graceful shutdown of the Scheduler Server process."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PID_FILE = REPO_ROOT / ".scheduler.pid"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

from src.features.scheduler_server.client.scheduler_client import SchedulerClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Kill or Stop Scheduler Daemon")
    parser.add_argument("--force", action="store_true", help="Force kill without TCP shutdown signal")
    parser.add_argument("--host", default="localhost", help="Scheduler host")
    parser.add_argument("--port", type=int, default=3200, help="Scheduler TCP port")
    parser.add_argument("--timeout", type=int, default=15, help="Shutdown timeout")
    args = parser.parse_args()

    client = SchedulerClient(host=args.host, port=args.port, timeout=float(args.timeout))

    if not args.force:
        print(f"Sending TCP shutdown signal to Scheduler at {args.host}:{args.port}...")
        res = client.shutdown_scheduler()
        print("Shutdown response:", res)

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            print(f"Found PID {pid} in {PID_FILE}")
            if args.force:
                print(f"Terminating PID {pid}...")
                os.kill(pid, signal.SIGTERM if sys.platform != "win32" else signal.SIGINT)
            PID_FILE.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to terminate PID from file: %s", exc)

    print("Scheduler stop complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
