#!/usr/bin/env python3
"""Scheduler Control CLI Tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.scheduler_server.client.scheduler_client import SchedulerClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheduler Server Control CLI")
    parser.add_argument("--host", default="localhost", help="Scheduler host (default: localhost)")
    parser.add_argument("--port", type=int, default=3200, help="Scheduler TCP port (default: 3200)")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    subparsers.add_parser("healthcheck", help="Check scheduler health status")
    subparsers.add_parser("running-jobs", help="List active running jobs")
    subparsers.add_parser("shutdown", help="Initiate graceful scheduler shutdown")

    query_p = subparsers.add_parser("query-job", help="Query job status")
    query_p.add_argument("job_id", help="Document ID or Job ID")

    kill_p = subparsers.add_parser("kill-job", help="Kill active document job")
    kill_p.add_argument("job_id", help="Document ID or Job ID")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    client = SchedulerClient(host=args.host, port=args.port)

    if args.command == "healthcheck":
        res = client.healthcheck()
    elif args.command == "running-jobs":
        res = client.get_running_jobs()
    elif args.command == "query-job":
        res = client.query_job_status(document_id=args.job_id, job_id=args.job_id)
    elif args.command == "kill-job":
        res = client.kill_job(document_id=args.job_id, job_id=args.job_id)
    elif args.command == "shutdown":
        res = client.shutdown_scheduler()
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
