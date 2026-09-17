from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
CONSUMER_API_ROOT = REPO_ROOT / "src" / "dms_service" / "consumer_api_service"

for candidate in (REPO_ROOT / ".env", CONSUMER_API_ROOT / ".env"):
    load_dotenv(candidate, override=False)


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "document_id": args.document_id,
        "document_name": args.document_name,
        "collection_name": args.collection_name,
        "tenant_id": args.tenant_id,
    }


def _publish(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    try:
        import pika
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pika'. Install with: pip install -e src/dms_service") from exc

    credentials = pika.PlainCredentials(args.rabbitmq_user, args.rabbitmq_pass)
    parameters = pika.ConnectionParameters(
        host=args.rabbitmq_host,
        port=args.rabbitmq_port,
        credentials=credentials,
    )
    connection = pika.BlockingConnection(parameters)
    try:
        channel = connection.channel()
        channel.queue_declare(queue=args.queue_name, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=args.queue_name,
            body=json.dumps(payload).encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post a scheduler-compatible document processing message to RabbitMQ."
    )
    parser.add_argument("--document-id", required=True, help="Document UUID/job id.")
    parser.add_argument("--document-name", required=True, help="Stored document name resolvable by processors.")
    parser.add_argument(
        "--collection-name",
        default=os.getenv("WEAVIATE_COLLECTION", "DocumentChunk"),
        help="Target collection name. Defaults to WEAVIATE_COLLECTION or DocumentChunk.",
    )
    parser.add_argument(
        "--tenant-id",
        default=os.getenv("DEFAULT_TENANT_ID"),
        help="Tenant id. Defaults to DEFAULT_TENANT_ID when set.",
    )
    parser.add_argument("--rabbitmq-host", default=os.getenv("RABBITMQ_HOST", "localhost"))
    parser.add_argument("--rabbitmq-port", type=int, default=int(os.getenv("RABBITMQ_PORT", "5672")))
    parser.add_argument("--rabbitmq-user", default=os.getenv("RABBITMQ_USER", "rabbitmq_user"))
    parser.add_argument("--rabbitmq-pass", default=os.getenv("RABBITMQ_PASS", "rabbitmq_password"))
    parser.add_argument("--queue-name", default=os.getenv("RABBITMQ_QUEUE_NAME", "document_processing_queue"))
    parser.add_argument("--dry-run", action="store_true", help="Print payload without publishing.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = _build_payload(args)

    print(json.dumps(payload, indent=2))
    if args.dry_run:
        return 0

    try:
        _publish(args, payload)
    except Exception as exc:
        print(f"Failed to publish message: {exc}", file=sys.stderr)
        return 1

    print(f"Published message to queue '{args.queue_name}' on {args.rabbitmq_host}:{args.rabbitmq_port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
