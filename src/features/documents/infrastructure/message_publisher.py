from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pika

logger = logging.getLogger(__name__)


@dataclass
class QueuePublisher:
    host: str
    port: int
    user: str
    password: str
    queue_name: str

    def ping(self) -> bool:
        """Return True when the broker accepts an AMQP connection (no message published)."""
        try:
            credentials = pika.PlainCredentials(self.user, self.password)
            parameters = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials)
            connection = pika.BlockingConnection(parameters)
            try:
                connection.close()
            except Exception:
                pass
            return True
        except Exception:
            return False

    def publish_document_job(self, payload: dict[str, Any]) -> None:
        document_id = str(payload.get("document_id") or "")
        processor_type = str(payload.get("processor_type") or "")
        logger.info(
            "PIPELINE_TRACE %s",
            {
                "stage": "rabbitmq_publish_open",
                "document_id": document_id,
                "processor_type": processor_type,
                "queue_name": self.queue_name,
                "host": self.host,
                "port": self.port,
            },
        )
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        try:
            channel = connection.channel()
            channel.queue_declare(queue=self.queue_name, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=json.dumps(payload).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
            )
            logger.info(
                "PIPELINE_TRACE %s",
                {
                    "stage": "rabbitmq_publish_written",
                    "document_id": document_id,
                    "processor_type": processor_type,
                    "queue_name": self.queue_name,
                    "exchange": "",
                    "routing_key": self.queue_name,
                },
            )
        finally:
            connection.close()

    def publish_json_to_queue(
        self,
        queue_name: str,
        payload: dict[str, Any],
        *,
        virtual_host: str = "/",
    ) -> None:
        """Publish JSON to a named queue via the default exchange (routing_key = queue name)."""
        credentials = pika.PlainCredentials(self.user, self.password)
        vhost = "/" if virtual_host in ("", "default") else virtual_host
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            virtual_host=vhost,
        )
        connection = None
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            try:
                channel.queue_declare(queue=queue_name, durable=True, passive=True)
            except Exception:
                channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=json.dumps(payload).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
            )
        finally:
            if connection is not None:
                connection.close()

