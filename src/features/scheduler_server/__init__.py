"""Scheduler Server Feature Package.

Provides an end-to-end scheduling daemon for managing document processor container pools,
ingesting jobs from RabbitMQ, maintaining state machine lifecycles in PostgreSQL/MySQL,
monitoring container health, auto-healing, auto-scaling worker capacity, and servicing
parent API control requests over TCP.
"""

__version__ = "2.0.0"
