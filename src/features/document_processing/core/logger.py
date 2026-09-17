import logging
import os

level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=level,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("src.processor_service")
log.setLevel(level)
