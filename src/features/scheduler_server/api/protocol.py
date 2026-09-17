from __future__ import annotations

import json
import struct
from typing import Any, Dict


def encode_frame(payload: Dict[str, Any]) -> bytes:
    """Pack JSON payload into 4-byte big-endian length-prefixed frame."""
    body = json.dumps(payload, default=str).encode("utf-8")
    header = struct.pack(">I", len(body))
    return header + body


def decode_frame(raw_data: bytes) -> Dict[str, Any]:
    """Unpack JSON payload from length-prefixed frame data."""
    if len(raw_data) < 4:
        raise ValueError("Invalid frame: Header length under 4 bytes")
    length = struct.unpack(">I", raw_data[:4])[0]
    body = raw_data[4 : 4 + length]
    return json.loads(body.decode("utf-8"))
