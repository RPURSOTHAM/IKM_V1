from __future__ import annotations

import re
from typing import Any


class PostProcessor:
    """Minimal post-processing pipeline for extracted text."""

    def run(self, text: str, processors: list[dict[str, Any]]) -> str:
        if not text:
            return ""

        output = text
        for proc in processors:
            ptype = str(proc.get("type", "")).lower()
            if ptype == "regex_replace":
                pattern = proc.get("pattern", "")
                replacement = proc.get("replacement", "")
                case_sensitive = str(proc.get("case_sensitive", "false")).lower() == "true"
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    output = re.sub(pattern, replacement, output, flags=flags)
                except re.error:
                    continue
            elif ptype == "lowercase":
                output = output.lower()
            elif ptype == "strip":
                output = output.strip()
        return output
