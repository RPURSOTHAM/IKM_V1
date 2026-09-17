"""Download Systran/faster-whisper-base into models/ for offline CIH STT."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "faster-whisper-base"
TARGET.mkdir(parents=True, exist_ok=True)
path = snapshot_download("Systran/faster-whisper-base", local_dir=str(TARGET))
print("downloaded", path)
