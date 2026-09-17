#!/usr/bin/env python3
"""Download the CrossEncoder reranker into src/models for offline Docker use.

Usage (online machine):
  python scripts/download_reranker_model.py

Then ensure Docker mounts REPO_ROOT/src/models at /app/src/models and set:
  RERANKER_MODEL=/app/src/models/bge-reranker-base
  TRANSFORMERS_OFFLINE=1
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "src" / "models" / "bge-reranker-base"
DEFAULT_HUB_ID = "BAAI/bge-reranker-base"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_HUB_ID)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true", help="Replace existing destination directory")
    args = parser.parse_args()

    dest: Path = args.dest
    if dest.exists() and any(dest.iterdir()):
        if not args.force:
            print(f"Destination already exists: {dest}")
            print("Use --force to replace.")
            return 0
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required. Install with: pip install huggingface_hub", file=sys.stderr)
        return 1

    print(f"Downloading {args.model_id} -> {dest} ...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    # Also create HF-cache-style alias for resolver discovery.
    alias = dest.parent / "models--BAAI--bge-reranker-base"
    if not alias.exists():
        try:
            alias.symlink_to(dest.name, target_is_directory=True)
        except OSError:
            # On Windows without symlink privileges, copy is too heavy; resolver
            # already checks bge-reranker-base directly.
            pass

    required = ["config.json"]
    missing = [name for name in required if not (dest / name).exists()]
    if missing:
        print(f"Download incomplete; missing: {missing}", file=sys.stderr)
        return 1

    print("Done.")
    print("Docker env:")
    print("  RERANKER_MODEL=/app/src/models/bge-reranker-base")
    print("  TRANSFORMERS_OFFLINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
