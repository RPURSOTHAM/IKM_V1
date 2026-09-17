#!/usr/bin/env python3
"""Download a sentence-transformers model into the mounted models host path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_models_root = os.getenv("SCHEDULER_MODELS_HOST_PATH") or str(repo_root / "src" / "models")

    parser = argparse.ArgumentParser(description="Download embedding model for offline processor/DMS use.")
    parser.add_argument(
        "--model-id",
        default="BAAI/bge-base-en",
        help="Hugging Face model id (default: BAAI/bge-base-en)",
    )
    parser.add_argument(
        "--local-dir",
        default="bge-base-en",
        help="Folder name under the mounted models root (default: bge-base-en)",
    )
    parser.add_argument(
        "--models-root",
        default=default_models_root,
        help=(
            "Host path mounted into processors as the models volume "
            "(default: SCHEDULER_MODELS_HOST_PATH or repo src/models)"
        ),
    )
    args = parser.parse_args()

    target = Path(args.models_root).expanduser().resolve() / args.local_dir
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print("Install huggingface_hub first: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Downloading {args.model_id} -> {target}")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=[
            "onnx/*",
            "openvino/*",
            "*.onnx",
            "*openvino*",
            "tf_model.h5",
            "rust_model.ot",
            "flax_model.msgpack",
        ],
    )
    print(f"Saved model to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
