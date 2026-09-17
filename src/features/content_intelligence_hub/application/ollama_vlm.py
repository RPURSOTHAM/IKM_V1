"""Vision summary client for CIH images (Ollama + Gemini fallback).

Updated CIH env:
  OPEN_WEIGHT_VLM_* / OLLAMA_*  — preferred Ollama cloud/local VLM
  GEMINI_API_KEY / GEMINI_MODEL_NAME — fallback when Ollama cannot do images
  CIH_IMAGE_SUMMARY_PROVIDER=auto|ollama|gemini
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "You are a content-intelligence assistant. Summarize this image clearly and "
    "concisely for a business/pharma document pipeline. Describe visible text, "
    "figures, charts, products, people, and any medical or regulatory cues. "
    "If text is readable, quote the important lines. Respond in plain English."
)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def vlm_enabled() -> bool:
    provider = _env("CIH_IMAGE_SUMMARY_PROVIDER", "auto").lower() or "auto"
    if provider == "gemini":
        return bool(_env("GEMINI_API_KEY")) and _env("IMAGE_DESCRIPTION_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if provider == "ollama":
        flag = _env("OPEN_WEIGHT_VLM_ENABLED", "false").lower()
        desc = _env("IMAGE_DESCRIPTION_ENABLED", "true").lower()
        return flag in {"1", "true", "yes", "on"} and desc not in {"0", "false", "no", "off"}
    # auto: either path ok
    ollama_on = _env("OPEN_WEIGHT_VLM_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    gemini_on = bool(_env("GEMINI_API_KEY"))
    desc_ok = _env("IMAGE_DESCRIPTION_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
    return desc_ok and (ollama_on or gemini_on)


def resolve_ollama_config() -> dict[str, Any]:
    api_url = _env("OPEN_WEIGHT_VLM_API_URL")
    host = _env("OLLAMA_HOST")
    if not api_url:
        if host:
            api_url = f"{host.rstrip('/')}/api/chat"
        else:
            api_url = "http://127.0.0.1:11434/api/chat"
    model = _env("OPEN_WEIGHT_VLM_MODEL") or _env("OLLAMA_TRANSLATION_MODEL") or "gemma4:31b"
    api_key = _env("OPEN_WEIGHT_VLM_API_KEY") or _env("OLLAMA_API_KEY")
    try:
        timeout = float(_env("OPEN_WEIGHT_VLM_TIMEOUT_SECONDS", "90") or "90")
    except ValueError:
        timeout = 90.0
    return {
        "api_url": api_url,
        "model": model,
        "api_key": api_key,
        "timeout": timeout,
        "confidence": _env("OPEN_WEIGHT_VLM_CONFIDENCE", "0.85"),
    }


def resolve_gemini_config() -> dict[str, Any]:
    api_key = _env("GEMINI_API_KEY")
    model = _env("GEMINI_MODEL_NAME") or "gemini-2.5-flash"
    try:
        timeout = float(_env("OPEN_WEIGHT_VLM_TIMEOUT_SECONDS", "90") or "90")
    except ValueError:
        timeout = 90.0
    return {"api_key": api_key, "model": model, "timeout": timeout}


def summarize_image_bytes(
    image_bytes: bytes,
    *,
    filename: str | None = None,
    prompt: str | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    if not vlm_enabled():
        raise RuntimeError(
            "Image summary is disabled. Set IMAGE_DESCRIPTION_ENABLED=true and either "
            "OPEN_WEIGHT_VLM_ENABLED=true (Ollama) or GEMINI_API_KEY."
        )
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    mime = mime_type or mimetypes.guess_type(filename or "")[0] or "image/png"
    user_prompt = (prompt or _DEFAULT_PROMPT).strip()
    if filename:
        user_prompt = f"{user_prompt}\n\nFilename: {filename}"

    provider_pref = (_env("CIH_IMAGE_SUMMARY_PROVIDER", "auto") or "auto").lower()
    errors: list[str] = []

    if provider_pref in {"auto", "ollama"} and _env("OPEN_WEIGHT_VLM_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            result = _summarize_ollama(image_bytes, user_prompt=user_prompt, mime=mime, filename=filename)
            return result
        except Exception as exc:
            logger.warning("Ollama image summary failed: %s", exc)
            errors.append(f"ollama: {exc}")
            if provider_pref == "ollama":
                raise RuntimeError("; ".join(errors)) from exc

    if provider_pref in {"auto", "gemini"} and _env("GEMINI_API_KEY"):
        try:
            return _summarize_gemini(image_bytes, user_prompt=user_prompt, mime=mime, filename=filename)
        except Exception as exc:
            logger.warning("Gemini image summary failed: %s", exc)
            errors.append(f"gemini: {exc}")
            if provider_pref == "gemini":
                raise RuntimeError("; ".join(errors)) from exc

    raise RuntimeError(
        "Image summary failed. "
        + ("; ".join(errors) if errors else "No vision provider configured.")
        + " Tip: gemma3:27b is retired on ollama.com — set OPEN_WEIGHT_VLM_MODEL to a "
        "vision-capable model you can access, or rely on GEMINI_MODEL_NAME fallback."
    )


def _summarize_ollama(
    image_bytes: bytes,
    *,
    user_prompt: str,
    mime: str,
    filename: str | None,
) -> dict[str, Any]:
    cfg = resolve_ollama_config()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
                "images": [b64],
            }
        ],
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    with httpx.Client(timeout=cfg["timeout"]) as client:
        response = client.post(cfg["api_url"], json=payload, headers=headers)
        if response.status_code >= 400:
            body = (response.text or "")[:500]
            raise RuntimeError(f"HTTP {response.status_code} at {cfg['api_url']}: {body}")
        data = response.json()

    summary = _extract_message_text(data)
    if not summary:
        raise RuntimeError("Ollama returned an empty summary.")
    return {
        "available": True,
        "provider": "ollama",
        "model": cfg["model"],
        "api_url": cfg["api_url"],
        "filename": filename,
        "mime_type": mime,
        "summary": summary,
        "confidence": cfg["confidence"],
    }


def _summarize_gemini(
    image_bytes: bytes,
    *,
    user_prompt: str,
    mime: str,
    filename: str | None,
) -> dict[str, Any]:
    cfg = resolve_gemini_config()
    if not cfg["api_key"]:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    b64 = base64.b64encode(image_bytes).decode("ascii")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg['model']}:generateContent?key={cfg['api_key']}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]
            }
        ]
    }
    with httpx.Client(timeout=cfg["timeout"]) as client:
        response = client.post(url, json=payload)
        if response.status_code >= 400:
            body = (response.text or "")[:500]
            raise RuntimeError(f"HTTP {response.status_code} from Gemini: {body}")
        data = response.json()

    summary = _extract_gemini_text(data)
    if not summary:
        raise RuntimeError("Gemini returned an empty summary.")
    return {
        "available": True,
        "provider": "gemini",
        "model": cfg["model"],
        "api_url": "https://generativelanguage.googleapis.com/v1beta",
        "filename": filename,
        "mime_type": mime,
        "summary": summary,
        "confidence": _env("OPEN_WEIGHT_VLM_CONFIDENCE", "0.85"),
    }


def _extract_message_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data or "").strip()
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            joined = "\n".join(p for p in parts if p.strip()).strip()
            if joined:
                return joined
    for key in ("response", "output_text", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_gemini_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for candidate in data.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        texts = [str(p.get("text") or "").strip() for p in parts if isinstance(p, dict)]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return ""
