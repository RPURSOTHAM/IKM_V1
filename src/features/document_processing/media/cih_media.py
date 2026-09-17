"""Audio/video helpers for Content Intelligence Hub transcription."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv"}
MEDIA_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES
PPT_SUFFIXES = {".pptx", ".ppt"}

TRANSCRIPT_JSON_SUFFIX = ".cih_transcript.json"
TRANSCRIPT_TXT_SUFFIX = ".cih_transcript.txt"


def is_media_suffix(suffix: str) -> bool:
    return str(suffix or "").lower() in MEDIA_SUFFIXES


def is_ppt_suffix(suffix: str) -> bool:
    return str(suffix or "").lower() in PPT_SUFFIXES


def transcript_sidecar_paths(document_path: Path) -> tuple[Path, Path]:
    return (
        document_path.with_suffix(document_path.suffix + TRANSCRIPT_JSON_SUFFIX),
        document_path.with_suffix(document_path.suffix + TRANSCRIPT_TXT_SUFFIX),
    )


def write_transcript_sidecars(document_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path, txt_path = transcript_sidecar_paths(document_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    full_text = str(payload.get("full_text") or "").strip()
    if not full_text:
        segments = payload.get("segments") or []
        parts = [str(seg.get("text") or "").strip() for seg in segments if isinstance(seg, dict)]
        full_text = "\n".join(part for part in parts if part)
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    return json_path, txt_path


def load_transcript_sidecar(document_path: Path) -> dict[str, Any] | None:
    json_path, _txt_path = transcript_sidecar_paths(document_path)
    if not json_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed reading transcript sidecar %s", json_path, exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_ffmpeg_exe() -> str:
    """Prefer system ffmpeg; fall back to imageio-ffmpeg's bundled binary."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return str(bundled)
    except Exception:
        logger.debug("imageio-ffmpeg unavailable", exc_info=True)
    raise RuntimeError(
        "ffmpeg is not available. Install system ffmpeg or pip install imageio-ffmpeg "
        "(required for video transcription)."
    )


def extract_audio_wav(source_path: Path, *, work_dir: Path | None = None) -> Path:
    """Extract mono 16kHz WAV via ffmpeg (required for video; optional normalize for audio)."""
    ffmpeg = _resolve_ffmpeg_exe()
    out_dir = work_dir or source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{source_path.stem}.cih_audio.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not wav_path.is_file():
        raise RuntimeError(
            f"ffmpeg failed extracting audio from {source_path.name}: "
            f"{(completed.stderr or completed.stdout or '')[-800:]}"
        )
    return wav_path


def _openai_client():
    provider = (os.getenv("CIH_TRANSCRIPTION_PROVIDER") or "openai").strip().lower()
    if provider in {"azure", "azure_openai"}:
        from openai import AzureOpenAI

        endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        deployment = (
            os.getenv("CIH_WHISPER_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT")
            or "whisper"
        ).strip()
        api_version = (
            os.getenv("CIH_WHISPER_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or "2024-06-01"
        ).strip()
        if not endpoint or not api_key:
            raise RuntimeError(
                "Azure transcription requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        ), deployment

    from openai import OpenAI

    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("CIH_OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "OpenAI transcription requires OPENAI_API_KEY (or CIH_OPENAI_API_KEY). "
            "Set CIH_TRANSCRIPTION_PROVIDER=azure_openai (with a Whisper deployment) "
            "or CIH_TRANSCRIPTION_PROVIDER=local for on-box faster-whisper."
        )
    return OpenAI(api_key=api_key), (os.getenv("CIH_WHISPER_MODEL") or "whisper-1").strip()


def _resolve_local_asr_settings() -> dict[str, Any]:
    """Resolve updated CIH ASR_* / CIH_* settings for local faster-whisper."""
    model_name = (
        os.getenv("CIH_FASTER_WHISPER_MODEL")
        or os.getenv("CIH_LOCAL_WHISPER_MODEL")
        or os.getenv("ASR_MODEL_NAME")
        or "small"
    ).strip() or "small"
    # Never prefer a stale absolute base-model path when ASR_MODEL_NAME is set to small/medium.
    asr_name = (os.getenv("ASR_MODEL_NAME") or "").strip()
    if asr_name and model_name.replace("\\", "/").endswith("faster-whisper-base"):
        model_name = asr_name

    device = (
        os.getenv("CIH_LOCAL_WHISPER_DEVICE")
        or os.getenv("ASR_DEVICE")
        or "cpu"
    ).strip() or "cpu"
    compute_type = (
        os.getenv("CIH_LOCAL_WHISPER_COMPUTE")
        or os.getenv("ASR_COMPUTE_TYPE")
        or "int8"
    ).strip() or "int8"
    language = (os.getenv("ASR_LANGUAGE") or os.getenv("CIH_ASR_LANGUAGE") or "").strip() or None

    # Hindi (or any non-English) speech → English text: Whisper task=translate.
    to_english = (os.getenv("CIH_TRANSCRIPT_TO_ENGLISH") or os.getenv("ASR_TRANSLATE_TO_ENGLISH") or "").strip().lower()
    task = (os.getenv("ASR_TASK") or "").strip().lower()
    if to_english in {"1", "true", "yes", "on"}:
        task = "translate"
    if not task:
        task = "translate" if to_english in {"1", "true", "yes", "on"} else "transcribe"
    if task not in {"transcribe", "translate"}:
        task = "transcribe"

    try:
        beam_size = int((os.getenv("ASR_BEAM_SIZE") or "3").strip() or "3")
    except ValueError:
        beam_size = 3

    return {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "task": task,
        "beam_size": max(1, beam_size),
    }


def current_asr_fingerprint() -> dict[str, Any]:
    """Public ASR settings used to decide whether a sidecar is stale."""
    settings = _resolve_local_asr_settings()
    provider = (os.getenv("CIH_TRANSCRIPTION_PROVIDER") or "local").strip().lower()
    return {
        "provider": provider,
        "model": settings["model"],
        "task": settings["task"],
        "language": settings["language"] or "auto",
    }


def sidecar_is_reusable(payload: dict[str, Any] | None) -> bool:
    """Return False when sidecar was produced with old model/task (e.g. base + transcribe)."""
    if not payload or not str(payload.get("full_text") or "").strip():
        return False
    force = (os.getenv("CIH_FORCE_RETRANSCRIBE") or "").strip().lower()
    if force in {"1", "true", "yes", "on"}:
        return False
    current = current_asr_fingerprint()
    if current["provider"] not in {"local", "faster_whisper", "faster-whisper", "whisper_local"}:
        return True
    sidecar_model = str(payload.get("model") or "").strip()
    sidecar_task = str(payload.get("task") or "").strip().lower()
    # Old sidecars have no task field and often point at faster-whisper-base.
    if not sidecar_task:
        return False
    if sidecar_task != current["task"]:
        return False
    if sidecar_model and sidecar_model != current["model"]:
        # Treat path-vs-name variants of the same size as matching only when both end with same tag.
        if "faster-whisper-base" in sidecar_model.replace("\\", "/") and current["model"] != "base":
            return False
        if sidecar_model != current["model"]:
            return False
    return True


def _transcribe_local(audio_path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """On-box STT via faster-whisper (updated CIH ASR_* — translate to English when configured)."""
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "Local transcription requires faster-whisper (and its deps). "
            "pip install faster-whisper requests, or set CIH_TRANSCRIPTION_PROVIDER=azure_openai "
            "after creating a Whisper deployment in Azure. "
            f"Import error: {exc}"
        ) from exc

    settings = _resolve_local_asr_settings()
    model = WhisperModel(
        settings["model"],
        device=settings["device"],
        compute_type=settings["compute_type"],
    )
    seg_iter, info = model.transcribe(
        str(audio_path),
        beam_size=settings["beam_size"],
        language=settings["language"],
        task=settings["task"],
    )
    detected = getattr(info, "language", None)
    segments: list[dict[str, Any]] = []
    parts: list[str] = []
    for index, seg in enumerate(seg_iter, start=1):
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        parts.append(text)
        segments.append(
            {
                "index": index,
                "start": getattr(seg, "start", None),
                "end": getattr(seg, "end", None),
                "text": text,
            }
        )
    meta = {
        "model": settings["model"],
        "task": settings["task"],
        "language": settings["language"] or detected or "auto",
        "detected_language": detected,
        "output_language": "en" if settings["task"] == "translate" else (settings["language"] or detected or "auto"),
    }
    return " ".join(parts).strip(), segments, meta


def _transcribe_cloud(audio_path: Path, provider: str) -> tuple[str, list[dict[str, Any]], str]:
    client, model = _openai_client()
    with audio_path.open("rb") as handle:
        try:
            result = client.audio.transcriptions.create(
                model=model,
                file=handle,
                response_format="verbose_json",
            )
        except Exception as first_err:
            err_text = str(first_err)
            if "DeploymentNotFound" in err_text or (
                "deployment" in err_text.lower() and "not exist" in err_text.lower()
            ):
                raise RuntimeError(
                    f"Azure Whisper deployment '{model}' was not found on "
                    f"{(os.getenv('AZURE_OPENAI_ENDPOINT') or '').rstrip('/')}. "
                    "In Azure AI Foundry, deploy model 'whisper' (or gpt-4o-transcribe) "
                    f"and set CIH_WHISPER_DEPLOYMENT to that name, or set "
                    f"CIH_TRANSCRIPTION_PROVIDER=local. Current={model!r}."
                ) from first_err
            handle.seek(0)
            try:
                result = client.audio.transcriptions.create(
                    model=model,
                    file=handle,
                    response_format="json",
                )
            except Exception as second_err:
                err2 = str(second_err)
                if "DeploymentNotFound" in err2 or (
                    "deployment" in err2.lower() and "not exist" in err2.lower()
                ):
                    raise RuntimeError(
                        f"Azure Whisper deployment '{model}' was not found. "
                        "Deploy Whisper in Azure or set CIH_TRANSCRIPTION_PROVIDER=local."
                    ) from second_err
                raise

    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    else:
        data = {"text": str(getattr(result, "text", "") or "")}

    full_text = str(data.get("text") or "").strip()
    segments: list[dict[str, Any]] = []
    raw_segments = data.get("segments") or []
    if isinstance(raw_segments, list):
        for index, seg in enumerate(raw_segments, start=1):
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "index": index,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": text,
                }
            )
    if not segments and full_text:
        segments = [{"index": 1, "start": 0.0, "end": None, "text": full_text}]
    return full_text, segments, model


def transcribe_media_file(document_path: Path) -> dict[str, Any]:
    """Run Whisper transcription. Uses ffmpeg for video; audio may go direct to STT."""
    provider = (os.getenv("CIH_TRANSCRIPTION_PROVIDER") or "openai").strip().lower()
    if provider in {"none", "off", "disabled"}:
        raise RuntimeError("CIH transcription is disabled (CIH_TRANSCRIPTION_PROVIDER=none).")

    work = Path(tempfile.mkdtemp(prefix="cih_transcribe_"))
    wav_path: Path | None = None
    upload_path = document_path
    try:
        suffix = document_path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            wav_path = extract_audio_wav(document_path, work_dir=work)
            upload_path = wav_path
        elif suffix in AUDIO_SUFFIXES:
            try:
                _resolve_ffmpeg_exe()
                wav_path = extract_audio_wav(document_path, work_dir=work)
                upload_path = wav_path
            except Exception:
                logger.warning("ffmpeg normalize unavailable; uploading original audio", exc_info=True)
                upload_path = document_path
        else:
            raise ValueError(f"Not a media file: {suffix}")

        if provider in {"local", "faster_whisper", "faster-whisper", "whisper_local"}:
            full_text, segments, local_meta = _transcribe_local(upload_path)
            return {
                "source": "media_transcription",
                "media_type": "video" if suffix in VIDEO_SUFFIXES else "audio",
                "provider": provider,
                "model": local_meta.get("model"),
                "task": local_meta.get("task"),
                "language": local_meta.get("language"),
                "detected_language": local_meta.get("detected_language"),
                "output_language": local_meta.get("output_language"),
                "full_text": full_text,
                "segments": segments,
                "segment_count": len(segments),
                "original_file_name": document_path.name,
            }

        full_text, segments, model = _transcribe_cloud(upload_path, provider)
        return {
            "source": "media_transcription",
            "media_type": "video" if suffix in VIDEO_SUFFIXES else "audio",
            "provider": provider,
            "model": model,
            "task": "transcribe",
            "full_text": full_text,
            "segments": segments,
            "segment_count": len(segments),
            "original_file_name": document_path.name,
        }
    finally:
        if wav_path and wav_path.is_file() and work in wav_path.parents:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass
