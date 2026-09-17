# Content Intelligence Hub

Thin DMS feature: `src/features/content_intelligence_hub`  
Swagger tag: **Content Intelligence Hub** under `/api/v1/cih`

## Endpoints

| # | Method | Path |
|---|--------|------|
| 1 | POST | `/api/v1/cih/upload` (requires `repository_id` + files) |
| 2 | GET | `/api/v1/cih/{document_id}/status` |
| 3 | GET | `/api/v1/cih/{document_id}/extraction` |
| 4 | GET | `/api/v1/cih/{document_id}/chunks` |
| 5 | GET | `/api/v1/cih/{document_id}/metadata` |
| 6 | GET | `/api/v1/cih/{document_id}/transcript` |
| 7 | POST | `/api/v1/cih/image-summary` (Ollama VLM image summary) |
| 8 | GET | `/api/v1/cih/{document_id}/image-summary` |

Uses `DocumentReceiverService` for upload/status/extraction, plus Ollama for image summary.

## Models (updated CIH `.env`)

| Capability | Env | Default / your value |
|------------|-----|----------------------|
| Local ASR | `CIH_TRANSCRIPTION_PROVIDER=local` + `CIH_FASTER_WHISPER_MODEL` | `/app/src/models/faster-whisper-small` |
| Hindi/other → **English** transcript | `ASR_TASK=translate` + `CIH_TRANSCRIPT_TO_ENGLISH=true` | Whisper translate (always English) |
| ASR tuning | `ASR_LANGUAGE` (empty=auto), `ASR_BEAM_SIZE`, `ASR_COMPUTE_TYPE` | from updated CIH block |

**Important:** Old jobs reused `.cih_transcript.json` from `faster-whisper-base` + `transcribe` (keeps Hindi/garbled).  
Stale sidecars are ignored when model/task change. Re-upload or republish `media_transcription` after changing ASR env.

| Image summary (Ollama) | `OPEN_WEIGHT_VLM_*`, `OLLAMA_*` | `gemma4:31b` (gemma3:27b retired) |
| Image summary fallback | `GEMINI_API_KEY`, `GEMINI_MODEL_NAME` | `gemini-2.5-flash` when `CIH_IMAGE_SUMMARY_PROVIDER=auto` |

Set `OPEN_WEIGHT_VLM_ENABLED=true` and `IMAGE_DESCRIPTION_ENABLED=true`.  
Provider: `CIH_IMAGE_SUMMARY_PROVIDER=auto|ollama|gemini`.

## Swagger test — image summary

1. Authorize with JWT from `POST /api/v1/auth/token`
2. **7. Summarize image (upload)** → choose a PNG/JPG → Execute  
   Expect `provider`/`model`/`summary`.
3. Or upload via CIH upload, then **8. Summarize uploaded image** with `document_id`

## Swagger test — audio/video transcript

1. Upload wav/mp4 with `repository_id`
2. Poll status until COMPLETED
3. Get transcript (uses Whisper **`small`** via `CIH_FASTER_WHISPER_MODEL` / `ASR_MODEL_NAME`)

Restart **doc processors** after changing ASR env so workers pick up `small` instead of the old `faster-whisper-base` path.
