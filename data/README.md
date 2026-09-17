# Data

Runtime data directories. Contents are gitignored; directories are kept via `.gitkeep`.

| Path | Purpose | Env |
|------|---------|-----|
| `documents/` | Uploaded source files | `DOCUMENT_ROOT`, `HOST_DOCUMENTS_DIR` |
| `templates/` | Document / SOP templates | — |
| `reports/` | Generated reports | — |

Legacy junctions at repo root (`_documents`, `_templates`, `_reports`) point here for older scripts.
