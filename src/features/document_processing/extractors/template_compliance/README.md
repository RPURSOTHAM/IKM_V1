# Template Compliance — Extraction Component

Standalone template extraction engine for Document Optimizer.

**Focus now:** DOCX (complete).  
**Later:** PDF uses the same JSON schema (stubbed).

This folder is intentionally self-contained so you can develop and test it without running the full Optimizer UI/API.

## JSON output (common schema)

| Area | Description |
|------|-------------|
| `metadata` | Core properties + Document No / Version / dates, etc. |
| `headers` / `footers` | Text, tables, images, and field labels per section |
| `logo` | Header (and body) logo presence + image parts |
| `tables` | Row/column counts and cell values (body + HF); images in cells |
| `repeated_fields` | Labels/values that repeat across the document |
| `sections` | Outline max 2 levels (`1.0`, `1.1` / `2`, `2.1`) |
| `fonts` | Dominant font name & size (body / heading) |
| `line_spacing` | Minimum detected line spacing |
| `toc` | Table of Contents presence + field/title signals |
| `images` | All embedded images including table cells |
| `section_indentation` | Left / first-line indent per outline section |

## Run locally (separate from Optimizer UI)

From repo `backend/document_optimizer`:

```bash
# Extract one DOCX → JSON
python -m template_compliance extract path\to\template.docx -o out.json

# Pretty-print to stdout
python -m template_compliance extract path\to\template.docx --pretty

# Run component unit tests
python -m unittest discover -s template_compliance/tests -v
```

From repo `backend` (if you prefer):

```bash
python -m document_optimizer.template_compliance extract path\to\template.docx -o out.json
```

(`document_optimizer` needs to be on `PYTHONPATH`; `cd backend` usually works.)

## Public API

```python
from template_compliance import extract_template

result = extract_template(r"C:\path\to\template.docx")
# result is a dict matching schema_version "1.0"
```

## Layout

```
template_compliance/
  __init__.py          # public exports
  __main__.py          # python -m template_compliance
  api.py               # extract_template()
  cli.py
  schema.py            # schema version + empty document factory
  exceptions.py
  docx_extract.py      # full DOCX extractor
  pdf_extract.py       # PDF stub (same schema later)
  tests/
    test_docx_extract.py
```

PDF extraction raises a clear “not implemented” error today so DOCX testing stays isolated.