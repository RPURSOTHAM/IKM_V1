# IKM OCR Port Package

Reference files from **rag-builder** to add Tesseract OCR (PDF scanned pages + image uploads) into an existing IKM that does not have OCR.

---

## Zip folder layout

```
ocr-port-package/
├── README.md                          ← This file (where to store + how to merge)
├── ANTIGRAVITY_PROMPT.txt             ← Copy-paste prompt for Antigravity (Gemini 3.7 Flash Medium)
├── ANTIGRAVITY_PROMPT_DOCX_PHASE2.txt ← Optional Phase 2: DOCX OCR extension
│
├── 01_COPY_AS_IS/                     ← Copy these files into target IKM (same paths)
│   └── src/features/document_processing/loaders/
│       ├── image_ocr_loader.py        ← NEW: Tesseract on images
│       └── scanned_pdf_ocr.py         ← NEW: PDF scanned-page OCR fallback
│   └── tests/unit/documents/
│       └── test_image_ocr_loader.py   ← Unit tests (recommended)
│
├── 02_MERGE_PATCHES/                  ← Do NOT overwrite — MERGE these into existing files
│   └── src/features/document_processing/loaders/
│       └── document_text.py           ← OCR integration hook
│   └── src/features/document_processing/shared_processor/deployment/
│       ├── ids.py                     ← OCR processor id enum
│       └── implementations.py         ← OcrProcessor class + registry
│   └── configs/
│       └── processors.yaml            ← ocr: true
│   └── deploy/application/
│       ├── Dockerfile.processor       ← apt install tesseract-ocr
│       └── Dockerfile.dms             ← apt install tesseract-ocr
│
└── 03_REFERENCE_ONLY/                 ← For context only — already exist in IKM
    ├── loader.py                      ← Block, DocumentLoader types
    └── component_classification.py    ← is_image_block() used by scanned_pdf_ocr
```

---

## Where to store files in target IKM

All paths below are **relative to your IKM repo root** (e.g. `rag-builder/`).

| Action | Source in this zip | Target in IKM repo |
|--------|-------------------|-------------------|
| **COPY** | `01_COPY_AS_IS/src/features/document_processing/loaders/image_ocr_loader.py` | `src/features/document_processing/loaders/image_ocr_loader.py` |
| **COPY** | `01_COPY_AS_IS/src/features/document_processing/loaders/scanned_pdf_ocr.py` | `src/features/document_processing/loaders/scanned_pdf_ocr.py` |
| **COPY** | `01_COPY_AS_IS/tests/unit/documents/test_image_ocr_loader.py` | `tests/unit/documents/test_image_ocr_loader.py` |
| **MERGE** | `02_MERGE_PATCHES/.../document_text.py` | `src/features/document_processing/loaders/document_text.py` |
| **MERGE** | `02_MERGE_PATCHES/.../ids.py` | `src/features/document_processing/shared_processor/deployment/ids.py` |
| **MERGE** | `02_MERGE_PATCHES/.../implementations.py` | `src/features/document_processing/shared_processor/deployment/implementations.py` |
| **MERGE** | `02_MERGE_PATCHES/configs/processors.yaml` | `configs/processors.yaml` |
| **MERGE** | `02_MERGE_PATCHES/deploy/application/Dockerfile.processor` | `deploy/application/Dockerfile.processor` |
| **MERGE** | `02_MERGE_PATCHES/deploy/application/Dockerfile.dms` | `deploy/application/Dockerfile.dms` |

---

## How to merge (manual)

### Step 1 — Copy new files (no merge needed)

Copy everything under `01_COPY_AS_IS/` into your IKM repo keeping the same folder structure.

### Step 2 — Merge `document_text.py`

Open your existing `document_text.py` and add these changes from the reference file:

1. **In `_ensure_format_processor_enabled()`** — for image suffixes, require both `image` and `ocr` processors:
   - `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` → check `is_processor_enabled("image")` AND `is_processor_enabled("ocr")`

2. **In `load_document_blocks()` — PDF branch:**
   ```python
   blocks = PdfLoader().load(document_path)
   from src.features.document_processing.loaders.scanned_pdf_ocr import apply_ocr_to_scanned_pdf
   blocks = apply_ocr_to_scanned_pdf(document_path, blocks)
   ```

3. **In `load_document_blocks()` — add image branch** (if missing):
   ```python
   elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
       from src.features.document_processing.loaders.image_ocr_loader import ImageOcrLoader
       blocks = ImageOcrLoader().load(document_path)
   ```

4. **Before return — add `ocr_used` metadata:**
   ```python
   ocr_used = any(
       isinstance(getattr(block, "metadata", None), dict) and block.metadata.get("ocr_used")
       for block in blocks
   )
   # Add "ocr_used": ocr_used to metadata dict
   ```

### Step 3 — Merge `ids.py`

Add to enum and catalogs:
- `OCR = "ocr"` in `DeploymentProcessorId`
- `DeploymentProcessorId.OCR` in `DEPLOYMENT_PROCESSOR_ORDER`
- `DeploymentProcessorId.OCR: "OCR"` in `DEPLOYMENT_PROCESSOR_DISPLAY_NAMES`

### Step 4 — Merge `implementations.py`

Add `OcrProcessor` class and register:
```python
DeploymentProcessorId.OCR: OcrProcessor,
```
in `PROCESSOR_CLASSES`.

### Step 5 — Merge `processors.yaml`

Add under `processors:`:
```yaml
  ocr: true
```
Also ensure `image: true` and `pdf: true`.

### Step 6 — Merge Dockerfiles

In the `apt-get install` line of **both** Dockerfiles, add:
```
tesseract-ocr
```

### Step 7 — Verify `requirements.txt`

Ensure these exist (likely already present):
```
pymupdf==1.24.14
Pillow>=10.0.0
```
**Do NOT add** pytesseract/easyocr/paddleocr for this OCR path — reference uses Tesseract CLI only.

### Step 8 — Rebuild and test

```bash
# Rebuild Docker images (processor + DMS)
docker compose -f deploy/application/docker-compose.application.yml build

# Run tests
pytest tests/unit/documents/test_image_ocr_loader.py -v
```

---

## What OCR covers

| Format | OCR? | How |
|--------|------|-----|
| PDF (scanned pages) | Yes | `apply_ocr_to_scanned_pdf()` after PdfLoader |
| Images (.png, .jpg, etc.) | Yes | `ImageOcrLoader()` |
| DOCX | No (Phase 1) | Native DocxLoader only — see Phase 2 prompt |
| Normal PDF with text | No | OCR skipped when native text is usable |

---

## Antigravity

Use **`ANTIGRAVITY_PROMPT.txt`** in this zip — attach the whole zip or the files listed above when running Antigravity with Gemini 3.7 Flash Medium.

For DOCX OCR later, use **`ANTIGRAVITY_PROMPT_DOCX_PHASE2.txt`**.

---

## Disable OCR

```yaml
# configs/processors.yaml
processors:
  ocr: false
```
Or env: `IKM_DISABLED_PROCESSORS=ocr` or `PROCESSOR_OCR_ENABLED=false`
