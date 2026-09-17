# Processor integration tests

Individual scripts to validate each document processor type against live infrastructure.

## Prerequisites

1. Start dependencies: `cd src/dependencies && docker compose up -d`
2. Ensure embedding model exists: `src/models/bge-base-en/`
3. For **HTTP** / **auto** chunking tests: scheduler-managed processors on ports `3100+`
4. Host Python packages (direct mode): `pip install redis neo4j weaviate-client python-dotenv`

## Run all tests

```powershell
cd rag-builder
$env:PYTHONPATH = (Get-Location).Path
python src/simulators/processor_tests/run_all.py
```

Default **`auto`** mode:

| Processor | Mode | Why |
|-----------|------|-----|
| `chunking_vectorizing` | HTTP → Docker processor | Needs Linux/torch stack + mounted models |
| All others | Direct in-process | Neo4j / Redis / rule-based extraction |

## Individual scripts

```powershell
python src/simulators/processor_tests/test_chunking_vectorizing.py --mode http --base-url http://localhost:3100
python src/simulators/processor_tests/test_metadata_extraction.py --mode direct
python src/simulators/processor_tests/test_template_extraction.py --mode direct
python src/simulators/processor_tests/test_reference_extraction.py --mode direct
python src/simulators/processor_tests/test_conversion_rendering.py --mode direct
python src/simulators/processor_tests/test_intelligence_extraction.py --mode direct
```

Options:

- `--mode direct|http|auto` — direct calls processor code; HTTP posts to running container
- `--base-url` — processor HTTP endpoint (default `http://localhost:3100`)
- `--keep` — leave Weaviate/Neo4j/Redis artifacts after test

## Fixture

Shared test document: `fixtures/sample_processor_test.txt` (metadata fields, numbered sections, reference patterns).

## What each test validates

| Script | Storage | Validation |
|--------|---------|------------|
| `test_chunking_vectorizing.py` | Weaviate | Objects in `ProcessorTestCollection` |
| `test_metadata_extraction.py` | Neo4j | `Equipment_Name` extracted |
| `test_template_extraction.py` | Neo4j | Skeleton sections present |
| `test_reference_extraction.py` | Neo4j | Reference IDs from fixture |
| `test_conversion_rendering.py` | Redis | HTML render cache key |
| `test_intelligence_extraction.py` | — | Expected rejection |
