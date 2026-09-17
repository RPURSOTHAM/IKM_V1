## RabbitMQ Message Simulator

Use `post_rabbitmq_message.py` to publish a scheduler-compatible document processing message to RabbitMQ.

The published payload matches the scheduler contract:

```json
{
  "document_id": "UUID",
  "document_name": "stored-file-name.pdf",
  "collection_name": "DocumentChunk",
  "tenant_id": "tenant-1"
}
```

Example:

```sh
python src/simulators/post_rabbitmq_message.py \
  --document-id 9d9b1f53-8d15-4ef1-9c35-6a0da99f2321 \
  --document-name 9d9b1f53-8d15-4ef1-9c35-6a0da99f2321.pdf \
  --collection-name DocumentChunk \
  --tenant-id tenant-1
```

Preview without publishing:

```sh
python src/simulators/post_rabbitmq_message.py \
  --document-id test-doc-1 \
  --document-name test-doc-1.pdf \
  --dry-run
```

RabbitMQ connection defaults are read from environment variables:

- `RABBITMQ_HOST`
- `RABBITMQ_PORT`
- `RABBITMQ_USER`
- `RABBITMQ_PASS`
- `RABBITMQ_QUEUE_NAME`

## Direct processor job (`submit_processor_job.py`)

When `src.processor_service` is running as its own HTTP service (default **`http://localhost:8082`**), use **`submit_document_for_processing()`** in `submit_processor_job.py`: it picks the **newest** `.pdf` / `.docx` / `.txt` under the repo **`_documents`** folder and **`POST /process`**. Import the function or run:

```sh
python src/simulators/submit_processor_job.py
```

Optional: `--document-path`, `--documents-dir`, `--base-url`, **`--wait`** (poll **`GET /health`** until the job finishes), **`--dry-run`**.

## Processor integration tests (`processor_tests/`)

Per-processor validation scripts with backend checks (Weaviate, Neo4j, Redis):

```sh
cd rag-builder
PYTHONPATH=. python src/simulators/processor_tests/run_all.py
```

See [`processor_tests/README.md`](processor_tests/README.md) for individual scripts and `--mode direct|http|auto`.

Use `streamlit_document_uploader.py` as a browser UI for the Consumer API: uploads, Weaviate repository admin, RabbitMQ queue admin, and scheduler / processor pool status.

Install simulator dependencies:

```sh
pip install streamlit requests
```

Run the app:

```sh
streamlit run src/simulators/streamlit_document_uploader.py
```

Tabs:

- **Upload / Documents** — `POST /api/v1/documents/upload`, document list and status refresh.
- **Repository** — `GET /api/v1/repository/collections`, tenants, documents, chunks; delete document or collection.
- **Queue** — `GET /api/v1/queues`, peek messages, delete from head, post test document message.
- **Scheduler** — `GET /api/v1/scheduler/health`, `GET /api/v1/scheduler/processors` (per-slot Docker + HTTP health and active document).

Configuration:

- Default API base URL is `http://localhost:8088`; version prefix defaults to `/api/v1`.
- Set `RAG_API_BASE_URL` or `CONSUMER_API_BASE_URL`, and `CONSUMER_API_PREFIX` if needed.
- When API keys are enabled, set `CONSUMER_API_KEY` or configure the sidebar field.
