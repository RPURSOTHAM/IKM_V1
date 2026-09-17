# API integration tests

Live HTTP tests against a running DMS Consumer API.

## Prerequisites

- DMS API running (default `http://localhost:8088`)
- MySQL + platform security seeded (`PLATFORM_BOOTSTRAP_ADMIN_*`)

## Run

```powershell
cd tests
pip install -r requirements.txt
pytest
```

Optional environment:

```env
DMS_API_BASE_URL=http://localhost:8088
DMS_API_USER=admin
DMS_API_PASSWORD=admin123
```

## Layout

| File | Resource |
|------|----------|
| `api/test_repository_catalogs.py` | Embedding models, chunking strategies |
| `api/test_repositories.py` | Repository CRUD |
| `api/test_repository_settings.py` | Repository settings PATCH |
| `api/test_repository_documents.py` | Repository documents list |
| `api/test_repository_registration.py` | Registration requests, activate |
| `api/test_repository_members.py` | Members, grants |
| `api/test_document_types.py` | Document type list/get/patch |
| `api/test_document_type_children.py` | Child types |
| `api/test_document_type_metadata_fields.py` | Metadata fields |
| `api/test_*_negative.py` | Negative-path / error-envelope tests per resource |
| `api/helpers.py` | `assert_status`, `assert_error` (checks `error.code`, `message`, `request_id`) |
