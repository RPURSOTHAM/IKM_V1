# Tools

Developer and operator helpers. **Not** part of the production Compose runtime.

| Path | Purpose |
|------|---------|
| `simulators/` | Streamlit uploader, queue publishers, processor smoke tests |

Run examples (from repo root, `PYTHONPATH=.`):

```powershell
python -m src.simulators.streamlit_document_uploader
```

(`src/simulators` is a junction to this folder for import compatibility.)
