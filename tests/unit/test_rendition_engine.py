"""Automated tests for Backend Document Rendition Engine.

Verifies:
- TEST 1: Valid 1-page PDF rendering
- TEST 2: Multi-page PDF rendering
- TEST 3: Retrieval of valid page image (image/png)
- TEST 4: Invalid page number request handling
- TEST 5: Unsupported file format handling
- TEST 6: Corrupted PDF document handling
"""

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from src.application.consumer_api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _create_sample_pdf(page_count: int = 1) -> bytes:
    """Helper to generate a valid PDF document with N pages in memory."""
    doc = fitz.open()
    for idx in range(page_count):
        page = doc.new_page(width=595, height=842)
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(50, 50, 500, 750))
        shape.finish(color=(0.2, 0.4, 0.8), fill=(0.9, 0.95, 1.0))
        shape.commit()
        page.insert_text((70, 100), f"Test Document - Page {idx + 1}", fontsize=20)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _extract_error_text(resp) -> str:
    """Extract human readable error string from HTTP response."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            if "error" in body and isinstance(body["error"], dict):
                return str(body["error"].get("message") or "")
            if "detail" in body:
                return str(body["detail"])
    except Exception:
        pass
    return resp.text


def test_1_render_valid_1_page_pdf(client):
    """TEST 1: Render a valid 1-page PDF."""
    pdf_bytes = _create_sample_pdf(page_count=1)
    response = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample_1page.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, f"Unexpected response: {response.text}"
    data = response.json()

    assert data["rendition_id"].startswith("ren_")
    assert data["document_name"] == "sample_1page.pdf"
    assert data["document_type"] == "pdf"
    assert data["status"] == "COMPLETED"
    assert data["page_count"] == 1
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert data["pages"][0]["width"] > 0
    assert data["pages"][0]["height"] > 0


def test_2_render_multi_page_pdf(client):
    """TEST 2: Render a multi-page PDF."""
    pdf_bytes = _create_sample_pdf(page_count=3)
    response = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample_3page.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, f"Unexpected response: {response.text}"
    data = response.json()

    assert data["rendition_id"].startswith("ren_")
    assert data["status"] == "COMPLETED"
    assert data["page_count"] == 3
    assert len(data["pages"]) == 3

    # Check status endpoint
    rendition_id = data["rendition_id"]
    status_resp = client.get(f"/api/v1/rendering/{rendition_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "COMPLETED"

    # Check details endpoint
    details_resp = client.get(f"/api/v1/rendering/{rendition_id}")
    assert details_resp.status_code == 200
    details_data = details_resp.json()
    assert details_data["page_count"] == 3


def test_3_get_valid_page_image(client):
    """TEST 3: Request a valid rendered page image."""
    pdf_bytes = _create_sample_pdf(page_count=2)
    render_resp = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample_2page.pdf", pdf_bytes, "application/pdf")},
    )
    rendition_id = render_resp.json()["rendition_id"]

    # Request Page 1
    page1_resp = client.get(f"/api/v1/rendering/{rendition_id}/pages/1")
    assert page1_resp.status_code == 200
    assert page1_resp.headers["content-type"] == "image/png"
    assert len(page1_resp.content) > 0

    # Request Page 2
    page2_resp = client.get(f"/api/v1/rendering/{rendition_id}/pages/2")
    assert page2_resp.status_code == 200
    assert page2_resp.headers["content-type"] == "image/png"
    assert len(page2_resp.content) > 0


def test_4_request_invalid_page_number(client):
    """TEST 4: Request an invalid page number (out of bounds)."""
    pdf_bytes = _create_sample_pdf(page_count=1)
    render_resp = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample_1page.pdf", pdf_bytes, "application/pdf")},
    )
    rendition_id = render_resp.json()["rendition_id"]

    # Request Page 99 (out of bounds)
    page_resp = client.get(f"/api/v1/rendering/{rendition_id}/pages/99")
    assert page_resp.status_code == 404
    error_text = _extract_error_text(page_resp)
    assert "out of bounds" in error_text.lower() or "total pages" in error_text.lower()


def test_5_unsupported_file_type(client):
    """TEST 5: Upload an unsupported file type."""
    response = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample.docx", b"dummy docx content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 400
    error_text = _extract_error_text(response)
    assert "unsupported" in error_text.lower() or "pdf" in error_text.lower()


def test_6_corrupted_pdf_handling(client):
    """TEST 6: Upload a corrupted PDF file."""
    corrupted_bytes = b"%PDF-1.4 Junk corrupted bytes that are not a valid PDF header or stream"
    response = client.post(
        "/api/v1/rendering/render",
        files={"file": ("corrupted.pdf", corrupted_bytes, "application/pdf")},
    )
    assert response.status_code in (400, 500)
    error_text = _extract_error_text(response)
    assert "corrupted" in error_text.lower() or "pdf" in error_text.lower() or "invalid" in error_text.lower()


def test_7_get_full_document_pages_view(client):
    """TEST 7: GET /api/v1/rendering/{rendition_id}/pages full document view."""
    pdf_bytes = _create_sample_pdf(page_count=2)
    render_resp = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample_fullview.pdf", pdf_bytes, "application/pdf")},
    )
    rendition_id = render_resp.json()["rendition_id"]

    # Request full document view (without page number)
    full_resp = client.get(f"/api/v1/rendering/{rendition_id}/pages")
    assert full_resp.status_code == 200
    assert "text/html" in full_resp.headers["content-type"]
    html_text = full_resp.text
    assert f"/api/v1/rendering/{rendition_id}/pages/1" in html_text
    assert f"/api/v1/rendering/{rendition_id}/pages/2" in html_text
    assert f"/api/v1/rendering/{rendition_id}/download?pages=all" in html_text

    # Request full view for invalid rendition ID
    invalid_resp = client.get("/api/v1/rendering/ren_invalid_99999/pages")
    assert invalid_resp.status_code == 404


def test_8_download_selected_pages_zip(client):
    """TEST 8: Download selected / all pages as ZIP."""
    import io
    import zipfile

    pdf_bytes = _create_sample_pdf(page_count=3)
    render_resp = client.post(
        "/api/v1/rendering/render",
        files={"file": ("sample_zip.pdf", pdf_bytes, "application/pdf")},
    )
    assert render_resp.status_code == 200, render_resp.text
    rendition_id = render_resp.json()["rendition_id"]

    all_resp = client.get(f"/api/v1/rendering/{rendition_id}/download", params={"pages": "all"})
    assert all_resp.status_code == 200
    assert all_resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(all_resp.content)) as zf:
        names = sorted(zf.namelist())
        assert names == ["page_1.png", "page_2.png", "page_3.png"]

    selected_resp = client.get(
        f"/api/v1/rendering/{rendition_id}/download",
        params={"pages": "1,3"},
    )
    assert selected_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(selected_resp.content)) as zf:
        assert sorted(zf.namelist()) == ["page_1.png", "page_3.png"]

    range_resp = client.get(
        f"/api/v1/rendering/{rendition_id}/download",
        params={"pages": "2-3"},
    )
    assert range_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(range_resp.content)) as zf:
        assert sorted(zf.namelist()) == ["page_2.png", "page_3.png"]

    bad_resp = client.get(
        f"/api/v1/rendering/{rendition_id}/download",
        params={"pages": "99"},
    )
    assert bad_resp.status_code == 400


def test_9_render_accepts_repository_id_form_field(client):
    """TEST 9: repository_id is accepted; invalid repo returns error (not preview-only success)."""
    pdf_bytes = _create_sample_pdf(page_count=1)
    response = client.post(
        "/api/v1/rendering/render",
        files={"file": ("repo_doc.pdf", pdf_bytes, "application/pdf")},
        data={
            "repository_id": "00000000-0000-0000-0000-000000000000",
            "submit_for_processing": "false",
        },
    )
    # Invalid / inactive repository should fail during upload — not silently preview-only.
    assert response.status_code in {400, 404, 422, 500, 503}
    assert response.status_code != 200 or response.json().get("repository_id")

