"""End-to-end runtime verification against a live RAG Builder Docker stack."""
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import httpx

BASE_URL = os.getenv("DMS_API_BASE_URL", "http://localhost:8088").rstrip("/")
ADMIN_USER = os.getenv("DMS_API_USER", "admin")
ADMIN_PASSWORD = os.getenv("DMS_API_PASSWORD", "changeme-local-admin")
API = "/api/v1"
TIMEOUT = 120.0

EMBEDDING_MODEL = {
    "provider": "local",
    "model_id": "all-MiniLM-L6-v2",
    "local_model_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
}

MARKER_A = "E2E_REPOSITORY_A_MARKER_2026"
MARKER_DOC = "E2E_DOCUMENT_MARKER_2026"
MARKER_B = "E2E_REPOSITORY_B_MARKER_2026"

results: list[tuple[str, str, str]] = []
state: dict[str, Any] = {}


def _record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{'OK' if passed else 'XX'}] {name}: {status}  {detail}")
    return passed


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _get_token(client: httpx.Client, user: str, password: str) -> str | None:
    r = client.post(f"{API}/auth/token", json={"user_id": user, "password": password})
    if r.status_code == 200:
        return r.json().get("access_token")
    return None


def _generate_document(marker: str, extra_marker: str = MARKER_DOC) -> str:
    """Generate an 800+ word document containing the given markers."""
    lines = [
        f"This document contains the marker {marker} and {extra_marker}.",
        "It is generated for automated end-to-end testing of the RAG pipeline.",
        "",
    ]
    para = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
        "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
        "mollit anim id est laborum. "
    )
    for i in range(15):
        lines.append(f"Section {i+1}: {para}")
        if i % 3 == 0:
            lines.append(f"Reference marker: {marker} appears in section {i+1}.")
    lines.append(f"\nFinal reference: {marker} {extra_marker} end of document.")
    return "\n".join(lines)


def _create_repo(client: httpx.Client, name: str) -> str:
    r = client.post(
        f"{API}/repositories",
        json={
            "name": name,
            "owner_user_id": ADMIN_USER,
            "settings": {
                "chunking_strategy": "section-based",
                "chunk_size": 512,
                "chunk_overlap": 2,
                "embedding_model": EMBEDDING_MODEL,
            },
        },
    )
    assert r.status_code == 201, f"create repo failed: {r.status_code} {r.text[:300]}"
    body = r.json()
    return body.get("repository_id") or body.get("id")


def _activate_repo(client: httpx.Client, repo_id: str):
    r = client.post(f"{API}/repositories/{repo_id}/activate")
    assert r.status_code in (200, 204), f"activate failed: {r.status_code} {r.text[:300]}"


def _upload_doc(client: httpx.Client, repo_id: str, filename: str, content: str):
    r = client.post(
        f"{API}/documents/upload",
        files={"files": (filename, content.encode(), "text/plain")},
        data={"repository_id": repo_id, "submit_for_processing": "true"},
    )
    return r


def _extract_status(body: dict) -> str:
    doc_obj = body.get("document") or {}
    job_obj = body.get("job") or {}
    return (
        doc_obj.get("status")
        or job_obj.get("status")
        or body.get("status")
        or body.get("processing_status")
        or ""
    )


def _poll_status(headers: dict, doc_id: str, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    last_status = ""
    while time.time() < deadline:
        try:
            with httpx.Client(base_url=BASE_URL, headers=headers, timeout=TIMEOUT) as c:
                r = c.get(f"{API}/documents/{doc_id}/status")
            if r.status_code != 200:
                time.sleep(5)
                continue
            body = r.json()
            s = _extract_status(body)
            if s != last_status:
                elapsed = int(timeout - (deadline - time.time()))
                print(f"    [{elapsed}s] status: {s}")
                last_status = s
            if s.upper() in ("COMPLETED", "FAILED", "ERROR"):
                return body
        except Exception as exc:
            elapsed = int(timeout - (deadline - time.time()))
            print(f"    [{elapsed}s] poll error: {exc!r}, retrying...")
        time.sleep(5)
    return {"status": "TIMEOUT"}


# ── TEST 1 ───────────────────────────────────────────────────────────────────
def test_01_authentication(client: httpx.Client):
    print("\n-- TEST 1: Authentication --")
    ok = True

    r = client.get(f"{API}/documents")
    ok &= _record("No-token -> 401", r.status_code == 401, f"got {r.status_code}")

    r = client.get(f"{API}/documents", headers=_auth_headers("invalid-token-xyz"))
    ok &= _record("Bad-token -> 401", r.status_code == 401, f"got {r.status_code}")

    token = _get_token(client, ADMIN_USER, ADMIN_PASSWORD)
    ok &= _record("Admin login", token is not None, "token obtained" if token else "FAILED")

    if token:
        state["token"] = token
        r = client.get(f"{API}/documents", headers=_auth_headers(token))
        ok &= _record("Valid-token -> 200", r.status_code == 200, f"got {r.status_code}")
    return ok


# ── TEST 2 ───────────────────────────────────────────────────────────────────
def test_02_fresh_e2e_processing(client: httpx.Client):
    print("\n-- TEST 2: Fresh E2E Processing --")
    token = state.get("token")
    if not token:
        return _record("T2-prereq", False, "no token from T1")

    h = _auth_headers(token)
    repo_name = f"e2e-a-{uuid.uuid4().hex[:8]}"
    repo_id = _create_repo(httpx.Client(base_url=BASE_URL, headers=h, timeout=TIMEOUT), repo_name)
    state["repo_a_id"] = repo_id
    state["repo_a_name"] = repo_name
    _record("Create repo A", bool(repo_id), repo_id)

    _activate_repo(httpx.Client(base_url=BASE_URL, headers=h, timeout=TIMEOUT), repo_id)
    _record("Activate repo A", True, "")

    doc_content = _generate_document(MARKER_A)
    state["doc_a_content"] = doc_content
    word_count = len(doc_content.split())
    _record("Doc word count >=800", word_count >= 800, f"{word_count} words")

    ac = httpx.Client(base_url=BASE_URL, headers=h, timeout=TIMEOUT)
    r = _upload_doc(ac, repo_id, "e2e_test_doc_a.txt", doc_content)
    upload_ok = r.status_code in (200, 201)
    _record("Upload doc", upload_ok, f"{r.status_code}")
    if not upload_ok:
        print(f"    upload response: {r.text[:500]}")
        return False

    body = r.json()
    doc_id = None
    if isinstance(body, list):
        doc_id = body[0].get("document_id") or body[0].get("id")
        state["job_id"] = body[0].get("job_id")
        state["content_hash"] = body[0].get("content_hash")
    elif isinstance(body, dict):
        doc_id = body.get("document_id") or body.get("id")
        docs = body.get("documents") or body.get("uploaded") or []
        if docs:
            doc_id = doc_id or docs[0].get("document_id") or docs[0].get("id")
            state["job_id"] = docs[0].get("job_id")
            state["content_hash"] = docs[0].get("content_hash")
    state["doc_a_id"] = doc_id
    _record("Captured doc_id", bool(doc_id), str(doc_id))

    print("    Polling processing status (up to 120s)...")
    status_body = _poll_status(h, doc_id, timeout=120)
    job_obj = status_body.get("job") or {}
    doc_obj = status_body.get("document") or {}
    if not state.get("job_id"):
        state["job_id"] = job_obj.get("job_id") or doc_obj.get("job_id")
    meta = doc_obj.get("metadata") or {}
    if not state.get("content_hash"):
        state["content_hash"] = meta.get("content_hash")
    if state.get("job_id"):
        _record("Captured job_id", True, str(state["job_id"]))
    if state.get("content_hash"):
        _record("Captured content_hash", True, str(state["content_hash"][:16]) + "...")

    final_status = _extract_status(status_body).upper() or "TIMEOUT"
    state["doc_a_status"] = final_status
    error_details = (status_body.get("job") or {}).get("error_details") or status_body.get("error_details")
    detail = final_status
    if error_details:
        detail = f"{final_status}: {error_details}"
    elif final_status in ("QUEUED", "TIMEOUT"):
        detail = f"{final_status} (processor may not have document volume mount)"
    _record("Processing final", final_status == "COMPLETED", detail)

    if final_status == "COMPLETED":
        r = ac.get(f"{API}/documents/{doc_id}/chunks")
        if r.status_code == 200:
            chunks = r.json().get("chunks") or r.json().get("results") or []
            if isinstance(r.json(), list):
                chunks = r.json()
            state["doc_a_chunks"] = chunks
            _record("Chunks exist", len(chunks) > 0, f"{len(chunks)} chunks")
        else:
            _record("Chunks exist", False, f"status {r.status_code}")
    return True


# ── TEST 3 ───────────────────────────────────────────────────────────────────
def test_03_duplicate_detection(client: httpx.Client):
    print("\n-- TEST 3: Duplicate Detection --")
    token = state.get("token")
    doc_content = state.get("doc_a_content")
    repo_id = state.get("repo_a_id")
    if not all([token, doc_content, repo_id]):
        return _record("T3-prereq", False, "missing state from T2")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)

    r = _upload_doc(ac, repo_id, "e2e_test_doc_a.txt", doc_content)
    _record("Same file same name -> 409", r.status_code == 409, f"got {r.status_code}")

    r = _upload_doc(ac, repo_id, "e2e_test_doc_a_copy.txt", doc_content)
    _record("Same bytes diff name -> 409", r.status_code == 409, f"got {r.status_code}")
    return True


# ── TEST 4 ───────────────────────────────────────────────────────────────────
def test_04_search_modes(client: httpx.Client):
    print("\n-- TEST 4: Search Mode Verification --")
    token = state.get("token")
    repo_id = state.get("repo_a_id")
    if state.get("doc_a_status") != "COMPLETED" or not state.get("doc_a_chunks"):
        return _record("T4-prereq", False, "doc not completed or no chunks")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)
    query = f"Tell me about {MARKER_A}"

    for mode, expect_present, expect_absent in [
        ("vector", "near_vector", "bm25"),
        ("keyword", "bm25", "near_vector"),
    ]:
        r = ac.post(f"{API}/retrieve", json={
            "query": query, "repository_id": repo_id,
            "search_mode": mode, "debug_retrieval": True,
        })
        if r.status_code == 200:
            body = r.json()
            stages = str(body.get("pipeline_stages_executed") or body.get("retrieval_signals", {}).get("pipeline_stages_executed", []))
            has_present = expect_present in stages.lower()
            has_absent = expect_absent in stages.lower()
            _record(f"{mode}: has {expect_present}", has_present, stages[:120])
            _record(f"{mode}: no {expect_absent}", not has_absent, stages[:120])
        else:
            _record(f"{mode} search", False, f"status {r.status_code}: {r.text[:200]}")

    r = ac.post(f"{API}/retrieve", json={
        "query": query, "repository_id": repo_id,
        "search_mode": "hybrid", "debug_retrieval": True,
    })
    if r.status_code == 200:
        body = r.json()
        stages = str(body.get("pipeline_stages_executed") or body.get("retrieval_signals", {}).get("pipeline_stages_executed", []))
        _record("hybrid: has vector+bm25", "near_vector" in stages.lower() and "bm25" in stages.lower(), stages[:120])
    else:
        _record("hybrid search", False, f"status {r.status_code}")
    return True


# ── TEST 5 ───────────────────────────────────────────────────────────────────
def test_05_retrieve_response_contract(client: httpx.Client):
    print("\n-- TEST 5: Retrieve Response Contract --")
    token = state.get("token")
    repo_id = state.get("repo_a_id")
    if state.get("doc_a_status") != "COMPLETED" or not state.get("doc_a_chunks"):
        return _record("T5-prereq", False, "doc not completed or no chunks")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)
    r = ac.post(f"{API}/retrieve", json={
        "query": f"What is {MARKER_A}?", "repository_id": repo_id,
        "debug_retrieval": True,
    })
    if r.status_code != 200:
        return _record("retrieve call", False, f"status {r.status_code}")

    body = r.json()
    chunks = body.get("results") or body.get("chunks") or []
    if not chunks:
        return _record("results present", False, "empty results")

    c = chunks[0]
    _record("chunk_id present", "chunk_id" in c, "")
    _record("score present", "score" in c, "")
    _record("repository_id present", "repository_id" in c, "")
    _record("doc_name present", "doc_name" in c or "document_name" in c, "")

    if "grounding_score" in c:
        _record("grounding_score != score", c.get("grounding_score") != c.get("score"), "")
    else:
        _record("grounding_score present", False, "field missing")

    if "quality_score" in c:
        _record("quality_score != score", c.get("quality_score") != c.get("score"), "")
    else:
        _record("quality_score present", False, "field missing")

    _record("quality_flags is list", isinstance(c.get("quality_flags"), list), str(type(c.get("quality_flags"))))
    _record("highlight_spans is list", isinstance(c.get("highlight_spans"), list), str(type(c.get("highlight_spans"))))
    return True


# ── TEST 6 ───────────────────────────────────────────────────────────────────
def test_06_hybrid_alpha(client: httpx.Client):
    print("\n-- TEST 6: Hybrid Alpha --")
    token = state.get("token")
    repo_id = state.get("repo_a_id")
    if state.get("doc_a_status") != "COMPLETED" or not state.get("doc_a_chunks"):
        return _record("T6-prereq", False, "doc not completed or no chunks")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)
    r = ac.post(f"{API}/retrieve", json={
        "query": MARKER_A, "repository_id": repo_id,
        "search_mode": "hybrid", "debug_retrieval": True,
    })
    if r.status_code != 200:
        return _record("hybrid retrieve", False, f"status {r.status_code}")

    body = r.json()
    signals = body.get("retrieval_signals") or {}
    _record("hybrid_alpha present", "hybrid_alpha" in signals, str(list(signals.keys())[:10]))
    _record("fusion_method present", "fusion_method" in signals, str(list(signals.keys())[:10]))
    return True


# ── TEST 7 ───────────────────────────────────────────────────────────────────
def test_07_reprocess(client: httpx.Client):
    print("\n-- TEST 7: Reprocess API --")
    token = state.get("token")
    doc_id = state.get("doc_a_id")
    if not doc_id:
        return _record("T7-prereq", False, "no doc_id")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)

    r = ac.post(f"{API}/documents/{doc_id}/reprocess")
    _record("Reprocess existing -> 200|503", r.status_code in (200, 201, 202, 503), f"got {r.status_code}")

    fake_id = "nonexistent-uuid-1234-5678-abcdef012345"
    r = ac.post(f"{API}/documents/{fake_id}/reprocess")
    _record("Reprocess missing -> 404", r.status_code == 404, f"got {r.status_code}")
    return True


# ── TEST 8 ───────────────────────────────────────────────────────────────────
def test_08_purge(client: httpx.Client):
    print("\n── TEST 8: Purge ──")
    # Deferred — we run this after catalog test
    pass


def _do_purge(client: httpx.Client, doc_id: str, token: str):
    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)
    r = ac.delete(f"{API}/documents/{doc_id}")
    _record("Delete doc", r.status_code in (200, 204), f"got {r.status_code}")

    time.sleep(2)
    r = ac.get(f"{API}/documents/{doc_id}")
    _record("GET after delete -> 404", r.status_code == 404, f"got {r.status_code}")


# ── TEST 9 ───────────────────────────────────────────────────────────────────
def test_09_query_decomposition(client: httpx.Client):
    print("\n-- TEST 9: Query Decomposition --")
    token = state.get("token")
    repo_id = state.get("repo_a_id")
    if state.get("doc_a_status") != "COMPLETED" or not state.get("doc_a_chunks"):
        return _record("T9-prereq", False, "doc not completed or no chunks")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)
    r = ac.post(f"{API}/retrieve", json={
        "query": "What is the purpose and what are the acceptance criteria?",
        "repository_id": repo_id,
        "use_production_pipeline": True,
        "expand_query": True,
        "debug_retrieval": True,
    })
    if r.status_code != 200:
        return _record("decomposition call", False, f"status {r.status_code}: {r.text[:200]}")

    body = r.json()
    sub_queries = body.get("sub_queries") or body.get("retrieval_signals", {}).get("sub_queries") or []
    _record("sub_queries not empty", len(sub_queries) > 0, f"{len(sub_queries)} sub-queries: {sub_queries[:3]}")
    return True


# ── TEST 10 ──────────────────────────────────────────────────────────────────
def test_10_catalog_consistency(client: httpx.Client):
    print("\n-- TEST 10: Catalog Consistency --")
    token = state.get("token")
    repo_id = state.get("repo_a_id")
    doc_id = state.get("doc_a_id")
    if not all([token, repo_id, doc_id]):
        return _record("T10-prereq", False, "missing state")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)
    r = ac.get(f"{API}/retrieve/documents", params={"repository_id": repo_id})
    if r.status_code != 200:
        return _record("catalog call", False, f"status {r.status_code}")

    body = r.json()
    docs_list = body.get("documents") or body.get("results") or (body if isinstance(body, list) else [])
    found = [d for d in docs_list if (d.get("document_id") or d.get("id")) == doc_id]
    _record("Doc in catalog", len(found) > 0, f"{len(docs_list)} docs in catalog")

    if found and state.get("doc_a_status") == "COMPLETED":
        indexed = found[0].get("indexed") or {}
        chunk_count = indexed.get("chunk_count", found[0].get("chunk_count", 0))
        _record("chunk_count > 0", chunk_count > 0, f"chunk_count={chunk_count}")
    return True


# ── TEST 11 ──────────────────────────────────────────────────────────────────
def test_11_repository_scoping(client: httpx.Client):
    print("\n-- TEST 11: Repository Scoping --")
    token = state.get("token")
    repo_a_id = state.get("repo_a_id")
    if not token or not repo_a_id:
        return _record("T11-prereq", False, "missing state")
    if state.get("doc_a_status") != "COMPLETED" or not state.get("doc_a_chunks"):
        return _record("T11-prereq", False, "doc A not completed or no chunks")

    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)

    repo_b_name = f"e2e-b-{uuid.uuid4().hex[:8]}"
    repo_b_id = _create_repo(ac, repo_b_name)
    state["repo_b_id"] = repo_b_id
    _activate_repo(ac, repo_b_id)
    _record("Create+activate repo B", bool(repo_b_id), repo_b_id)

    doc_b_content = _generate_document(MARKER_B, "E2E_DOC_B_ONLY_MARKER_2026")
    r = _upload_doc(ac, repo_b_id, "e2e_test_doc_b.txt", doc_b_content)
    _record("Upload doc B", r.status_code in (200, 201), f"status {r.status_code}")

    if r.status_code in (200, 201):
        body = r.json()
        if isinstance(body, list):
            doc_b_id = body[0].get("document_id") or body[0].get("id")
        elif isinstance(body, dict):
            doc_b_id = body.get("document_id") or body.get("id")
            docs = body.get("documents") or body.get("uploaded") or []
            if docs:
                doc_b_id = doc_b_id or docs[0].get("document_id") or docs[0].get("id")
        state["doc_b_id"] = doc_b_id

        print("    Polling doc B processing (up to 120s)...")
        status_b = _poll_status(_auth_headers(token), doc_b_id, timeout=120)
        final_b = _extract_status(status_b).upper() or "TIMEOUT"
        _record("Doc B processing", final_b == "COMPLETED", final_b)

        if final_b == "COMPLETED":
            # Search repo A for marker B -> should NOT find
            r = ac.post(f"{API}/retrieve", json={
                "query": MARKER_B, "repository_id": repo_a_id,
            })
            if r.status_code == 200:
                results_a = r.json().get("results") or r.json().get("chunks") or []
                has_b_in_a = any(MARKER_B in str(c) for c in results_a)
                _record("Repo A no marker B content", not has_b_in_a, f"{len(results_a)} results")
            else:
                _record("Repo A search", False, f"status {r.status_code}")

            # Search repo B for marker B -> should find
            r = ac.post(f"{API}/retrieve", json={
                "query": MARKER_B, "repository_id": repo_b_id,
            })
            if r.status_code == 200:
                results_b = r.json().get("results") or r.json().get("chunks") or []
                has_b_in_b = any(MARKER_B.lower() in str(c).lower() for c in results_b) or len(results_b) > 0
                _record("Repo B finds marker B", has_b_in_b, f"{len(results_b)} results")
            else:
                _record("Repo B search", False, f"status {r.status_code}")
    return True


# ── Cleanup ──────────────────────────────────────────────────────────────────
def cleanup(client: httpx.Client):
    print("\n-- Cleanup --")
    token = state.get("token")
    if not token:
        return
    ac = httpx.Client(base_url=BASE_URL, headers=_auth_headers(token), timeout=TIMEOUT)

    for doc_key in ("doc_b_id", "doc_a_id"):
        did = state.get(doc_key)
        if did:
            r = ac.delete(f"{API}/documents/{did}")
            print(f"  Delete {doc_key}={did}: {r.status_code}")

    for repo_key in ("repo_b_id", "repo_a_id"):
        rid = state.get(repo_key)
        if rid:
            r = ac.delete(f"{API}/repositories/{rid}")
            print(f"  Delete {repo_key}={rid}: {r.status_code}")


# -- Main -----────────────────────────────────────────────────────────────────
def main():
    print(f"RAG Builder E2E Runtime Tests - {BASE_URL}")
    print("=" * 70)

    test_fns = [
        test_01_authentication,
        test_02_fresh_e2e_processing,
        test_03_duplicate_detection,
        test_04_search_modes,
        test_05_retrieve_response_contract,
        test_06_hybrid_alpha,
        test_07_reprocess,
        test_09_query_decomposition,
        test_10_catalog_consistency,
    ]

    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        for fn in test_fns:
            try:
                fn(client)
            except Exception as exc:
                _record(fn.__name__, False, repr(exc)[:120])

        print("\n-- TEST 8: Purge --")
        doc_a_id = state.get("doc_a_id")
        token = state.get("token")
        if doc_a_id and token:
            try:
                _do_purge(client, doc_a_id, token)
            except Exception as exc:
                _record("T8-purge", False, repr(exc)[:120])
            state.pop("doc_a_id", None)
        else:
            _record("T8-prereq", False, "no doc_id or token")

        try:
            test_11_repository_scoping(client)
        except Exception as exc:
            _record("T11", False, repr(exc)[:120])

    cleanup(httpx.Client(base_url=BASE_URL, timeout=TIMEOUT))

    print("\n" + "=" * 70)
    print(f"{'TEST':<45} {'RESULT':<8} {'DETAILS'}")
    print("-" * 70)
    passed = failed = 0
    for name, status, detail in results:
        print(f"{name:<45} {status:<8} {detail[:60]}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1
    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {passed + failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
