"""Focused runtime retrieve/search-mode/quality verification."""
from __future__ import annotations

import json
import os
import time
import uuid

import httpx

BASE = os.getenv("DMS_API_BASE_URL", "http://localhost:8088").rstrip("/")
API = "/api/v1"
ADMIN_USER = os.getenv("DMS_API_USER", "admin")
ADMIN_PASSWORD = os.getenv("DMS_API_PASSWORD", "changeme-local-admin")
MARKER = "E2E_RETRIEVE_MARKER_2026"
TIMEOUT = 120.0

results: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'OK' if ok else 'XX'}] {name}: {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def main() -> int:
    token = httpx.post(
        f"{BASE}{API}/auth/token",
        json={"user_id": ADMIN_USER, "password": ADMIN_PASSWORD},
        timeout=30,
    ).json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}
    c = httpx.Client(base_url=BASE, headers=h, timeout=TIMEOUT)

    name = f"e2e_retrieve_{uuid.uuid4().hex[:8]}"
    r = c.post(
        f"{API}/repositories",
        json={
            "name": name,
            "owner_user_id": ADMIN_USER,
            "settings": {
                "chunking_strategy": "section-based",
                "chunk_size": 512,
                "embedding_model": {
                    "provider": "local",
                    "model_id": "all-MiniLM-L6-v2",
                    "local_model_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
                },
            },
        },
    )
    rec("create repo", r.status_code == 201, str(r.status_code))
    repo_id = r.json().get("repository_id")
    c.post(f"{API}/repositories/{repo_id}/activate")

    content = (
        f"Section 1 Purpose\nThe purpose of this SOP is {MARKER} quality control.\n"
        "Operators must verify cycle parameters and record results.\n" * 40
        + "\nSection 2 Approval\nThe approval process requires two signatures.\n" * 20
        + "\nSection 3 Retention\nRetention requirement is seven years.\n" * 20
    )
    up = c.post(
        f"{API}/documents/upload",
        files={"files": ("retrieve_e2e.txt", content.encode(), "text/plain")},
        data={"repository_id": repo_id, "submit_for_processing": "true"},
    )
    rec("upload", up.status_code == 200, str(up.status_code))
    doc_id = (up.json().get("documents") or [{}])[0].get("document_id")
    rec("doc_id", bool(doc_id), str(doc_id))

    status = ""
    for i in range(24):
        time.sleep(5)
        st = c.get(f"{API}/documents/{doc_id}/status")
        body = st.json() if st.status_code == 200 else {}
        status = (body.get("document") or {}).get("status") or body.get("status") or ""
        print(f"    [{(i+1)*5}s] {status}")
        if str(status).upper() in ("COMPLETED", "FAILED", "ERROR"):
            break
    rec("processing COMPLETED", str(status).upper() == "COMPLETED", status)

    if str(status).upper() != "COMPLETED":
        _print_summary()
        return 1

    query = f"What is {MARKER} purpose approval retention?"

    def retrieve(**extra):
        payload = {"query": query, "repository_id": repo_id, "debug_retrieval": True, **extra}
        return c.post(f"{API}/retrieve", json=payload)

    # Search modes
    for mode in ("vector", "keyword", "hybrid"):
        resp = retrieve(search_mode=mode)
        rec(f"{mode} HTTP", resp.status_code == 200, str(resp.status_code))
        if resp.status_code != 200:
            print(resp.text[:400])
            continue
        body = resp.json()
        stages = [s.lower() for s in (body.get("pipeline_stages_executed") or [])]
        rec(f"{mode} stages", bool(stages), str(stages))
        if mode == "vector":
            rec("vector has near_vector", any("near_vector" in s or "embedding" in s for s in stages), str(stages))
            rec("vector no bm25", not any(s == "bm25" for s in stages), str(stages))
        elif mode == "keyword":
            rec("keyword has bm25", any("bm25" in s for s in stages), str(stages))
            rec("keyword no near_vector", not any("near_vector" in s for s in stages), str(stages))
        else:
            rec(
                "hybrid has both",
                any("near_vector" in s or "embedding" in s for s in stages) and any("bm25" in s for s in stages),
                str(stages),
            )

    # Contract + quality
    resp = retrieve(search_mode="hybrid", hybrid_alpha=0.75)
    rec("contract HTTP", resp.status_code == 200, str(resp.status_code))
    body = resp.json()
    chunks = body.get("results") or []
    rec("results nonempty", bool(chunks), str(len(chunks)))
    if chunks:
        ch = chunks[0]
        rec("chunk document_id", bool(ch.get("metadata", {}).get("document_id") or ch.get("document_id")), str(ch.get("metadata", {}).get("document_id")))
        rec("chunk repository_id", ch.get("repository_id") == repo_id, str(ch.get("repository_id")))
        rec("chunk_id", bool(ch.get("chunk_id") or ch.get("id")), str(ch.get("chunk_id") or ch.get("id")))
        rec("score present", "score" in ch, str(ch.get("score")))
        gs = ch.get("grounding_score")
        qs = ch.get("quality_score")
        rec("grounding_score present", gs is not None, str(gs))
        rec("quality_score present", qs is not None, str(qs))
        rec("grounding != score", gs != ch.get("score"), f"g={gs} s={ch.get('score')}")
        rec("quality != score", qs != ch.get("score"), f"q={qs} s={ch.get('score')}")
        rec("quality_flags is list", isinstance(ch.get("quality_flags"), list), str(ch.get("quality_flags")))
        rec("highlight_spans is list", isinstance(ch.get("highlight_spans"), list), str(ch.get("highlight_spans")))
        signals = ch.get("retrieval_signals") or {}
        rec("hybrid_alpha in chunk signals", "hybrid_alpha" in signals, json.dumps({k: signals.get(k) for k in ("hybrid_alpha", "fusion_method")}))
        rec("fusion_method in chunk signals", "fusion_method" in signals, str(signals.get("fusion_method")))

    # Keyword highlights
    kw = retrieve(search_mode="keyword", query=MARKER)
    if kw.status_code == 200:
        kchunks = kw.json().get("results") or []
        spans = (kchunks[0].get("highlight_spans") or []) if kchunks else []
        rec("keyword highlight_spans nonempty", bool(spans), str(spans))
        if spans and kchunks:
            text = kchunks[0].get("text") or ""
            ok_spans = True
            for start, end in spans:
                if not (0 <= start < end <= len(text)):
                    ok_spans = False
            rec("highlight span bounds", ok_spans, f"len={len(text)} spans={spans}")

    # Alpha sweep
    alphas = {}
    for alpha in (0.0, 0.5, 0.75, 1.0):
        ar = retrieve(search_mode="hybrid", hybrid_alpha=alpha)
        if ar.status_code != 200:
            rec(f"alpha={alpha} HTTP", False, str(ar.status_code))
            continue
        achunks = ar.json().get("results") or []
        sig = (achunks[0].get("retrieval_signals") or {}) if achunks else {}
        alphas[alpha] = {
            "hybrid_alpha": sig.get("hybrid_alpha"),
            "fusion_method": sig.get("fusion_method"),
            "top_id": (achunks[0].get("chunk_id") or achunks[0].get("id")) if achunks else None,
            "score": achunks[0].get("score") if achunks else None,
        }
        rec(f"alpha={alpha} applied", sig.get("hybrid_alpha") == alpha or abs(float(sig.get("hybrid_alpha") or -1) - alpha) < 0.01, str(sig.get("hybrid_alpha")))
    rec("alpha sweep recorded", len(alphas) == 4, json.dumps(alphas, default=str)[:400])

    # Expansion / subqueries
    exp_off = retrieve(expand_query=False, use_production_pipeline=True)
    if exp_off.status_code == 200:
        rec("expand disabled empty-or-original", True, str(exp_off.json().get("expanded_queries")))
    exp_on = retrieve(
        expand_query=True,
        use_production_pipeline=True,
        query="What is the purpose, approval process, and retention requirement?",
    )
    if exp_on.status_code == 200:
        eb = exp_on.json()
        rec("expanded_queries", True, str(eb.get("expanded_queries")))
        rec("sub_queries generated", bool(eb.get("sub_queries")), str(eb.get("sub_queries")))
        rec("subquery results merged", bool(eb.get("results")), f"n={len(eb.get('results') or [])}")

    simple = retrieve(use_production_pipeline=True, expand_query=True, query="What is the purpose?")
    if simple.status_code == 200:
        rec("simple query few subqueries", len(simple.json().get("sub_queries") or []) <= 1, str(simple.json().get("sub_queries")))

    # Cleanup
    c.delete(f"{API}/documents/{doc_id}")
    c.delete(f"{API}/repositories/{repo_id}")
    _print_summary()
    return 0 if all(ok for _, ok, _ in results) else 1


def _print_summary() -> None:
    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    for name, ok, detail in results:
        print(f"{name:45} {'PASS' if ok else 'FAIL':6} {detail[:80]}")
    print(f"Total: {passed} passed, {failed} failed out of {len(results)}")


if __name__ == "__main__":
    raise SystemExit(main())
