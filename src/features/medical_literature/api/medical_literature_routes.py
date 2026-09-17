"""HTTP routes for repository-scoped external evidence + PubMed agent."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.application.consumer_api.security import security_scheme
from src.features.medical_literature.domain.literature_exceptions import EvidenceError
from src.features.medical_literature.schemas.literature_schemas import (
    ClinicalTrialsSearchRequest,
    EvidenceImportRequest,
    EvidenceImportResponse,
    EvidenceRegistryListResponse,
    EvidenceSearchResponse,
    PubmedAgentRequest,
    PubmedAgentResponse,
    PubMedSearchRequest,
)
from src.features.medical_literature.application.literature_research_service import get_evidence_service

router = APIRouter(
    prefix="/repositories",
    tags=["External Evidence"],
    dependencies=[Depends(security_scheme)],
)

pubmed_router = APIRouter(
    prefix="/repositories",
    tags=["PubMed AI Agent"],
    dependencies=[Depends(security_scheme)],
)


def _require_authenticated_actor():
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.authorization.application.authorization_service import AuthenticatedUser
    from src.features.authentication.domain.authentication_exceptions import AuthenticationError

    current = get_current_user_from_context()
    if current is None or current.auth_method not in {"jwt", "api_key"} or current.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    return AuthenticatedUser(
        user_id=current.user_id,
        platform_role=current.platform_role,
        auth_method=current.auth_method,
        must_change_password=current.must_change_password,
        is_admin_api_key=current.is_admin_api_key,
        is_consumer_api_key=current.is_consumer_api_key,
    )


def _require_valid_repository_id(repository_id: str) -> str:
    from src.features.repositories.domain.repository_exceptions import ValidationError
    from src.features.repositories.domain.repository import parse_repository_id

    try:
        return parse_repository_id(repository_id)
    except Exception as exc:
        err = ValidationError(
            "repository_id must be a valid UUID.",
            details={"repository_id": repository_id, "reason": "invalid_uuid"},
        )
        err.code = "validation_error"
        err.http_status = 422
        raise err from exc


def _evidence_http_error(exc: EvidenceError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@router.post(
    "/{repository_id}/evidence/pubmed/search",
    response_model=EvidenceSearchResponse,
    summary="Search PubMed evidence (preview only)",
)
async def search_pubmed(repository_id: str, body: PubMedSearchRequest) -> EvidenceSearchResponse:
    from src.features.users.application.user_service import get_platform_security_service

    actor = _require_authenticated_actor()
    repository_id = _require_valid_repository_id(repository_id)
    # Search requires upload-capable role (contributor+) — human will select for import.
    get_platform_security_service().check_upload_access(actor, repository_id)
    try:
        return get_evidence_service().search_pubmed(repository_id=repository_id, body=body)
    except EvidenceError as exc:
        raise _evidence_http_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={
                "code": "external_provider_error",
                "message": "PubMed search failed",
                "details": {"reason": type(exc).__name__, "error": str(exc)[:300]},
            },
        ) from exc


@router.post(
    "/{repository_id}/evidence/clinical-trials/search",
    response_model=EvidenceSearchResponse,
    summary="Search ClinicalTrials.gov evidence (preview only)",
)
async def search_clinical_trials(
    repository_id: str,
    body: ClinicalTrialsSearchRequest,
) -> EvidenceSearchResponse:
    from src.features.users.application.user_service import get_platform_security_service

    actor = _require_authenticated_actor()
    repository_id = _require_valid_repository_id(repository_id)
    get_platform_security_service().check_upload_access(actor, repository_id)
    try:
        return get_evidence_service().search_clinical_trials(repository_id=repository_id, body=body)
    except EvidenceError as exc:
        raise _evidence_http_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={
                "code": "external_provider_error",
                "message": "ClinicalTrials.gov search failed",
                "details": {"reason": type(exc).__name__, "error": str(exc)[:300]},
            },
        ) from exc


@router.post(
    "/{repository_id}/evidence/import",
    response_model=EvidenceImportResponse,
    summary="Import human-selected external evidence into the standard document pipeline",
)
async def import_evidence(repository_id: str, body: EvidenceImportRequest) -> EvidenceImportResponse:
    from src.features.users.application.user_service import get_platform_security_service

    actor = _require_authenticated_actor()
    repository_id = _require_valid_repository_id(repository_id)
    get_platform_security_service().check_upload_access(actor, repository_id)
    try:
        return await get_evidence_service().import_selected(repository_id=repository_id, body=body)
    except EvidenceError as exc:
        raise _evidence_http_error(exc) from exc


@router.get(
    "/{repository_id}/evidence/imports",
    response_model=EvidenceRegistryListResponse,
    summary="List imported external evidence for a repository",
)
async def list_evidence_imports(
    repository_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> EvidenceRegistryListResponse:
    from src.features.users.application.user_service import get_platform_security_service

    actor = _require_authenticated_actor()
    repository_id = _require_valid_repository_id(repository_id)
    get_platform_security_service().check_retrieval_access(actor, repository_id)
    try:
        return get_evidence_service().list_imports(
            repository_id=repository_id,
            limit=limit,
            offset=offset,
        )
    except EvidenceError as exc:
        raise _evidence_http_error(exc) from exc


@pubmed_router.post(
    "/{repository_id}/pubmed-agent/chat",
    response_model=PubmedAgentResponse,
    summary="PubMed AI Agent — grounded chat with [REPO:DOCUMENT:PAGE] citations",
)
async def pubmed_agent_chat(repository_id: str, body: PubmedAgentRequest) -> PubmedAgentResponse:
    from src.features.medical_literature.agents.medical_literature_agent import PubmedAgentService
    from src.features.authentication.domain.authentication_exceptions import PlatformSecurityError
    from src.features.users.application.user_service import get_platform_security_service
    from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
    from src.features.retrieval.application.retrieval_service import get_retrieval_service

    actor = _require_authenticated_actor()
    repository_id = _require_valid_repository_id(repository_id)
    try:
        get_platform_security_service().check_retrieval_access(actor, repository_id)
    except PlatformSecurityError:
        raise

    # Prompt / safety guards (reuse chat stack)
    from src.features.security.moderation.hybrid_prompt_guard import HybridPromptGuard
    from src.features.security.dlp.policy_loader import scan_keyword_policies

    pg = HybridPromptGuard()
    pg_res = pg.check_prompt(body.query, pipeline="pubmed_agent")
    if not pg_res.get("allowed", True):
        return PubmedAgentResponse(
            allowed=False,
            safe=True,
            risk_type=pg_res.get("risk_type"),
            severity=str(pg_res.get("severity") or "high"),
            reason=pg_res.get("reason"),
            safe_message=pg_res.get("safe_message") or "Request blocked by security policy.",
            action="blocked",
            answer=pg_res.get("safe_message") or "Request blocked by security policy.",
            repository_id=repository_id,
            evidence_sufficient=False,
            grounded=True,
        )
    kw = scan_keyword_policies(body.query)
    if kw.get("action") == "block":
        return PubmedAgentResponse(
            allowed=False,
            safe=True,
            risk_type="blacklist",
            severity="high",
            reason="Prompt blocked by keyword policy.",
            safe_message="Request blocked by security policy.",
            action="blocked",
            answer="Request blocked by security policy.",
            repository_id=repository_id,
            evidence_sufficient=False,
            grounded=True,
        )

    retrieve_kwargs: dict[str, Any] = {
        "query": body.query,
        "repository_id": repository_id,
    }
    if body.top_k is not None:
        retrieve_kwargs["top_k"] = body.top_k
    if body.reranking_enabled is not None:
        retrieve_kwargs["use_rerank"] = body.reranking_enabled
    if body.query_expansion_enabled is not None:
        retrieve_kwargs["expand_query"] = body.query_expansion_enabled
    if body.parallel_retrieval_enabled is not None:
        # Parallel retrieval maps to production hybrid pipeline when enabled.
        retrieve_kwargs["use_production_pipeline"] = body.parallel_retrieval_enabled

    # Inherit repository settings when request omits overrides.
    try:
        from src.features.repositories.infrastructure.repository_repository import get_repository_store

        store = get_repository_store()
        if store is not None:
            settings = store.get_settings(repository_id) or {}
            if body.reranking_enabled is None and settings.get("reranking") is not None:
                retrieve_kwargs["use_rerank"] = bool(settings.get("reranking"))
            if body.top_k is None and settings.get("retrieval_top_k") is not None:
                retrieve_kwargs["top_k"] = int(settings["retrieval_top_k"])
            if body.query_expansion_enabled is None and settings.get("query_expansion_enabled") is not None:
                retrieve_kwargs["expand_query"] = bool(settings.get("query_expansion_enabled"))
            if body.parallel_retrieval_enabled is None and settings.get("parallel_retrieval_enabled") is not None:
                retrieve_kwargs["use_production_pipeline"] = bool(
                    settings.get("parallel_retrieval_enabled")
                )
            mode = settings.get("retrieval_search_mode")
            if mode:
                retrieve_kwargs["search_mode"] = mode
    except Exception:
        pass

    retrieve_req = RetrieveRequest(**retrieve_kwargs)
    try:
        ret_res = await get_retrieval_service().retrieve(retrieve_req)
    except Exception:
        return PubmedAgentResponse(
            allowed=True,
            answer="Retrieval is temporarily unavailable. Please try again shortly.",
            citations=[],
            grounded=True,
            evidence_sufficient=False,
            repository_id=repository_id,
            action="refuse",
            reason="retrieval_unavailable",
        )

    if getattr(ret_res, "blocked", False):
        return PubmedAgentResponse(
            allowed=False,
            safe=True,
            risk_type="jailbreak",
            severity="high",
            reason=str(ret_res.block_reason or "prompt blocked"),
            safe_message="Request blocked by security policy.",
            action="blocked",
            answer="Request blocked by security policy.",
            repository_id=repository_id,
            evidence_sufficient=False,
            grounded=True,
        )

    chunks = list(ret_res.results or [])
    from src.features.security.dlp.sensitive_data_detector import mask_text_content

    for chunk in chunks:
        if getattr(chunk, "text", None):
            chunk.text = mask_text_content(chunk.text)

    return await PubmedAgentService().answer(
        repository_id=repository_id,
        body=body,
        chunks=chunks,
    )
