from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.features.document_processing.core.logger import log
from src.features.references.extraction.reference_extractor import (
    is_same_document_id,
    normalize_document_id,
)


@dataclass(frozen=True)
class Neo4jReferenceConfig:
    uri: str | None
    user: str
    password: str
    database: str | None = None

    @classmethod
    def from_env(cls) -> "Neo4jReferenceConfig":
        auth_password = (os.getenv("NEO4J_AUTH", "neo4j/").split("/", 1)[-1] or "").strip()
        database = (os.getenv("NEO4J_DATABASE") or "").strip() or None
        return cls(
            uri=(os.getenv("NEO4J_URI") or os.getenv("NEO4J_BOLT_URI") or "").strip() or None,
            user=(os.getenv("NEO4J_USER") or "neo4j").strip(),
            password=(os.getenv("NEO4J_PASSWORD") or auth_password).strip(),
            database=database,
        )


class Neo4jReferenceStore:
    def __init__(self, config: Neo4jReferenceConfig | None = None) -> None:
        self.config = config or Neo4jReferenceConfig.from_env()

    @property
    def enabled(self) -> bool:
        return bool(self.config.uri and self.config.user and self.config.password)

    def _driver(self):
        if not self.enabled:
            raise RuntimeError("Neo4j reference store is disabled; set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.")
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j driver is not installed.") from exc
        return GraphDatabase.driver(self.config.uri, auth=(self.config.user, self.config.password))

    def _session_kwargs(self) -> dict[str, str]:
        return {"database": self.config.database} if self.config.database else {}

    def _find_exact_target_documents(
        self,
        session,
        normalized_target_id: str,
    ) -> list[dict[str, Any]]:
        if not normalized_target_id:
            return []
        return [
            dict(row)
            for row in session.run(
                """
                MATCH (target:Document {normalized_document_id: $normalized_target_id})
                RETURN target.document_id AS document_id,
                       coalesce(target.document_name, target.title) AS document_name,
                       target.normalized_document_id AS normalized_document_id
                """,
                normalized_target_id=normalized_target_id,
            )
        ]

    def _attempt_resolution(
        self,
        session,
        *,
        source_document_id: str,
        extracted_target_id: str,
        normalized_target_id: str,
    ) -> dict[str, Any] | None:
        source_normalized_id = normalize_document_id(source_document_id)
        matches = self._find_exact_target_documents(session, normalized_target_id)
        candidate = matches[0] if len(matches) == 1 else None
        log.info(
            "resolution_attempt source_document_id=%s source_normalized_id=%s "
            "extracted_target_id=%s normalized_target_id=%s "
            "candidate_target_document_id=%s candidate_target_normalized_id=%s candidate_count=%s",
            source_document_id,
            source_normalized_id,
            extracted_target_id,
            normalized_target_id,
            candidate.get("document_id") if candidate else None,
            candidate.get("normalized_document_id") if candidate else None,
            len(matches),
        )
        if not matches:
            log.info(
                "resolution_result=exact_match_not_found source_document_id=%s normalized_target_id=%s",
                source_document_id,
                normalized_target_id,
            )
            return None
        if len(matches) > 1:
            log.warning(
                "resolution_result=ambiguous_match source_document_id=%s normalized_target_id=%s candidate_count=%s",
                source_document_id,
                normalized_target_id,
                len(matches),
            )
            return None
        target = matches[0]
        candidate_normalized = str(target.get("normalized_document_id") or "")
        if candidate_normalized != normalized_target_id:
            log.info(
                "resolution_result=target_mismatch_skipped source_document_id=%s normalized_target_id=%s "
                "candidate_target_document_id=%s candidate_target_normalized_id=%s",
                source_document_id,
                normalized_target_id,
                target.get("document_id"),
                candidate_normalized,
            )
            return None
        if is_same_document_id(target.get("document_id"), source_document_id):
            log.info(
                "resolution_result=self_reference_skipped source_document_id=%s normalized_target_id=%s",
                source_document_id,
                normalized_target_id,
            )
            return None
        log.info(
            "resolution_result=exact_match source_document_id=%s normalized_target_id=%s "
            "candidate_target_document_id=%s candidate_target_normalized_id=%s",
            source_document_id,
            normalized_target_id,
            target.get("document_id"),
            candidate_normalized,
        )
        return target

    @staticmethod
    def _normalized_source_id(source_document_id: str, source_title: str | None = None) -> str:
        return normalize_document_id(source_title or source_document_id)

    @staticmethod
    def _target_matches_reference(normalized_target_id: str, target: dict[str, Any] | None) -> bool:
        if not target:
            return False
        return str(target.get("normalized_document_id") or "") == normalized_target_id

    def register_document(
        self,
        *,
        document_id: str,
        document_name: str | None = None,
        source_title: str | None = None,
        tenant_id: str | None = None,
        repository_id: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        resolved_document_name = document_name or source_title or document_id
        normalized_document_id = normalize_document_id(resolved_document_name or document_id)
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                session.run(
                    """
                    MERGE (doc:Document {normalized_document_id: $normalized_document_id})
                    SET doc.doc_id = $document_id,
                        doc.document_id = $document_id,
                        doc.document_name = $document_name,
                        doc.title = coalesce($document_name, doc.title),
                        doc.tenant_id = coalesce($tenant_id, doc.tenant_id),
                        doc.repository_id = coalesce($repository_id, doc.repository_id),
                        doc.updated_at = datetime()
                    """,
                    document_id=document_id,
                    document_name=resolved_document_name,
                    normalized_document_id=normalized_document_id,
                    tenant_id=tenant_id,
                    repository_id=repository_id,
                )
        finally:
            driver.close()

    def save_document_reference_graph(
        self,
        *,
        source_document_id: str,
        source_title: str | None,
        tenant_id: str | None,
        repository_id: str | None,
        references: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Persist pre-chunk document-level references and resolve when possible."""
        if not self.enabled:
            log.info("Neo4j reference extraction disabled; skipping %s references.", len(references))
            return {"saved": 0, "resolved": 0, "unresolved": 0}
        if not references:
            return {"saved": 0, "resolved": 0, "unresolved": 0}

        saved = 0
        resolved_count = 0
        unresolved_count = 0
        source_normalized_id = self._normalized_source_id(source_document_id, source_title)
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                session.run(
                    """
                    MERGE (src:Document {normalized_document_id: $source_normalized_id})
                    SET src.doc_id = $source_document_id,
                        src.document_id = $source_document_id,
                        src.document_name = coalesce($source_title, src.document_name),
                        src.title = coalesce($source_title, src.title),
                        src.tenant_id = coalesce($tenant_id, src.tenant_id),
                        src.repository_id = coalesce($repository_id, src.repository_id),
                        src.updated_at = datetime()
                    """,
                    source_document_id=source_document_id,
                    source_title=source_title,
                    source_normalized_id=source_normalized_id,
                    tenant_id=tenant_id,
                    repository_id=repository_id,
                )
                for reference in references:
                    extracted_target_id = str(
                        reference.get("extracted_target_id")
                        or reference.get("normalized_reference")
                        or reference.get("target_document_id")
                        or ""
                    ).strip()
                    normalized_target_id = normalize_document_id(
                        reference.get("normalized_target_id") or extracted_target_id
                    )
                    if not normalized_target_id:
                        continue

                    if is_same_document_id(normalized_target_id, source_document_id) or is_same_document_id(normalized_target_id, source_title):
                        log.info(
                            "self_reference_skipped source_document_id=%s extracted_target_id=%s source_sentence=%r",
                            source_document_id,
                            extracted_target_id,
                            reference.get("source_sentence"),
                        )
                        continue

                    resolved_target = self._attempt_resolution(
                        session,
                        source_document_id=source_document_id,
                        extracted_target_id=extracted_target_id,
                        normalized_target_id=normalized_target_id,
                    )
                    payload = build_document_reference_payload(
                        source_document_id=source_document_id,
                        reference=reference,
                        resolved_target=resolved_target,
                        tenant_id=tenant_id,
                        repository_id=repository_id,
                    )
                    log.info(
                        "reference_save source_document_id=%s extracted_target_id=%s normalized_target_id=%s "
                        "resolved_target_found=%s source_document_id_key=%s normalized_target_id_key=%s",
                        source_document_id,
                        payload["extracted_target_id"],
                        payload["normalized_target_id"],
                        payload["resolved"],
                        payload["source_document_id"],
                        payload["normalized_target_id"],
                    )
                    session.run(
                        """
                        MATCH (src:Document {normalized_document_id: $source_normalized_id})
                        MERGE (ref:Reference {
                            source_document_id: $source_document_id,
                            normalized_target_id: $normalized_target_id
                        })
                        SET ref.reference_key = $reference_key,
                            ref.reference_text = $reference_text,
                            ref.normalized_reference = $normalized_target_id,
                            ref.extracted_target_id = $extracted_target_id,
                            ref.trigger_phrase = $trigger_phrase,
                            ref.source_sentence = $source_sentence,
                            ref.source_scope = $source_scope,
                            ref.confidence = $confidence,
                            ref.resolved = $resolved,
                            ref.target_document_id = $resolved_target_document_id,
                            ref.target_normalized_document_id = $resolved_target_normalized_id,
                            ref.document_id = $source_document_id,
                            ref.tenant_id = $tenant_id,
                            ref.repository_id = $repository_id,
                            ref.extraction_method = $extraction_method,
                            ref.reference_type = $reference_type,
                            ref.evidence_sentences = $evidence_sentences,
                            ref.updated_at = datetime(),
                            ref.created_at = coalesce(ref.created_at, datetime())
                        MERGE (src)-[m:MENTIONS_REFERENCE]->(ref)
                        SET m.confidence = $confidence,
                            m.created_at = coalesce(m.created_at, datetime())
                        """,
                        source_normalized_id=source_normalized_id,
                        **payload,
                    )
                    if payload["create_refers_to"]:
                        if not self._target_matches_reference(
                            payload["normalized_target_id"],
                            {
                                "normalized_document_id": payload["resolved_target_normalized_id"],
                            },
                        ):
                            log.info(
                                "target_mismatch_skipped source_document_id=%s normalized_target_id=%s "
                                "candidate_target_normalized_id=%s",
                                source_document_id,
                                payload["normalized_target_id"],
                                payload["resolved_target_normalized_id"],
                            )
                            unresolved_count += 1
                            saved += 1
                            continue
                        session.run(
                            """
                            MATCH (source:Document {normalized_document_id: $source_normalized_id})
                            MATCH (ref:Reference {
                                source_document_id: $source_document_id,
                                normalized_target_id: $normalized_target_id
                            })
                            MATCH (target:Document {normalized_document_id: $normalized_target_id})
                            WHERE target.normalized_document_id = $normalized_target_id
                              AND ref.normalized_target_id = $normalized_target_id
                            MERGE (source)-[d:REFERS_TO]->(target)
                            SET d.reference_key = $reference_key,
                                d.reference_type = $reference_type,
                                d.confidence = $confidence,
                                d.created_at = coalesce(d.created_at, datetime())
                            MERGE (ref)-[:TARGETS]->(target)
                            SET ref.resolved = true,
                                ref.target_document_id = target.document_id,
                                ref.target_normalized_document_id = target.normalized_document_id,
                                ref.updated_at = datetime()
                            """,
                            source_normalized_id=source_normalized_id,
                            source_document_id=payload["source_document_id"],
                            reference_key=payload["reference_key"],
                            normalized_target_id=payload["normalized_target_id"],
                            reference_type=payload["reference_type"],
                            confidence=payload["confidence"],
                        )
                        resolved_count += 1
                        log.info(
                            "created_REFERS_TO source_document_id=%s extracted_target_id=%s "
                            "normalized_target_id=%s target_document_id=%s target_normalized_document_id=%s",
                            source_document_id,
                            payload["extracted_target_id"],
                            payload["normalized_target_id"],
                            payload["resolved_target_document_id"],
                            payload["resolved_target_normalized_id"],
                        )
                        log.info(
                            "created_TARGETS reference_key=%s target_document_id=%s target_normalized_document_id=%s",
                            payload["reference_key"],
                            payload["resolved_target_document_id"],
                            payload["resolved_target_normalized_id"],
                        )
                    else:
                        unresolved_count += 1
                        log.info(
                            "Reference unresolved: %s -> %s",
                            source_document_id,
                            payload["normalized_target_id"],
                        )
                    saved += 1
        finally:
            driver.close()
        log.info("Neo4j write successful: saved=%s resolved=%s unresolved=%s", saved, resolved_count, unresolved_count)
        return {"saved": saved, "resolved": resolved_count, "unresolved": unresolved_count}

    def resolve_pending_references(
        self,
        *,
        uploaded_document_id: str,
        uploaded_document_name: str | None = None,
        tenant_id: str | None = None,
        repository_id: str | None = None,
    ) -> int:
        """Resolve previously unresolved references when a target document is uploaded."""
        if not self.enabled:
            return 0
        normalized_uploaded = normalize_document_id(uploaded_document_name or uploaded_document_id)
        driver = self._driver()
        resolved = 0
        try:
            with driver.session(**self._session_kwargs()) as session:
                self.register_document(
                    document_id=uploaded_document_id,
                    document_name=uploaded_document_name,
                    tenant_id=tenant_id,
                    repository_id=repository_id,
                )
                rows = session.run(
                    """
                    MATCH (ref:Reference)
                    WHERE coalesce(ref.resolved, false) = false
                      AND ref.normalized_target_id = $normalized_uploaded
                    RETURN ref.source_document_id AS source_document_id,
                           ref.extracted_target_id AS extracted_target_id,
                           ref.normalized_target_id AS normalized_target_id,
                           ref.reference_type AS reference_type,
                           ref.confidence AS confidence
                    """,
                    normalized_uploaded=normalized_uploaded,
                )
                pending = list(rows)
                for row in pending:
                    source_document_id = row.get("source_document_id")
                    normalized_target_id = str(row.get("normalized_target_id") or "")
                    if not source_document_id or not normalized_target_id:
                        continue
                    if is_same_document_id(uploaded_document_id, source_document_id):
                        log.info(
                            "self_reference_skipped source_document_id=%s extracted_target_id=%s",
                            source_document_id,
                            uploaded_document_id,
                        )
                        continue
                    resolved_target = self._attempt_resolution(
                        session,
                        source_document_id=source_document_id,
                        extracted_target_id=str(row.get("extracted_target_id") or normalized_target_id),
                        normalized_target_id=normalized_target_id,
                    )
                    if not resolved_target or not self._target_matches_reference(
                        normalized_target_id, resolved_target
                    ):
                        log.info(
                            "target_mismatch_skipped source_document_id=%s normalized_target_id=%s "
                            "candidate_target_normalized_id=%s",
                            source_document_id,
                            normalized_target_id,
                            resolved_target.get("normalized_document_id") if resolved_target else None,
                        )
                        continue
                    source_normalized_id = self._normalized_source_id(source_document_id)
                    reference_key = f"{source_document_id}::{normalized_target_id}"
                    session.run(
                        """
                        MATCH (ref:Reference {
                            source_document_id: $source_document_id,
                            normalized_target_id: $normalized_target_id
                        })
                        MATCH (source:Document {normalized_document_id: $source_normalized_id})
                        MATCH (target:Document {normalized_document_id: $normalized_target_id})
                        WHERE target.normalized_document_id = $normalized_target_id
                          AND ref.normalized_target_id = $normalized_target_id
                        SET ref.resolved = true,
                            ref.target_document_id = target.document_id,
                            ref.target_normalized_document_id = target.normalized_document_id,
                            ref.updated_at = datetime()
                        MERGE (source)-[d:REFERS_TO]->(target)
                        SET d.reference_key = $reference_key,
                            d.reference_type = $reference_type,
                            d.confidence = $confidence,
                            d.created_at = coalesce(d.created_at, datetime())
                        MERGE (ref)-[:TARGETS]->(target)
                        """,
                        source_document_id=source_document_id,
                        reference_key=reference_key,
                        source_normalized_id=source_normalized_id,
                        normalized_target_id=normalized_target_id,
                        reference_type=row.get("reference_type"),
                        confidence=float(row.get("confidence") or 0.0),
                    )
                    resolved += 1
                    log.info(
                        "Automatic resolution executed: %s -> %s via source_document_id=%s normalized_target_id=%s",
                        source_document_id,
                        uploaded_document_id,
                        source_document_id,
                        normalized_target_id,
                    )
        finally:
            driver.close()
        return resolved

    def debug_references_for(self, document_id: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (src:Document)
        WHERE src.doc_id = $document_id OR src.document_id = $document_id
        MATCH (src)-[:MENTIONS_REFERENCE]->(ref:Reference)
        RETURN ref.extracted_target_id AS extracted_target_id,
               ref.normalized_target_id AS normalized_target_id,
               ref.source_document_id AS source_document_id,
               ref.reference_text AS reference_text,
               ref.source_sentence AS source_sentence,
               coalesce(ref.resolved, false) AS resolved,
               ref.target_document_id AS target_document_id,
               ref.target_normalized_document_id AS target_normalized_document_id
        ORDER BY ref.normalized_target_id
        """
        return self._read_rows(cypher, document_id=document_id)

    def verify_reference_invariant(self) -> list[dict[str, Any]]:
        """Return resolved references where normalized_target_id != target.normalized_document_id."""
        cypher = """
        MATCH (s:Document)-[:MENTIONS_REFERENCE]->(r:Reference)-[:TARGETS]->(t:Document)
        WHERE coalesce(r.resolved, false) = true
          AND r.normalized_target_id <> t.normalized_document_id
        RETURN s.document_id AS source_document_id,
               r.extracted_target_id AS extracted_target_id,
               r.normalized_target_id AS normalized_target_id,
               t.document_id AS target_document_id,
               t.normalized_document_id AS target_normalized_document_id
        """
        if not self.enabled:
            return []
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                return [dict(row) for row in session.run(cypher)]
        finally:
            driver.close()

    def reference_resolution_rows(self) -> list[dict[str, Any]]:
        cypher = """
        MATCH (s:Document)-[:MENTIONS_REFERENCE]->(r:Reference)
        OPTIONAL MATCH (r)-[:TARGETS]->(t:Document)
        RETURN s.document_id AS source_document_id,
               r.extracted_target_id AS extracted_target_id,
               r.normalized_target_id AS normalized_target_id,
               t.document_id AS target_document_id,
               t.normalized_document_id AS target_normalized_document_id,
               coalesce(r.resolved, false) AS resolved
        ORDER BY s.document_id, r.normalized_target_id
        """
        if not self.enabled:
            return []
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                return [dict(row) for row in session.run(cypher)]
        finally:
            driver.close()

    def debug_resolve_target(self, target_id: str) -> dict[str, Any]:
        normalized_target_id = normalize_document_id(target_id)
        exact_matches = self._find_exact_target_documents_from_store(normalized_target_id)
        return {
            "input_target_id": target_id,
            "normalized_target_id": normalized_target_id,
            "exact_matches": exact_matches,
            "would_resolve": len(exact_matches) == 1,
        }

    def _find_exact_target_documents_from_store(self, normalized_target_id: str) -> list[dict[str, Any]]:
        if not self.enabled or not normalized_target_id:
            return []
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                return self._find_exact_target_documents(session, normalized_target_id)
        finally:
            driver.close()

    def references_for(self, document_id: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (src:Document {doc_id: $document_id})-[:REFERS_TO]->(target:Document)
        RETURN target.doc_id AS document_id,
               target.title AS title,
               'REFERS_TO' AS relationship,
               null AS matched_text,
               null AS reference_type,
               null AS extraction_method,
               1.0 AS confidence,
               false AS unresolved,
               null AS created_at
        UNION
        MATCH (src:Document {doc_id: $document_id})-[:MENTIONS_REFERENCE]->(ref:Reference)
        WHERE coalesce(ref.resolved, false) = false
        RETURN ref.extracted_target_id AS document_id,
               null AS title,
               'MENTIONS_REFERENCE' AS relationship,
               ref.reference_text AS matched_text,
               ref.reference_type AS reference_type,
               ref.extraction_method AS extraction_method,
               ref.confidence AS confidence,
               true AS unresolved,
               ref.created_at AS created_at
        ORDER BY document_id
        """
        return self._read_rows(cypher, document_id=document_id)

    def referenced_by(self, document_id: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (src:Document)-[:REFERS_TO]->(target:Document {doc_id: $document_id})
        RETURN src.doc_id AS document_id,
               src.title AS title,
               'REFERS_TO' AS relationship,
               null AS matched_text,
               null AS reference_type,
               null AS extraction_method,
               1.0 AS confidence,
               false AS unresolved,
               null AS created_at
        ORDER BY src.doc_id
        """
        return self._read_rows(cypher, document_id=document_id)

    def reference_graph(self, document_id: str) -> dict[str, list[dict[str, Any]]]:
        if not self.enabled:
            return {"nodes": [], "edges": []}
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                for row in session.run(
                    """
                    MATCH (center:Document {doc_id: $document_id})
                    OPTIONAL MATCH (center)-[:REFERS_TO]->(other:Document)
                    RETURN center.doc_id AS center_id,
                           center.title AS center_title,
                           other.doc_id AS other_id,
                           other.title AS other_title,
                           'REFERS_TO' AS edge_type,
                           center.doc_id AS source_id,
                           other.doc_id AS target_id
                    """,
                    document_id=document_id,
                ):
                    center_id = row.get("center_id")
                    other_id = row.get("other_id")
                    if center_id:
                        nodes.setdefault(
                            center_id,
                            {"id": center_id, "label": "Document", "title": row.get("center_title")},
                        )
                    if other_id:
                        nodes.setdefault(
                            other_id,
                            {"id": other_id, "label": "Document", "title": row.get("other_title")},
                        )
                        edges.append(
                            {
                                "source": row.get("source_id"),
                                "target": row.get("target_id"),
                                "type": "REFERS_TO",
                            }
                        )

                for row in session.run(
                    """
                    MATCH (src:Document)
                    WHERE src.doc_id = $document_id OR src.document_id = $document_id
                    MATCH (src)-[m:MENTIONS_REFERENCE]->(ref:Reference)
                    OPTIONAL MATCH (ref)-[:TARGETS]->(target:Document)
                    RETURN src.doc_id AS source_id,
                           ref.source_document_id AS source_document_id,
                           ref.normalized_target_id AS normalized_target_id,
                           ref.reference_text AS reference_text,
                           ref.resolved AS resolved,
                           target.doc_id AS target_id
                    """,
                    document_id=document_id,
                ):
                    ref_id = (
                        f"{row.get('source_document_id')}::{row.get('normalized_target_id')}"
                        if row.get("source_document_id") and row.get("normalized_target_id")
                        else None
                    )
                    source_id = row.get("source_id")
                    if ref_id:
                        nodes.setdefault(
                            ref_id,
                            {
                                "id": ref_id,
                                "label": "Reference",
                                "reference_text": row.get("reference_text"),
                                "resolved": row.get("resolved"),
                            },
                        )
                    if source_id and ref_id:
                        edges.append(
                            {
                                "source": source_id,
                                "target": ref_id,
                                "type": "MENTIONS_REFERENCE",
                            }
                        )
                    target_id = row.get("target_id")
                    if ref_id and target_id:
                        nodes.setdefault(target_id, {"id": target_id, "label": "Document"})
                        edges.append(
                            {
                                "source": ref_id,
                                "target": target_id,
                                "type": "TARGETS",
                            }
                        )
        finally:
            driver.close()
        return {"nodes": list(nodes.values()), "edges": edges}

    def _read_rows(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        driver = self._driver()
        try:
            with driver.session(**self._session_kwargs()) as session:
                return [_coerce_row(dict(row)) for row in session.run(cypher, **params)]
        finally:
            driver.close()


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(row)
    created_at = coerced.get("created_at")
    if created_at is not None:
        coerced["created_at"] = str(created_at)
    return coerced


def get_reference_store() -> Neo4jReferenceStore:
    return Neo4jReferenceStore()


def build_document_reference_payload(
    *,
    source_document_id: str,
    reference: dict[str, Any],
    resolved_target: dict[str, Any] | None,
    tenant_id: str | None,
    repository_id: str | None,
) -> dict[str, Any]:
    extracted_target_id = str(
        reference.get("extracted_target_id")
        or reference.get("reference_text")
        or reference.get("target_document_id")
        or ""
    ).strip()
    normalized_target_id = normalize_document_id(
        reference.get("normalized_target_id")
        or reference.get("normalized_reference")
        or extracted_target_id
    )
    reference_key = f"{source_document_id}::{normalized_target_id}"
    evidence = reference.get("evidence_sentences") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    resolved_target_document_id = None
    resolved_target_normalized_id = None
    if resolved_target:
        candidate_normalized = str(resolved_target.get("normalized_document_id") or "")
        if candidate_normalized == normalized_target_id:
            resolved_target_document_id = resolved_target.get("document_id")
            resolved_target_normalized_id = candidate_normalized

    resolved = bool(resolved_target_document_id and resolved_target_normalized_id)
    if resolved and is_same_document_id(resolved_target_document_id, source_document_id):
        resolved = False
        resolved_target_document_id = None
        resolved_target_normalized_id = None

    return {
        "source_document_id": source_document_id,
        "reference_key": reference_key,
        "reference_text": reference.get("reference_text"),
        "normalized_target_id": normalized_target_id,
        "extracted_target_id": extracted_target_id or normalized_target_id,
        "trigger_phrase": reference.get("trigger_phrase"),
        "source_sentence": reference.get("source_sentence"),
        "source_scope": reference.get("source_scope", "document"),
        "confidence": float(reference.get("confidence", 0.0)),
        "resolved": resolved,
        "resolved_target_document_id": resolved_target_document_id,
        "resolved_target_normalized_id": resolved_target_normalized_id,
        "tenant_id": tenant_id,
        "repository_id": repository_id,
        "extraction_method": reference.get("extraction_method", "pre_chunk_keyword_regex"),
        "reference_type": reference.get("reference_type", "document"),
        "evidence_sentences": evidence,
        "create_refers_to": resolved,
    }
