"""YAML-backed blacklist/allowlist/DLP policy loader with hot reload."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from flashtext import KeywordProcessor  # type: ignore
except Exception:  # pragma: no cover
    KeywordProcessor = None


def _default_config_dir() -> Path:
    env = (os.getenv("SECURITY_POLICY_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # rag-builder/configs/security — this file is src/features/security/dlp/
    return Path(__file__).resolve().parents[4] / "configs" / "security"


class SecurityPolicyConfigurationError(RuntimeError):
    """Raised when required security policy files are missing/invalid."""


_REQUIRED_POLICY_FILES = (
    "blacklist.yaml",
    "allowlist.yaml",
    "dlp_policies.yaml",
    "security_policy.yaml",
)


def _strict_policy_validation_enabled() -> bool:
    raw = (os.getenv("SECURITY_POLICY_STRICT") or "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return True


def _validate_policy_files(config_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    file_status: dict[str, bool] = {}
    for filename in _REQUIRED_POLICY_FILES:
        exists = (config_dir / filename).is_file()
        file_status[filename] = exists
        if not exists:
            missing.append(str(config_dir / filename))

    if missing and _strict_policy_validation_enabled():
        raise SecurityPolicyConfigurationError(
            "Missing required security policy files: "
            + ", ".join(missing)
            + ". Either mount the configs directory correctly or set SECURITY_POLICY_DIR."
        )

    if missing:
        logger.error(
            "Security policy files missing with strict mode disabled: %s",
            ", ".join(missing),
        )

    return {
        "config_dir": str(config_dir),
        "strict_mode": _strict_policy_validation_enabled(),
        "files": file_status,
        "missing_files": missing,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        # Minimal fallback: no nested structures beyond simple lists under keys.
        data: dict[str, Any] = {}
        current: str | None = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current = line[:-1].strip()
                data[current] = []
                continue
            if current and line.strip().startswith("- "):
                data[current].append(line.strip()[2:].strip().strip("'\""))
        return data
    loaded = yaml.safe_load(text) or {}
    return loaded if isinstance(loaded, dict) else {}


class KeywordPolicyEngine:
    """FlashText (or regex fallback) matcher with allowlist override."""

    def __init__(self, config_dir: Path | None = None, *, reload_seconds: float = 30.0) -> None:
        self.config_dir = config_dir or _default_config_dir()
        self.reload_seconds = reload_seconds
        self._lock = threading.RLock()
        self._loaded_at = 0.0
        self._blacklist: dict[str, list[str]] = {}
        self._allowlist: dict[str, list[str]] = {}
        self._dlp_policies: dict[str, Any] = {}
        self._security_policy: dict[str, Any] = {}
        self._processor = None
        self._category_by_term: dict[str, str] = {}
        self._last_summary: dict[str, Any] = {}
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if not force and (now - self._loaded_at) < self.reload_seconds:
                return
            validation = _validate_policy_files(self.config_dir)
            self._blacklist = _load_yaml(self.config_dir / "blacklist.yaml")
            self._allowlist = _load_yaml(self.config_dir / "allowlist.yaml")
            self._dlp_policies = _load_yaml(self.config_dir / "dlp_policies.yaml")
            self._security_policy = _load_yaml(self.config_dir / "security_policy.yaml")
            self._category_by_term = {}
            if KeywordProcessor is not None:
                proc = KeywordProcessor(case_sensitive=False)
                for category, terms in self._blacklist.items():
                    if not isinstance(terms, list):
                        continue
                    for term in terms:
                        t = str(term).strip()
                        if not t:
                            continue
                        proc.add_keyword(t, t.lower())
                        self._category_by_term[t.lower()] = str(category)
                self._processor = proc
            else:
                self._processor = None
                for category, terms in self._blacklist.items():
                    if not isinstance(terms, list):
                        continue
                    for term in terms:
                        t = str(term).strip().lower()
                        if t:
                            self._category_by_term[t] = str(category)
            self._loaded_at = now
            self._last_summary = self._build_summary(validation=validation)
            logger.info(
                "SECURITY_POLICIES_LOADED %s",
                json.dumps(self._last_summary, ensure_ascii=True),
            )

    def _maybe_reload(self) -> None:
        self.reload(force=False)

    def dlp_policies(self) -> dict[str, Any]:
        self._maybe_reload()
        with self._lock:
            return dict(self._dlp_policies)

    def security_policy(self) -> dict[str, Any]:
        self._maybe_reload()
        with self._lock:
            return dict(self._security_policy)

    def summary(self) -> dict[str, Any]:
        self._maybe_reload()
        with self._lock:
            return dict(self._last_summary)

    def _build_summary(self, *, validation: dict[str, Any]) -> dict[str, Any]:
        blacklist_categories = sum(
            1
            for _cat, terms in self._blacklist.items()
            if isinstance(terms, list) and terms
        )
        allowlist_categories = sum(
            1
            for _cat, terms in self._allowlist.items()
            if isinstance(terms, list) and terms
        )
        moderation_cfg = self._dlp_policies.get("moderation")
        moderation_rule_count = 0
        if isinstance(moderation_cfg, dict):
            for value in moderation_cfg.values():
                if isinstance(value, list):
                    moderation_rule_count += len(value)
                elif value is not None:
                    moderation_rule_count += 1

        topic_cfg = self._dlp_policies.get("topics")
        topic_policy_count = len(topic_cfg) if isinstance(topic_cfg, dict) else 0
        dlp_blacklist_cfg = self._dlp_policies.get("blacklist_categories")
        dlp_category_count = len(dlp_blacklist_cfg) if isinstance(dlp_blacklist_cfg, dict) else 0

        return {
            "config_dir": str(self.config_dir),
            "blacklist_categories": blacklist_categories,
            "blacklist_terms": len(self._category_by_term),
            "allowlist_categories": allowlist_categories,
            "allowlist_terms": sum(
                len(terms)
                for terms in self._allowlist.values()
                if isinstance(terms, list)
            ),
            "dlp_blacklist_categories": dlp_category_count,
            "dlp_topic_policies": topic_policy_count,
            "moderation_rules": moderation_rule_count,
            "document_classification_patterns": len(
                (self._security_policy.get("document_classification") or {}).get("metadata_patterns") or {}
            )
            if isinstance(self._security_policy.get("document_classification"), dict)
            else 0,
            "security_score_weights": len(
                (self._security_policy.get("security_scores") or {}).get("ingress_weights") or {}
            )
            if isinstance(self._security_policy.get("security_scores"), dict)
            else 0,
            "missing_files": validation.get("missing_files") or [],
            "strict_mode": bool(validation.get("strict_mode")),
            "files": validation.get("files") or {},
        }

    def _allowlist_phrases_in_text(self, text: str) -> list[str]:
        lower = text.lower()
        present: list[str] = []
        for terms in self._allowlist.values():
            if not isinstance(terms, list):
                continue
            for term in terms:
                phrase = str(term).strip().lower()
                if phrase and phrase in lower:
                    present.append(phrase)
        return present

    def _is_allowlisted_hit(self, matched: str, allowlist_present: list[str]) -> bool:
        needle = matched.lower()
        for phrase in allowlist_present:
            # Only override when the allowlist phrase itself appears in the document
            # and the blacklist hit is contained in that phrase.
            if needle in phrase:
                return True
        return False

    def scan_text(self, text: str) -> dict[str, Any]:
        """Return blacklist hits that are not allowlisted."""
        self._maybe_reload()
        if not text:
            return {"matches": [], "blocked": False, "action": "allow", "categories": [], "allowlist_hits": []}

        allowlist_present = self._allowlist_phrases_in_text(text)
        hits: list[dict[str, Any]] = []
        with self._lock:
            if self._processor is not None:
                found = self._processor.extract_keywords(text)
                for term in found:
                    key = str(term).lower()
                    if self._is_allowlisted_hit(key, allowlist_present):
                        continue
                    category = self._category_by_term.get(key, "blacklist")
                    hits.append(
                        {
                            "category": category,
                            "type": f"blacklist:{category}",
                            "severity": self._severity_for(category),
                            "matched_value": term,
                            "masked_value": "***",
                            "location": "Text Block",
                        }
                    )
            else:
                lower = text.lower()
                for term, category in self._category_by_term.items():
                    if self._is_allowlisted_hit(term, allowlist_present):
                        continue
                    if re.search(rf"\b{re.escape(term)}\b", lower):
                        hits.append(
                            {
                                "category": category,
                                "type": f"blacklist:{category}",
                                "severity": self._severity_for(category),
                                "matched_value": term,
                                "masked_value": "***",
                                "location": "Text Block",
                            }
                        )

        action = "allow"
        categories = sorted({h["category"] for h in hits})
        policy_cats = (self._dlp_policies.get("blacklist_categories") or {}) if isinstance(self._dlp_policies, dict) else {}
        for cat in categories:
            cfg = policy_cats.get(cat) if isinstance(policy_cats, dict) else None
            cat_action = (cfg or {}).get("action", "block") if isinstance(cfg, dict) else "block"
            if cat_action == "block":
                action = "block"
                break
            if cat_action == "human_review" and action != "block":
                action = "human_review"
            elif cat_action == "warning" and action == "allow":
                action = "warning"

        logger.info(
            "SECURITY_BLACKLIST_EVAL action=%s matches=%d categories=%s allowlist_hits=%d",
            action,
            len(hits),
            categories,
            len(allowlist_present),
        )
        return {
            "matches": hits,
            "blocked": action == "block",
            "action": action,
            "categories": categories,
            "allowlist_hits": allowlist_present,
        }

    def _severity_for(self, category: str) -> str:
        policy_cats = (self._dlp_policies.get("blacklist_categories") or {}) if isinstance(self._dlp_policies, dict) else {}
        cfg = policy_cats.get(category) if isinstance(policy_cats, dict) else None
        if isinstance(cfg, dict) and cfg.get("severity"):
            return str(cfg["severity"])
        return "high"


_engine: KeywordPolicyEngine | None = None
_engine_lock = threading.Lock()


def get_keyword_policy_engine() -> KeywordPolicyEngine:
    global _engine
    with _engine_lock:
        desired = _default_config_dir()
        if _engine is None or _engine.config_dir != desired:
            _engine = KeywordPolicyEngine(desired)
        return _engine


def scan_keyword_policies(text: str) -> dict[str, Any]:
    return get_keyword_policy_engine().scan_text(text)


def load_dlp_policies() -> dict[str, Any]:
    return get_keyword_policy_engine().dlp_policies()


def validate_policy_configuration(*, force_reload: bool = True) -> dict[str, Any]:
    """Validate mounted policy files and return a structured summary."""
    engine = get_keyword_policy_engine()
    if force_reload:
        engine.reload(force=True)
    return engine.summary()


def is_auto_block_enabled() -> bool:
    """Whether DLP may hard-block without a reviewer dropdown.

    Driven by configs/security/dlp_policies.yaml ``enforcement.auto_block_enabled``
    (default false). Override with env ``DLP_AUTO_BLOCK_ENABLED``.
    """
    env = (os.getenv("DLP_AUTO_BLOCK_ENABLED") or "").strip().lower()
    if env:
        return env in {"1", "true", "yes", "on"}
    policies = load_dlp_policies()
    enf = policies.get("enforcement") if isinstance(policies.get("enforcement"), dict) else {}
    return bool(enf.get("auto_block_enabled", False))


def is_reopen_on_reviewer_decide_enabled() -> bool:
    """Whether apply-decision may reopen a terminal queue row before recording a new choice."""
    env = (os.getenv("DLP_REOPEN_ON_REVIEWER_DECIDE") or "").strip().lower()
    if env:
        return env in {"1", "true", "yes", "on"}
    policies = load_dlp_policies()
    enf = policies.get("enforcement") if isinstance(policies.get("enforcement"), dict) else {}
    return bool(enf.get("reopen_on_reviewer_decide", True))


def is_auto_allow_only_low() -> bool:
    """When true, only LOW + DLP ALLOW may skip human review on upload."""
    env = (os.getenv("DLP_AUTO_ALLOW_ONLY_LOW") or "").strip().lower()
    if env:
        return env in {"1", "true", "yes", "on"}
    policies = load_dlp_policies()
    enf = policies.get("enforcement") if isinstance(policies.get("enforcement"), dict) else {}
    return bool(enf.get("auto_allow_only_low", True))


_security_policy_cache: dict[str, Any] | None = None
_security_policy_loaded_at: float = 0.0
_security_policy_lock = threading.RLock()
_SECURITY_POLICY_RELOAD_SECONDS = 30.0


def _security_policy_path() -> Path:
    return _default_config_dir() / "security_policy.yaml"


def load_security_policy(*, force: bool = False) -> dict[str, Any]:
    """Load configs/security/security_policy.yaml (cached, hot-reloadable)."""
    global _security_policy_cache, _security_policy_loaded_at
    with _security_policy_lock:
        now = time.time()
        if (
            not force
            and _security_policy_cache is not None
            and (now - _security_policy_loaded_at) < _SECURITY_POLICY_RELOAD_SECONDS
        ):
            return dict(_security_policy_cache)
        # Reuse strict validation so missing policy files fail loudly.
        _validate_policy_files(_default_config_dir())
        _security_policy_cache = _load_yaml(_security_policy_path())
        _security_policy_loaded_at = now
        return dict(_security_policy_cache)


def _prompt_guard_policy() -> dict[str, Any]:
    policy = load_security_policy()
    section = policy.get("prompt_guard")
    return section if isinstance(section, dict) else {}


def prompt_guard_fail_open() -> bool:
    """When false (default), model guard unavailability escalates to human_review."""
    env = (os.getenv("PROMPT_GUARD_FAIL_OPEN") or "").strip().lower()
    if env:
        return env in {"1", "true", "yes", "on"}
    return bool(_prompt_guard_policy().get("fail_open", False))


def prompt_guard_unavailable_action() -> str:
    env = (os.getenv("PROMPT_GUARD_UNAVAILABLE_ACTION") or "").strip().lower()
    if env:
        return env
    action = str(_prompt_guard_policy().get("unavailable_action") or "human_review").strip().lower()
    return action or "human_review"


def load_pdf_preprocessor_config() -> dict[str, Any]:
    policy = load_security_policy()
    section = policy.get("pdf_preprocessor")
    if isinstance(section, dict):
        return dict(section)
    return {}


def load_document_classification_config() -> dict[str, Any]:
    policy = load_security_policy()
    section = policy.get("document_classification")
    if isinstance(section, dict):
        return dict(section)
    return {
        "confidence_threshold": 0.70,
        "weights": {"metadata": 0.4, "structure": 0.4, "ml": 0.2},
        "metadata_patterns": {},
        "structure_patterns": {},
    }


def load_security_score_weights() -> dict[str, float]:
    policy = load_security_policy()
    section = policy.get("security_scores")
    if not isinstance(section, dict):
        return {"dlp": 0.30, "similarity": 0.25, "prompt_guard": 0.20, "moderation": 0.15, "keyword": 0.10}
    ingress = section.get("ingress_weights")
    if isinstance(ingress, dict):
        return {str(k): float(v) for k, v in ingress.items()}
    return {"dlp": 0.30, "similarity": 0.25, "prompt_guard": 0.20, "moderation": 0.15, "keyword": 0.10}
