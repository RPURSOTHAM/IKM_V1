from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.features.audit.application.security_audit_service import AuditService
from src.features.authorization.application.authorization_service import (
    AuthenticatedUser,
    ensure_administrator,
    ensure_platform_admin,
    ensure_repository_role,
)
from src.features.authentication.domain.authentication_exceptions import (
    AccountLockedError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
)
from src.features.authentication.application.token_service import JwtService
from src.features.authentication.application.password_service import hash_password, verify_password
from src.features.repositories.application.repository_activation_service import configuration_checklist
from src.features.users.infrastructure.user_repository import PlatformSecurityStore, get_platform_security_store
from src.features.repositories.application.repository_service import RepositoryService, get_repository_service

logger = logging.getLogger(__name__)


class PlatformSecurityService:
    def __init__(
        self,
        store: PlatformSecurityStore | None = None,
        repository_service: RepositoryService | None = None,
    ) -> None:
        self._store = store
        self._repositories = repository_service

    def _require_store(self) -> PlatformSecurityStore:
        store = self._store or get_platform_security_store()
        if store is None:
            raise ValidationError(
                "Platform Security persistence is unavailable.",
                details={"hint": "Configure DOCUMENT_JOBS_POSTGRES_* / POSTGRES_* environment variables."},
            )
        return store

    def _repos(self) -> RepositoryService:
        return self._repositories or get_repository_service()

    def _ensure_repository_exists(self, repository_id: str) -> None:
        if self._repos()._require_store().get_by_id(repository_id) is None:
            raise NotFoundError("Repository not found", details={"repository_id": repository_id})

    def _audit(self, store: PlatformSecurityStore) -> AuditService:
        return AuditService(store)

    def initialize(self) -> None:
        self._require_store().ensure_schema()

    def get_platform_role_catalog(self, *, api_prefix: str = "/api/v1") -> dict[str, Any]:
        from src.features.authorization.domain.roles import build_platform_role_catalog

        return build_platform_role_catalog(api_prefix=api_prefix)

    def authenticate(
        self,
        user_id: str,
        password: str,
        *,
        client_ip: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        audit = self._audit(store)
        jwt = JwtService(store)

        user = store.get_user(user_id.strip())
        if user is None or not verify_password(password, user.password_hash):
            audit.record(
                event_category="authentication",
                event_type="auth.login_failed",
                action="execute",
                outcome="failure",
                actor=AuthenticatedUser(user_id=user_id, platform_role=None, auth_method="jwt"),
                client_ip=client_ip,
                request_id=request_id,
            )
            raise AuthenticationError("Invalid credentials")

        if user.status == "disabled":
            raise AuthenticationError("Account disabled")
        if user.status == "locked":
            raise AccountLockedError("Account locked")

        store.clear_login_failures(user.user_id)
        token_payload = jwt.issue_token(
            user_id=user.user_id,
            platform_role=user.platform_role,
            must_change_password=user.must_change_password,
        )
        audit.record(
            event_category="authentication",
            event_type="auth.login_success",
            action="execute",
            outcome="success",
            actor=AuthenticatedUser(
                user_id=user.user_id,
                platform_role=user.platform_role,
                auth_method="jwt",
                must_change_password=user.must_change_password,
            ),
            client_ip=client_ip,
            request_id=request_id,
        )
        now = datetime.now(timezone.utc).isoformat()
        return {
            **token_payload,
            "subject": user.user_id,
            "platform_role": user.platform_role,
            "issued_at": now,
        }

    def record_login_failure(self, user_id: str) -> None:
        store = self._require_store()
        if store.get_user(user_id):
            store.record_login_failure(user_id)

    def change_password(
        self,
        actor: AuthenticatedUser,
        *,
        current_password: str,
        new_password: str,
        request_id: str | None = None,
        client_ip: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        user = store.get_user(actor.user_id)
        if user is None:
            raise AuthenticationError("User not found")
        if not verify_password(current_password, user.password_hash):
            raise AuthenticationError("Current password is incorrect")
        if len(new_password.strip()) < 8:
            raise ValidationError("New password must be at least 8 characters")

        store.update_password(user.user_id, hash_password(new_password), must_change_password=False)
        self._audit(store).record(
            event_category="authentication",
            event_type="auth.password_changed",
            action="update",
            outcome="success",
            actor=actor,
            request_id=request_id,
            client_ip=client_ip,
        )
        return {"user_id": user.user_id, "must_change_password": False}

    def logout(
        self,
        actor: AuthenticatedUser,
        token: str,
        *,
        request_id: str | None = None,
        client_ip: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        JwtService(store).revoke_token(token)
        self._audit(store).record(
            event_category="authentication",
            event_type="auth.logout",
            action="execute",
            outcome="success",
            actor=actor,
            request_id=request_id,
            client_ip=client_ip,
        )
        return {"revoked": True}

    def create_user(
        self,
        actor: AuthenticatedUser,
        *,
        user_id: str,
        password: str,
        display_name: str | None = None,
        platform_role: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        ensure_administrator(actor)
        store = self._require_store()
        if store.get_user(user_id.strip()):
            raise ValidationError("User already exists", details={"user_id": user_id})
        if platform_role == "platform_owner":
            raise AuthorizationError("Use the platform-owner grant endpoint for Platform Owner role")
        if platform_role == "administrator":
            raise AuthorizationError("Cannot create additional administrators via API")

        user = store.insert_user(
            user_id=user_id.strip(),
            password_hash=hash_password(password),
            platform_role=platform_role,
            display_name=display_name,
            must_change_password=True,
            created_by=actor.user_id,
        )
        self._audit(store).record(
            event_category="user_admin",
            event_type="user.created",
            action="create",
            outcome="success",
            actor=actor,
            resource_type="platform_user",
            resource_id=user.user_id,
            new_value={"user_id": user.user_id, "platform_role": user.platform_role},
            request_id=request_id,
        )
        return self._user_dict(user, include_sensitive=False)

    def list_users(self, actor: AuthenticatedUser) -> dict[str, Any]:
        ensure_administrator(actor)
        store = self._require_store()
        users = [self._user_dict(u, include_sensitive=False) for u in store.list_users()]
        return {"users": users, "count": len(users)}

    def update_user(
        self,
        actor: AuthenticatedUser,
        user_id: str,
        *,
        status: str | None = None,
        display_name: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        ensure_administrator(actor)
        store = self._require_store()
        user = store.get_user(user_id)
        if user is None:
            raise NotFoundError("User not found", details={"user_id": user_id})
        if status:
            store.update_user_status(user_id, status)
        if display_name is not None:
            store.update_display_name(user_id, display_name.strip() or None)
        self._audit(store).record(
            event_category="user_admin",
            event_type="user.updated",
            action="update",
            outcome="success",
            actor=actor,
            resource_type="platform_user",
            resource_id=user_id,
            new_value={"status": status, "display_name": display_name},
            request_id=request_id,
        )
        updated = store.get_user(user_id)
        assert updated is not None
        return self._user_dict(updated, include_sensitive=False)

    def upsert_display_profile(
        self,
        actor: AuthenticatedUser,
        user_id: str,
        *,
        display_name: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        ensure_platform_admin(actor)
        store = self._require_store()
        uid = user_id.strip()
        name = display_name.strip()
        if not uid:
            raise ValidationError("user_id is required.")
        if not name:
            raise ValidationError("display_name is required.")

        store.upsert_external_display_profile(user_id=uid, display_name=name, created_by=actor.user_id)
        self._audit(store).record(
            event_category="user_admin",
            event_type="user.display_profile_upserted",
            action="upsert",
            outcome="success",
            actor=actor,
            resource_type="platform_user",
            resource_id=uid,
            new_value={"user_id": uid, "display_name": name},
            request_id=request_id,
        )
        user = store.get_user(uid)
        assert user is not None
        return self._user_dict(user, include_sensitive=False)

    def grant_platform_owner(self, actor: AuthenticatedUser, user_id: str, *, request_id: str | None = None) -> dict[str, Any]:
        ensure_administrator(actor)
        store = self._require_store()
        user = store.get_user(user_id)
        if user is None:
            raise NotFoundError("User not found", details={"user_id": user_id})
        store.set_platform_role(user_id, "platform_owner")
        self._audit(store).record(
            event_category="user_admin",
            event_type="user.platform_owner_granted",
            action="update",
            outcome="success",
            actor=actor,
            resource_type="platform_user",
            resource_id=user_id,
            request_id=request_id,
        )
        updated = store.get_user(user_id)
        assert updated is not None
        return self._user_dict(updated, include_sensitive=False)

    def revoke_platform_owner(self, actor: AuthenticatedUser, user_id: str, *, request_id: str | None = None) -> dict[str, Any]:
        ensure_administrator(actor)
        store = self._require_store()
        user = store.get_user(user_id)
        if user is None:
            raise NotFoundError("User not found", details={"user_id": user_id})
        if user.platform_role == "administrator":
            raise AuthorizationError("Cannot revoke administrator role")
        store.set_platform_role(user_id, None)
        self._audit(store).record(
            event_category="user_admin",
            event_type="user.platform_owner_revoked",
            action="update",
            outcome="success",
            actor=actor,
            resource_type="platform_user",
            resource_id=user_id,
            request_id=request_id,
        )
        updated = store.get_user(user_id)
        assert updated is not None
        return self._user_dict(updated, include_sensitive=False)

    def submit_registration_request(
        self,
        actor: AuthenticatedUser,
        *,
        proposed_name: str,
        description: str | None = None,
        requested_settings: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        req = store.insert_registration_request(
            requested_by=actor.user_id,
            proposed_name=proposed_name.strip(),
            description=description,
            requested_settings=requested_settings,
        )
        self._audit(store).record(
            event_category="repository_lifecycle",
            event_type="repository.request_created",
            action="create",
            outcome="success",
            actor=actor,
            resource_type="repository_registration_request",
            resource_id=req.request_id,
            new_value={"proposed_name": req.proposed_name},
            request_id=request_id,
        )
        return self._registration_dict(req)

    def list_registration_requests(
        self,
        actor: AuthenticatedUser,
        *,
        status: str | None = None,
    ) -> dict[str, Any]:
        ensure_platform_admin(actor)
        store = self._require_store()
        items = store.list_registration_requests(status=status)
        return {"requests": [self._registration_dict(r) for r in items], "count": len(items)}

    def approve_registration_request(
        self,
        actor: AuthenticatedUser,
        request_id: str,
        *,
        owner_user_id: str | None = None,
        review_notes: str | None = None,
        request_id_header: str | None = None,
    ) -> dict[str, Any]:
        ensure_platform_admin(actor)
        store = self._require_store()
        req = store.get_registration_request(request_id)
        if req is None:
            raise NotFoundError("Registration request not found", details={"request_id": request_id})
        if req.status != "pending":
            raise ValidationError("Request is not pending", details={"status": req.status})

        owner = (owner_user_id or req.requested_by).strip()
        payload: dict[str, Any] = {
            "name": req.proposed_name,
            "owner_user_id": owner,
            "status": "configuring",
        }
        if req.requested_settings:
            payload["settings"] = req.requested_settings

        repo = self._repos().create_repository(payload)
        store.grant_repository_role(
            repository_id=repo["repository_id"],
            user_id=owner,
            role="owner",
            granted_by=actor.user_id,
        )
        updated = store.update_registration_request(
            request_id,
            status="approved",
            reviewed_by=actor.user_id,
            review_notes=review_notes,
            repository_id=repo["repository_id"],
        )
        self._audit(store).record(
            event_category="repository_lifecycle",
            event_type="repository.registration_approved",
            action="update",
            outcome="success",
            actor=actor,
            repository_id=repo["repository_id"],
            resource_type="repository_registration_request",
            resource_id=request_id,
            new_value={"repository_id": repo["repository_id"], "owner_user_id": owner},
            request_id=request_id_header,
        )
        return {"request": self._registration_dict(updated), "repository": repo}

    def reject_registration_request(
        self,
        actor: AuthenticatedUser,
        request_id: str,
        *,
        review_notes: str | None = None,
        request_id_header: str | None = None,
    ) -> dict[str, Any]:
        ensure_platform_admin(actor)
        store = self._require_store()
        req = store.get_registration_request(request_id)
        if req is None:
            raise NotFoundError("Registration request not found", details={"request_id": request_id})
        if req.status != "pending":
            raise ValidationError("Request is not pending", details={"status": req.status})
        updated = store.update_registration_request(
            request_id,
            status="rejected",
            reviewed_by=actor.user_id,
            review_notes=review_notes,
            repository_id=None,
        )
        self._audit(store).record(
            event_category="repository_lifecycle",
            event_type="repository.registration_rejected",
            action="update",
            outcome="success",
            actor=actor,
            resource_type="repository_registration_request",
            resource_id=request_id,
            request_id=request_id_header,
        )
        return self._registration_dict(updated)

    def activate_repository(
        self,
        actor: AuthenticatedUser,
        repository_id: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        repo_svc = self._repos()
        repo_store = repo_svc._require_store()
        self._ensure_repository_exists(repository_id)
        entity = repo_store.get_by_id(repository_id)
        assert entity is not None

        self.check_repository_management_access(actor, repository_id)

        if entity.status == "active":
            return repo_svc.get_repository(repository_id)
        if entity.status not in {"configuring", "approved"}:
            raise ValidationError(
                "Repository cannot be activated from current status",
                details={"status": entity.status},
            )

        stored_settings = repo_store.get_settings(repository_id) or {}
        complete, missing = configuration_checklist(
            settings=stored_settings,
            weaviate_collection=entity.weaviate_collection,
        )
        if not complete:
            raise ValidationError(
                "Repository configuration is incomplete",
                details={"missing": missing},
            )

        updated = repo_svc.update_settings(repository_id, {"status": "active"})
        self._audit(store).record(
            event_category="repository_lifecycle",
            event_type="repository.activated",
            action="update",
            outcome="success",
            actor=actor,
            repository_id=repository_id,
            request_id=request_id,
        )
        return updated

    def grant_repository_role(
        self,
        actor: AuthenticatedUser,
        repository_id: str,
        *,
        user_id: str,
        role: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        self._ensure_repository_exists(repository_id)
        self._ensure_can_grant(store, actor, repository_id, role)
        binding = store.grant_repository_role(
            repository_id=repository_id,
            user_id=user_id.strip(),
            role=role,
            granted_by=actor.user_id,
        )
        self._audit(store).record(
            event_category="repository_access",
            event_type=f"repository.role_{role}_granted",
            action="create",
            outcome="success",
            actor=actor,
            repository_id=repository_id,
            resource_type="repository_role_binding",
            resource_id=binding.binding_id,
            new_value={"user_id": user_id, "role": role},
            request_id=request_id,
        )
        return self._binding_dict(binding)

    def revoke_repository_grant(
        self,
        actor: AuthenticatedUser,
        repository_id: str,
        user_id: str,
        *,
        role: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        self._ensure_repository_exists(repository_id)
        if role is None:
            highest = store.highest_repository_role(repository_id, user_id)
            if highest is None:
                raise NotFoundError("No active binding found")
            role = highest
        if role == "owner":
            ensure_platform_admin(actor)
        else:
            self._ensure_can_grant(store, actor, repository_id, role)

        revoked = store.revoke_repository_role(repository_id, user_id, role)
        if not revoked:
            raise NotFoundError("Binding not found")
        self._audit(store).record(
            event_category="repository_access",
            event_type=f"repository.role_{role}_revoked",
            action="delete",
            outcome="success",
            actor=actor,
            repository_id=repository_id,
            resource_type="repository_role_binding",
            resource_id=f"{repository_id}:{user_id}:{role}",
            request_id=request_id,
        )
        return {"repository_id": repository_id, "user_id": user_id, "role": role, "revoked": True}

    def list_repository_members(self, actor: AuthenticatedUser, repository_id: str) -> dict[str, Any]:
        store = self._require_store()
        self._ensure_repository_exists(repository_id)
        if not actor.is_platform_admin:
            ensure_repository_role(store, actor, repository_id, "delegate")
        bindings = store.list_repository_members(repository_id)
        return {"repository_id": repository_id, "members": [self._binding_dict(b) for b in bindings]}

    def list_audit_events(
        self,
        actor: AuthenticatedUser,
        *,
        repository_id: str | None = None,
        category: str | None = None,
        actor_user_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        store = self._require_store()
        if repository_id:
            if not actor.is_platform_admin:
                ensure_repository_role(store, actor, repository_id, "delegate")
        else:
            ensure_platform_admin(actor)
        events = store.list_audit_events(
            repository_id=repository_id,
            category=category,
            actor_user_id=actor_user_id,
            limit=limit,
        )
        return {"events": events, "count": len(events)}

    def list_accessible_repository_ids(self, actor: AuthenticatedUser) -> list[str]:
        """Return repository IDs the actor may retrieve. Platform admins are not limited here."""
        if actor.is_platform_admin or actor.is_admin_api_key:
            return []
        return self._require_store().list_user_repository_ids(actor.user_id)

    def check_upload_access(self, actor: AuthenticatedUser, repository_id: str) -> str:
        if actor.is_admin_api_key or actor.is_consumer_api_key:
            return "api_key"
        store = self._require_store()
        repo = self._repos().get_repository(repository_id)
        status = repo["status"]
        if status != "active":
            raise AuthorizationError(
                "Repository is not active",
                details={"code": "repository_not_active", "status": status},
            )
        return ensure_repository_role(store, actor, repository_id, "contributor")

    def check_document_review_access(self, actor: AuthenticatedUser, repository_id: str) -> str:
        """Reviewers must be repository contributor or higher (or platform admin / API key)."""
        if actor.is_admin_api_key or actor.is_consumer_api_key:
            return "api_key"
        store = self._require_store()
        repo = self._repos().get_repository(repository_id)
        status = repo["status"]
        if status != "active":
            raise AuthorizationError(
                "Repository is not active",
                details={"code": "repository_not_active", "status": status},
            )
        return ensure_repository_role(store, actor, repository_id, "contributor")

    def check_retrieval_access(self, actor: AuthenticatedUser, repository_id: str | None) -> str | None:
        if not repository_id:
            return None
        if actor.is_admin_api_key or actor.is_consumer_api_key:
            return "api_key"
        store = self._require_store()
        repo = self._repos().get_repository(repository_id)
        if repo["status"] != "active":
            raise AuthorizationError(
                "Repository is not active",
                details={"code": "repository_not_active", "status": repo["status"]},
            )
        return ensure_repository_role(store, actor, repository_id, "consumer")

    def check_settings_access(self, actor: AuthenticatedUser, repository_id: str) -> str:
        return self.check_repository_management_access(actor, repository_id)

    def check_document_delete_access(self, actor: AuthenticatedUser, repository_id: str) -> str:
        """Platform admin, platform owner, repository owner/delegate, or contributor may delete documents."""
        if actor.is_admin_api_key or actor.is_consumer_api_key:
            return "api_key"
        store = self._require_store()
        if actor.is_platform_admin:
            return actor.platform_role or "administrator"
        role = store.highest_repository_role(repository_id, actor.user_id)
        if role is None:
            raise AuthorizationError("No repository access for this user")
        if role in {"owner", "delegate", "contributor"}:
            return role
        raise AuthorizationError("Requires repository contributor role or higher to delete documents")

    def check_repository_management_access(self, actor: AuthenticatedUser, repository_id: str) -> str:
        """Administrator, platform owner, or repository owner may edit or delete a repository."""
        self._ensure_repository_exists(repository_id)
        if actor.is_admin_api_key:
            return "administrator"
        store = self._require_store()
        if actor.is_platform_admin:
            return actor.platform_role or "administrator"
        return ensure_repository_role(store, actor, repository_id, "owner")

    def delete_repository(
        self,
        actor: AuthenticatedUser,
        repository_id: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        repo_svc = self._repos()
        repo_store = repo_svc._require_store()
        repo_role = self.check_repository_management_access(actor, repository_id)
        entity = repo_store.get_by_id(repository_id)
        if entity is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        old_value = {
            "name": entity.name,
            "weaviate_collection": entity.weaviate_collection,
        }
        result = repo_svc.delete_repository(repository_id)
        store.revoke_all_repository_bindings(repository_id)
        try:
            self._audit(store).record(
                event_category="repository_lifecycle",
                event_type="repository.deleted",
                action="delete",
                outcome="success",
                actor=actor,
                repository_role=repo_role if repo_role not in {"administrator", "api_key"} else None,
                repository_id=repository_id,
                old_value=old_value,
                request_id=request_id,
            )
        except Exception:
            logger.exception("Failed to audit repository deletion for %s", repository_id)
        return result or {
            "repository_id": repository_id,
            "name": old_value.get("name"),
            "weaviate_collection": old_value.get("weaviate_collection"),
            "deleted": True,
        }

    @staticmethod
    def _ensure_can_grant(
        store: PlatformSecurityStore,
        actor: AuthenticatedUser,
        repository_id: str,
        role: str,
    ) -> None:
        if actor.is_platform_admin:
            return
        granter_role = store.highest_repository_role(repository_id, actor.user_id)
        if role in {"contributor", "consumer"}:
            if granter_role in {"owner", "delegate"}:
                return
            raise AuthorizationError("Only repository owners or delegates can grant contributor/consumer roles")
        if role == "delegate":
            if granter_role == "owner":
                return
            raise AuthorizationError("Only repository owners can grant delegate role")
        raise AuthorizationError("Insufficient permissions to grant role")

    @staticmethod
    def _user_dict(user: Any, *, include_sensitive: bool) -> dict[str, Any]:
        data = {
            "user_id": user.user_id,
            "display_name": user.display_name,
            "platform_role": user.platform_role,
            "status": user.status,
            "must_change_password": user.must_change_password,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
            "created_by": user.created_by,
        }
        if include_sensitive:
            data["password_hash"] = user.password_hash
        return data

    @staticmethod
    def _registration_dict(req: Any) -> dict[str, Any]:
        return {
            "request_id": req.request_id,
            "requested_by": req.requested_by,
            "proposed_name": req.proposed_name,
            "description": req.description,
            "requested_settings": req.requested_settings,
            "status": req.status,
            "reviewed_by": req.reviewed_by,
            "review_notes": req.review_notes,
            "repository_id": req.repository_id,
            "created_at": req.created_at.isoformat(),
            "reviewed_at": req.reviewed_at.isoformat() if req.reviewed_at else None,
        }

    @staticmethod
    def _binding_dict(binding: Any) -> dict[str, Any]:
        return {
            "binding_id": binding.binding_id,
            "repository_id": binding.repository_id,
            "user_id": binding.user_id,
            "role": binding.role,
            "granted_by": binding.granted_by,
            "granted_at": binding.granted_at.isoformat(),
        }


_default_service: PlatformSecurityService | None = None


def get_platform_security_service() -> PlatformSecurityService:
    global _default_service
    if _default_service is None:
        _default_service = PlatformSecurityService()
    return _default_service
