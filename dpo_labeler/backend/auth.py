from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import cookies
from typing import Any, Mapping

from .common import APP_VERSION, AuthenticationError, normalize_text

_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
_INSTANCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


@dataclass(frozen=True)
class ReviewerSession:
    reviewer_username: str
    client_instance_id: str
    issued_at: str
    expires_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "reviewer_username": self.reviewer_username,
            "client_instance_id": self.client_instance_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


class AuthService:
    COOKIE_NAME = "dpo_labeler_session"

    def __init__(
        self,
        invite_token: str,
        session_secret: str | None = None,
        session_max_age_seconds: int = 60 * 60 * 24 * 7,
        cookie_secure: bool = False,
    ) -> None:
        token = normalize_text(invite_token)
        if not token:
            raise AuthenticationError("invite_token must be configured")
        self.invite_token = token
        self.session_secret = normalize_text(session_secret) or base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
        self.session_max_age_seconds = max(int(session_max_age_seconds), 300)
        self.cookie_secure = bool(cookie_secure)

    def start_session(self, invite_token: str, reviewer_username: str, client_instance_id: str) -> tuple[ReviewerSession, str]:
        if normalize_text(invite_token) != self.invite_token:
            raise AuthenticationError("Invalid invite token")
        username = self._validate_username(reviewer_username)
        instance_id = self._validate_instance_id(client_instance_id)
        issued_at_dt = datetime.now(timezone.utc).replace(microsecond=0)
        expires_at_dt = issued_at_dt + timedelta(seconds=self.session_max_age_seconds)
        session = ReviewerSession(
            reviewer_username=username,
            client_instance_id=instance_id,
            issued_at=issued_at_dt.isoformat(),
            expires_at=expires_at_dt.isoformat(),
        )
        return session, self._session_cookie_header(session)

    def end_session_header(self) -> str:
        morsel = cookies.SimpleCookie()
        morsel[self.COOKIE_NAME] = ""
        morsel[self.COOKIE_NAME]["path"] = "/"
        morsel[self.COOKIE_NAME]["httponly"] = True
        morsel[self.COOKIE_NAME]["max-age"] = 0
        morsel[self.COOKIE_NAME]["samesite"] = "Strict"
        if self.cookie_secure:
            morsel[self.COOKIE_NAME]["secure"] = True
        return morsel.output(header="").strip()

    def get_session(self, cookie_header: str | None) -> ReviewerSession | None:
        if not cookie_header:
            return None
        try:
            parsed = cookies.SimpleCookie()
            parsed.load(cookie_header)
            morsel = parsed.get(self.COOKIE_NAME)
            if morsel is None:
                return None
            return self._decode_session_cookie(morsel.value)
        except Exception:
            return None

    def require_session(self, cookie_header: str | None) -> ReviewerSession:
        session = self.get_session(cookie_header)
        if session is None:
            raise AuthenticationError("Authentication required")
        return session

    def _session_cookie_header(self, session: ReviewerSession) -> str:
        payload = base64.urlsafe_b64encode(json.dumps(session.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).decode("ascii")
        signature = hmac.new(self.session_secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).hexdigest()
        token = f"{payload}.{signature}"

        morsel = cookies.SimpleCookie()
        morsel[self.COOKIE_NAME] = token
        morsel[self.COOKIE_NAME]["path"] = "/"
        morsel[self.COOKIE_NAME]["httponly"] = True
        morsel[self.COOKIE_NAME]["samesite"] = "Strict"
        morsel[self.COOKIE_NAME]["max-age"] = self.session_max_age_seconds
        if self.cookie_secure:
            morsel[self.COOKIE_NAME]["secure"] = True
        return morsel.output(header="").strip()

    def _decode_session_cookie(self, token: str) -> ReviewerSession:
        try:
            payload, signature = token.split(".", 1)
        except ValueError as exc:
            raise AuthenticationError("Invalid session token") from exc
        expected = hmac.new(self.session_secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise AuthenticationError("Invalid session signature")
        try:
            decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
            data = json.loads(decoded.decode("utf-8"))
        except Exception as exc:
            raise AuthenticationError("Invalid session payload") from exc
        if not isinstance(data, Mapping):
            raise AuthenticationError("Invalid session payload")
        session = ReviewerSession(
            reviewer_username=self._validate_username(data.get("reviewer_username")),
            client_instance_id=self._validate_instance_id(data.get("client_instance_id")),
            issued_at=normalize_text(data.get("issued_at")),
            expires_at=normalize_text(data.get("expires_at")),
        )
        if not session.issued_at or not session.expires_at:
            raise AuthenticationError("Invalid session timestamps")
        expires_at_dt = datetime.fromisoformat(session.expires_at)
        if expires_at_dt.tzinfo is None:
            expires_at_dt = expires_at_dt.replace(tzinfo=timezone.utc)
        if expires_at_dt <= datetime.now(timezone.utc):
            raise AuthenticationError("Session expired")
        return session

    @staticmethod
    def _validate_username(value: Any) -> str:
        username = normalize_text(value)
        if not _USERNAME_PATTERN.fullmatch(username):
            raise AuthenticationError(
                "reviewer_username must be 2-64 chars and use letters, digits, dot, underscore, or dash"
            )
        return username

    @staticmethod
    def _validate_instance_id(value: Any) -> str:
        instance_id = normalize_text(value)
        if not _INSTANCE_PATTERN.fullmatch(instance_id):
            raise AuthenticationError("client_instance_id is invalid")
        return instance_id
