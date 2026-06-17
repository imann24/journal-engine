"""Web auth token helpers — pure and testable (no Streamlit import).

The persistent "remember this browser" cookie stores an HMAC token, never the
password. The signing key defaults to the password itself, so changing the
password (or setting JOURNAL_AUTH_SECRET) invalidates every existing cookie.
"""

from __future__ import annotations

import hashlib
import hmac
import os

COOKIE_NAME = "journal_auth"
_MESSAGE = b"journal-auth-v1"


def auth_token(password: str) -> str:
    secret = os.environ.get("JOURNAL_AUTH_SECRET", password)
    return hmac.new(secret.encode(), _MESSAGE, hashlib.sha256).hexdigest()


def verify_token(token: str | None, password: str) -> bool:
    """Constant-time check that a cookie token matches the current password."""
    if not token:
        return False
    return hmac.compare_digest(token, auth_token(password))


def verify_password(attempt: str, password: str) -> bool:
    return hmac.compare_digest(attempt, password)
