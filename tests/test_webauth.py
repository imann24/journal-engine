"""Tests for the persistent-login token (cookie auth)."""

from __future__ import annotations

from journal import webauth


def test_token_is_deterministic_for_same_password(monkeypatch):
    monkeypatch.delenv("JOURNAL_AUTH_SECRET", raising=False)
    assert webauth.auth_token("hunter2") == webauth.auth_token("hunter2")


def test_token_changes_with_password(monkeypatch):
    monkeypatch.delenv("JOURNAL_AUTH_SECRET", raising=False)
    assert webauth.auth_token("hunter2") != webauth.auth_token("other-pass")


def test_verify_token_roundtrip(monkeypatch):
    monkeypatch.delenv("JOURNAL_AUTH_SECRET", raising=False)
    tok = webauth.auth_token("hunter2")
    assert webauth.verify_token(tok, "hunter2") is True
    assert webauth.verify_token(tok, "wrong") is False
    assert webauth.verify_token(None, "hunter2") is False
    assert webauth.verify_token("", "hunter2") is False


def test_explicit_secret_decouples_from_password(monkeypatch):
    monkeypatch.setenv("JOURNAL_AUTH_SECRET", "stable-secret")
    a = webauth.auth_token("pass-one")
    b = webauth.auth_token("pass-two")
    # With a fixed secret, the token no longer depends on the password.
    assert a == b


def test_verify_password_constant_time_semantics():
    assert webauth.verify_password("abc", "abc") is True
    assert webauth.verify_password("abc", "abd") is False
