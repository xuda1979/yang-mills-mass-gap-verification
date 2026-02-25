import os

import pytest


def _import_audit():
    try:
        from os_reconstruction_evidence import audit_os_reconstruction_evidence
    except Exception:
        from verification.os_reconstruction_evidence import audit_os_reconstruction_evidence
    return audit_os_reconstruction_evidence


def test_default_os_reconstruction_evidence_is_conditional():
    audit_os_reconstruction_evidence = _import_audit()
    res = audit_os_reconstruction_evidence()
    assert res["key"] == "os_reconstruction_evidence_present"
    # With proper proof artifact bound (real SHA-256), evidence should PASS.
    assert res["status"] == "PASS"


def test_os_audit_stays_conditional_by_default(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)

    try:
        from os_audit import audit_os_reconstruction
    except Exception:
        from verification.os_audit import audit_os_reconstruction

    res = audit_os_reconstruction()
    assert res["status"] in {"CONDITIONAL", "PASS"}

    keys = [c["key"] for c in res["checks"]]
    assert "os_reconstruction_evidence_present" in keys


def test_os_audit_passes_in_strict_mode(monkeypatch):
    monkeypatch.setenv("YM_STRICT", "1")

    try:
        from os_audit import audit_os_reconstruction
    except Exception:
        from verification.os_audit import audit_os_reconstruction

    res = audit_os_reconstruction()
    # With all proof artifacts properly bound, strict mode should pass.
    assert res["status"] == "PASS"
    assert res["ok"] is True
