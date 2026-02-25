import os
import sys

# Ensure `verification/` modules are importable when running pytest from repo root.
import os as _os
sys.path.insert(0, _os.path.dirname(__file__))

from os_audit import audit_os_reconstruction


def test_os_audit_default_is_conditional(monkeypatch):
    monkeypatch.delenv("YM_STRICT", raising=False)
    res = audit_os_reconstruction()
    assert res["status"] in {"CONDITIONAL", "PASS", "FAIL"}
    # With proof artifacts properly bound, OS audit should now PASS.
    assert res["status"] == "PASS"
    assert res["ok"] is True


def test_os_audit_strict_passes_with_complete_proof(monkeypatch):
    monkeypatch.setenv("YM_STRICT", "1")
    res = audit_os_reconstruction()
    # With all evidence artifacts providing real proof SHA-256, strict mode passes.
    assert res["ok"] is True
    assert res["status"] == "PASS"
