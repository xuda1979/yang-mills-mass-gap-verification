import json


def test_rp_evidence_is_pass_with_proof_artifact():
    # With proper proof artifact bound (real SHA-256), RP evidence should PASS.
    try:
        from rp_evidence import audit_rp_evidence, default_rp_evidence_path
    except Exception:
        from verification.rp_evidence import audit_rp_evidence, default_rp_evidence_path

    res = audit_rp_evidence(default_rp_evidence_path())
    assert res["status"] in {"PASS", "FAIL", "CONDITIONAL"}
    assert res["status"] == "PASS"


def test_os_obligations_rp_item_exists():
    try:
        from os_obligations import os_obligations
    except Exception:
        from verification.os_obligations import os_obligations

    obs = os_obligations()
    rp_item = next(o for o in obs if o.get("key") == "os_rp_lattice_proved")
    assert rp_item.get("status") in {"PASS", "CONDITIONAL", "FAIL"}
    # With valid rp evidence, this should now be PASS.
    assert rp_item.get("status") == "PASS"
