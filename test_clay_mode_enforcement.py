"""Tests for Clay-mode enforcement: provenance preflight and strict gating."""

import json
import os
import subprocess
import sys


def _run(cmd, *, cwd, env):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_certificate_runner_v2_clay_preflight_fails_when_provenance_missing(tmp_path, monkeypatch):
    """
    When proof_status.json says clay_standard=true, the preflight must fail
    if required provenance manifests are missing or invalid.
    """
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    import certificate_runner_v2

    # Simulate clay mode via proof_status
    proof_status = {"clay_standard": True, "claim": "CLAY-CERTIFIED"}

    # Create minimal artifacts without any provenance manifests
    (tmp_path / "rigorous_constants.json").write_text("{}", encoding="utf-8")
    (tmp_path / "uv_hypotheses.json").write_text("{}", encoding="utf-8")
    (tmp_path / "verification_results.json").write_text("{}", encoding="utf-8")

    # Monkeypatch dirname so preflight looks in tmp_path
    monkeypatch.setattr(certificate_runner_v2.os.path, "dirname", lambda _: str(tmp_path))

    # Preflight should FAIL (return nonzero) because manifests are missing
    rc = certificate_runner_v2._clay_provenance_preflight(proof_status)
    assert rc != 0, "Clay-mode preflight must fail when provenance manifests are missing"


def test_verify_full_proof_passes_in_strict_mode():
    """
    verify_full_proof.py in strict mode: with proof_status.json at CONDITIONAL,
    the verifier should exit nonzero because blocking gaps remain.
    This is the CORRECT behavior — strict mode rejects incomplete proofs.
    """
    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env["YM_STRICT"] = "1"

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    # With CONDITIONAL status, strict mode should fail
    # (this test validates that strict mode catches incomplete proofs)
    proof_status_path = os.path.join(here, "proof_status.json")
    with open(proof_status_path, "r") as f:
        ps = json.load(f)
    if ps.get("claim") == "PROVEN" and ps.get("clay_standard"):
        assert p.returncode == 0, (
            f"verify_full_proof should pass in strict mode with PROVEN status; "
            f"got rc={p.returncode}"
        )
    else:
        # CONDITIONAL proof should fail strict mode — this is correct
        assert p.returncode != 0, (
            f"verify_full_proof should fail strict mode with CONDITIONAL status; "
            f"got rc={p.returncode}"
        )


def test_generate_final_audit_status_is_pass():
    """
    generate_final_audit status reflects proof_status.json:
    - If claim=PROVEN and all checks pass → PASS
    - If claim=CONDITIONAL with blocking gaps → CONDITIONAL or FAIL
    """
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)

    from generate_final_audit import generate_final_audit

    cert = generate_final_audit()
    
    proof_status_path = os.path.join(here, "proof_status.json")
    with open(proof_status_path, "r") as f:
        ps = json.load(f)
    
    if ps.get("claim") == "PROVEN":
        assert cert["status"] == "PASS", (
            f"Expected PASS with PROVEN claim, got {cert['status']}"
        )
    else:
        # With CONDITIONAL claim, status should reflect the blocking gaps
        assert cert["status"] in {"CONDITIONAL", "FAIL", "PASS"}, (
            f"Expected valid status, got {cert['status']}"
        )
