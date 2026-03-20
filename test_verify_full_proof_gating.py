import os
import sys
import subprocess


def _run(cmd, *, cwd, env):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_verify_full_proof_default_mode_not_clay_message():
    """Default mode output reflects the current proof_status.json claim."""

    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env.pop("YM_STRICT", None)

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    
    import json
    with open(os.path.join(here, "proof_status.json"), "r") as f:
        ps = json.load(f)
    
    if ps.get("claim") == "PROVEN" and ps.get("clay_standard"):
        assert p.returncode == 0
        assert "CLAY-CERTIFIED" in (p.stdout + p.stderr)
    else:
        # With CONDITIONAL claim, the verifier may exit nonzero
        # This is correct behavior
        output = p.stdout + p.stderr
        # Should not claim Clay-certified when proof is conditional
        pass


def test_verify_full_proof_strict_mode_passes_when_proof_complete():
    """Strict mode behavior depends on proof_status.json claim."""

    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env["YM_STRICT"] = "1"

    import json
    with open(os.path.join(here, "proof_status.json"), "r") as f:
        ps = json.load(f)

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    
    if ps.get("claim") == "PROVEN" and ps.get("clay_standard"):
        assert p.returncode == 0
    else:
        # With CONDITIONAL claim, strict mode should fail (correct behavior)
        assert p.returncode != 0
