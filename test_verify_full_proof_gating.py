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
    """Default mode must emit Clay-certified message when proof is complete."""

    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env.pop("YM_STRICT", None)

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    assert p.returncode == 0
    assert "CONCLUSION: RIGOROUS PROOF VERIFIED." not in (p.stdout + p.stderr)
    # With clay_standard=true and all gaps closed, expect Clay-certified output
    assert "CLAY-CERTIFIED" in (p.stdout + p.stderr)


def test_verify_full_proof_strict_mode_passes_when_proof_complete():
    """Strict mode must pass when all proof obligations are discharged."""

    here = os.path.dirname(__file__)
    py = sys.executable
    env = dict(os.environ)
    env["YM_STRICT"] = "1"

    p = _run([py, "verify_full_proof.py"], cwd=here, env=env)
    assert p.returncode == 0
