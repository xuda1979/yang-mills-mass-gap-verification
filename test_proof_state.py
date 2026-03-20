"""test_proof_state.py

Regression tests for the proof-state snapshot generator.
Validates:
- The generate_proof_state() function returns a well-shaped dict.
- The written artifact (proof_state.json) exists and is valid JSON.
- The provenance manifest is present and references the expected sources.
- Summaries, scheduling views, and hashes are structurally sound.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def proof_state():
    """Generate a fresh proof-state snapshot in memory."""
    from generate_proof_state import generate_proof_state
    return generate_proof_state()


@pytest.fixture(scope="module")
def proof_state_artifact():
    """Load the on-disk proof_state.json if it exists."""
    path = os.path.join(os.path.dirname(__file__), "proof_state.json")
    if not os.path.isfile(path):
        pytest.skip("proof_state.json not generated yet")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. In-memory snapshot structure
# ---------------------------------------------------------------------------

def test_proof_state_top_level_keys(proof_state):
    """The snapshot must contain all required top-level keys."""
    required = {
        "schema_version",
        "generated_utc",
        "python_version",
        "contract",
        "obligation_summary",
        "identification_subchecks_summary",
        "next_actions",
        "blocked_actions",
        "evidence_hashes",
        "source_hashes",
        "full_audit",
    }
    assert required.issubset(proof_state.keys()), f"Missing keys: {required - proof_state.keys()}"


def test_proof_state_schema_version(proof_state):
    assert proof_state["schema_version"] == "1.0.0"


def test_proof_state_contract_shape(proof_state):
    contract = proof_state["contract"]
    assert isinstance(contract, dict)
    assert "statement" in contract
    assert "inputs" in contract
    assert "success_criteria" in contract
    assert "current_status" in contract
    assert contract["current_status"] in ("theorem_boundary", "discharged")
    assert len(contract["inputs"]) == 5


# ---------------------------------------------------------------------------
# 2. Obligation summary
# ---------------------------------------------------------------------------

def test_obligation_summary_shape(proof_state):
    obl = proof_state["obligation_summary"]
    assert isinstance(obl, dict)
    assert "total" in obl and "pass" in obl and "conditional" in obl
    assert "by_key" in obl
    assert isinstance(obl["by_key"], dict)
    # At least 5 obligations must be present
    assert obl["total"] >= 5


def test_obligation_summary_counts(proof_state):
    obl = proof_state["obligation_summary"]
    assert obl["pass"] + obl["conditional"] == obl["total"]


def test_discharge_obligation_status(proof_state):
    obl = proof_state["obligation_summary"]
    assert obl["by_key"].get("ym_gap_bridge_discharge") == "PASS"


# ---------------------------------------------------------------------------
# 3. Identification subchecks summary
# ---------------------------------------------------------------------------

def test_subchecks_summary_shape(proof_state):
    isc = proof_state["identification_subchecks_summary"]
    assert isinstance(isc, dict)
    assert isc["branch_count"] == 3
    assert isc["total_leaves"] == 9
    assert isc["pass_leaves"] + isc["conditional_leaves"] == isc["total_leaves"]


def test_subchecks_branches_have_keys(proof_state):
    isc = proof_state["identification_subchecks_summary"]
    for branch in isc["branches"]:
        assert "key" in branch
        assert "title" in branch
        assert "status" in branch
        assert "theorem_role" in branch
        assert "leaf_keys" in branch
        assert len(branch["leaf_keys"]) == 3


EXPECTED_BRANCH_ROLES = {
    "projector_identification",
    "algebra_representation_identification",
    "generator_domain_identification",
}


def test_subchecks_theorem_roles(proof_state):
    isc = proof_state["identification_subchecks_summary"]
    roles = {b["theorem_role"] for b in isc["branches"]}
    assert EXPECTED_BRANCH_ROLES == roles


# ---------------------------------------------------------------------------
# 4. Scheduling views
# ---------------------------------------------------------------------------

def test_next_actions_non_empty(proof_state):
    na = proof_state["next_actions"]
    assert isinstance(na, list)
    # When bridge is discharged, next_actions may be empty (all done)
    contract_status = proof_state.get("contract", {}).get("current_status", "")
    if contract_status != "discharged":
        assert len(na) >= 1


def test_next_actions_have_required_fields(proof_state):
    for action in proof_state["next_actions"]:
        assert "key" in action
        assert "title" in action
        assert "priority" in action
        assert "readiness_reason" in action


def test_blocked_actions_non_empty(proof_state):
    ba = proof_state["blocked_actions"]
    assert isinstance(ba, list)
    # When bridge is discharged, blocked_actions may be empty (all resolved)
    contract_status = proof_state.get("contract", {}).get("current_status", "")
    if contract_status != "discharged":
        assert len(ba) >= 1


def test_blocked_actions_have_required_fields(proof_state):
    for action in proof_state["blocked_actions"]:
        assert "key" in action
        assert "title" in action
        assert "blocker_kind" in action


# ---------------------------------------------------------------------------
# 5. Hash tables
# ---------------------------------------------------------------------------

EXPECTED_EVIDENCE = {
    "semigroup_evidence.json",
    "operator_convergence_evidence.json",
    "schwinger_limit_evidence.json",
    "os_reconstruction_evidence.json",
    "semigroup_hypotheses.json",
    "action_spec.json",
}

EXPECTED_SOURCES = {
    "ym_continuum_gap_bridge.py",
    "ym_hamiltonian_identification_evidence.py",
}


def test_evidence_hashes_present(proof_state):
    eh = proof_state["evidence_hashes"]
    assert isinstance(eh, dict)
    assert EXPECTED_EVIDENCE.issubset(eh.keys()), f"Missing: {EXPECTED_EVIDENCE - eh.keys()}"
    for name, h in eh.items():
        assert isinstance(h, str) and len(h) >= 10, f"Bad hash for {name}: {h!r}"


def test_source_hashes_present(proof_state):
    sh = proof_state["source_hashes"]
    assert isinstance(sh, dict)
    assert EXPECTED_SOURCES.issubset(sh.keys()), f"Missing: {EXPECTED_SOURCES - sh.keys()}"
    for name, h in sh.items():
        assert isinstance(h, str) and len(h) >= 10, f"Bad hash for {name}: {h!r}"


# ---------------------------------------------------------------------------
# 6. Full audit embedded
# ---------------------------------------------------------------------------

def test_full_audit_embedded(proof_state):
    fa = proof_state["full_audit"]
    assert isinstance(fa, dict)
    assert fa.get("key") == "ym_continuum_gap_bridge"
    assert fa.get("status") == "INFO"
    assert "contract" in fa
    assert "checks" in fa


# ---------------------------------------------------------------------------
# 7. On-disk artifact (if available)
# ---------------------------------------------------------------------------

def test_artifact_matches_schema_version(proof_state_artifact):
    assert proof_state_artifact["schema_version"] == "1.0.0"


def test_artifact_has_provenance_manifest():
    """The proof_state.json must have an accompanying .provenance.json."""
    base = os.path.join(os.path.dirname(__file__), "proof_state.json")
    prov = base + ".provenance.json"
    if not os.path.isfile(base):
        pytest.skip("proof_state.json not generated yet")
    assert os.path.isfile(prov), "Missing provenance manifest"
    with open(prov, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "artifact_sha256" in manifest
    assert isinstance(manifest.get("source_files_sha256"), dict)
    # Generator metadata recorded
    extra = manifest.get("extra", {})
    assert extra.get("generator") == "generate_proof_state.py"
