"""generate_proof_state.py

Generates a versioned proof-state snapshot (``proof_state.json``) that
records the current status of every obligation in the Yang--Mills
continuum-gap bridge, together with the derived scheduling views
(next-actions & blocked-actions) and provenance metadata.

Purpose
-------
The bridge module (``ym_continuum_gap_bridge.py``) evaluates obligations
dynamically.  This script persists a *frozen* snapshot as a JSON
artifact so that:

1. Progress can be tracked across repository versions.
2. The obligation tree, evidence pointers, and scheduling views are
   available offline without re-running the full audit.
3. A provenance manifest binds the snapshot to the exact source code
   that produced it.

Usage
-----
::

    cd verification
    python generate_proof_state.py          # writes proof_state.json
                                            #        proof_state.json.provenance.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

# Flat-module import convention: run from verification/
sys.path.insert(0, os.path.dirname(__file__))

from ym_continuum_gap_bridge import (
    audit_ym_continuum_gap_bridge,
    ym_continuum_gap_bridge_contract,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(os.path.abspath(__file__))
_ARTIFACT = os.path.join(_DIR, "proof_state.json")

# Evidence artifacts that feed into the bridge audit
_EVIDENCE_FILES = [
    "semigroup_evidence.json",
    "operator_convergence_evidence.json",
    "schwinger_limit_evidence.json",
    "os_reconstruction_evidence.json",
    "semigroup_hypotheses.json",
    "action_spec.json",
]

# Source modules whose content determines the snapshot
_SOURCE_FILES = [
    "ym_continuum_gap_bridge.py",
    "ym_hamiltonian_identification_evidence.py",
    "semigroup_evidence.py",
    "operator_convergence_evidence.py",
    "schwinger_limit_evidence.py",
    "os_reconstruction_evidence.py",
    "functional_analysis_gap_transfer.py",
]


def _sha256(path: str) -> str:
    """Return hex SHA-256 of a file, or 'FILE_NOT_FOUND'."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "FILE_NOT_FOUND"


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


# ---------------------------------------------------------------------------
# Obligation-tree summary helpers
# ---------------------------------------------------------------------------

def _summarise_obligations(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact summary of the obligation list from the bridge audit."""
    total = 0
    pass_count = 0
    conditional_count = 0
    by_key: Dict[str, str] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        key = check.get("key", "?")
        status = check.get("status", "UNKNOWN")
        by_key[key] = status
        total += 1
        if status == "PASS":
            pass_count += 1
        else:
            conditional_count += 1
    return {
        "total": total,
        "pass": pass_count,
        "conditional": conditional_count,
        "by_key": by_key,
    }


def _summarise_subchecks(subchecks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the nested identification subchecks tree."""
    branches: List[Dict[str, Any]] = []
    total_leaves = 0
    pass_leaves = 0
    conditional_leaves = 0
    for sc in subchecks:
        if not isinstance(sc, dict):
            continue
        leaves = sc.get("subclauses") or []
        branch_pass = 0
        branch_cond = 0
        leaf_keys: List[str] = []
        for leaf in leaves:
            if not isinstance(leaf, dict):
                continue
            total_leaves += 1
            leaf_keys.append(leaf.get("key", "?"))
            if leaf.get("status") == "PASS":
                pass_leaves += 1
                branch_pass += 1
            else:
                conditional_leaves += 1
                branch_cond += 1
        branches.append({
            "key": sc.get("key"),
            "title": sc.get("title"),
            "status": sc.get("status"),
            "theorem_role": sc.get("theorem_role"),
            "leaf_count": len(leaf_keys),
            "leaf_pass": branch_pass,
            "leaf_conditional": branch_cond,
            "leaf_keys": leaf_keys,
        })
    return {
        "branch_count": len(branches),
        "total_leaves": total_leaves,
        "pass_leaves": pass_leaves,
        "conditional_leaves": conditional_leaves,
        "branches": branches,
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_proof_state() -> Dict[str, Any]:
    """Run the bridge audit and build the proof-state snapshot."""
    audit = audit_ym_continuum_gap_bridge()
    contract = ym_continuum_gap_bridge_contract()

    checks = audit.get("checks", [])
    discharge = next(
        (c for c in checks if isinstance(c, dict) and c.get("key") == "ym_gap_bridge_discharge"),
        None,
    )
    subchecks = (
        discharge.get("diagnostics", {}).get("subchecks", [])
        if isinstance(discharge, dict) and isinstance(discharge.get("diagnostics"), dict)
        else []
    )

    # Evidence file hashes (for reproducibility)
    evidence_hashes: Dict[str, str] = {}
    for name in _EVIDENCE_FILES:
        evidence_hashes[name] = _sha256(os.path.join(_DIR, name))

    # Source file hashes
    source_hashes: Dict[str, str] = {}
    for name in _SOURCE_FILES:
        source_hashes[name] = _sha256(os.path.join(_DIR, name))

    state: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python_version": _python_version(),
        "contract": contract,
        "obligation_summary": _summarise_obligations(checks),
        "identification_subchecks_summary": _summarise_subchecks(subchecks),
        "next_actions": audit.get("next_actions", []),
        "blocked_actions": audit.get("blocked_actions", []),
        "evidence_hashes": evidence_hashes,
        "source_hashes": source_hashes,
        "full_audit": audit,
    }
    return state


def write_proof_state(path: str | None = None) -> str:
    """Generate and write proof_state.json with a provenance manifest.

    Returns the path to the written artifact.
    """
    out = path or _ARTIFACT
    state = generate_proof_state()

    with open(out, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    # Record provenance
    try:
        from provenance import record_derivation
    except ImportError:
        from .provenance import record_derivation

    source_abs = [os.path.join(_DIR, name) for name in _SOURCE_FILES]
    record_derivation(
        artifact_path=out,
        source_files=source_abs,
        extra_metadata={
            "generator": "generate_proof_state.py",
            "schema_version": state["schema_version"],
        },
    )

    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    artifact = write_proof_state()
    print(f"[proof-state] Written: {artifact}")
    print(f"[proof-state] Provenance: {artifact}.provenance.json")

    # Quick summary
    with open(artifact, "r", encoding="utf-8") as f:
        data = json.load(f)
    obl = data.get("obligation_summary", {})
    isc = data.get("identification_subchecks_summary", {})
    na = data.get("next_actions", [])
    ba = data.get("blocked_actions", [])
    print(f"\n  Obligations: {obl.get('pass', 0)}/{obl.get('total', 0)} PASS")
    print(f"  Leaf obligations: {isc.get('pass_leaves', 0)}/{isc.get('total_leaves', 0)} PASS")
    print(f"  Next actions:   {len(na)}")
    print(f"  Blocked actions: {len(ba)}")
    print(f"  Contract status: {data.get('contract', {}).get('current_status', 'unknown')}")
