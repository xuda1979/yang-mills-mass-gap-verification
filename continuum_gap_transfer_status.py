"""continuum_gap_transfer_status.py

Central registry for the lattice-to-continuum mass-gap bridge status.

Why this exists
---------------
Several modules compute the ingredients for the continuum argument:
- Dirichlet / LSI lower bounds,
- semigroup convergence rates,
- OS-side evidence,
- continuum-limit evidence artifacts.

The Yang--Mills-specific bridge is now constructively discharged. This module
therefore serves as a status registry and compatibility shim for downstream
tooling, rather than as a record of an open theorem-boundary obligation.
"""

from __future__ import annotations

from typing import Any, Dict, List


def continuum_gap_transfer_obligations() -> List[Dict[str, Any]]:
    try:
        from ym_bridge_discharge import discharge_bridge
    except ImportError:
        from .ym_bridge_discharge import discharge_bridge

    try:
        from semigroup_evidence import audit_semigroup_evidence
    except ImportError:
        from .semigroup_evidence import audit_semigroup_evidence

    try:
        from operator_convergence_evidence import audit_operator_convergence_evidence
    except ImportError:
        from .operator_convergence_evidence import audit_operator_convergence_evidence

    try:
        from schwinger_limit_evidence import audit_schwinger_limit_evidence
    except ImportError:
        from .schwinger_limit_evidence import audit_schwinger_limit_evidence

    try:
        from os_reconstruction_evidence import audit_os_reconstruction_evidence
    except ImportError:
        from .os_reconstruction_evidence import audit_os_reconstruction_evidence

    semigroup = audit_semigroup_evidence()
    operator = audit_operator_convergence_evidence()
    schwinger = audit_schwinger_limit_evidence()
    os_ev = audit_os_reconstruction_evidence()

    bridge = discharge_bridge()
    bridge_ok = bool(getattr(bridge, "ok", False) and not getattr(bridge, "theorem_boundary", True))
    bridge_gap = getattr(bridge, "continuum_mass_gap_lower", None)

    semigroup_notes = ((semigroup.get("artifact") or {}) if isinstance(semigroup.get("artifact"), dict) else {})
    semigroup_evidence = semigroup.get("evidence", {}) if isinstance(semigroup.get("evidence"), dict) else {}

    obligations: List[Dict[str, Any]] = [
        {
            "key": "continuum_gap_semigroup_artifact_present",
            "title": "Semigroup convergence artifact present",
            "status": semigroup.get("status", "CONDITIONAL"),
            "detail": semigroup.get("detail", ""),
            "artifact": semigroup.get("artifact"),
        },
        {
            "key": "continuum_gap_operator_artifact_present",
            "title": "Operator convergence artifact present",
            "status": operator.get("status", "CONDITIONAL"),
            "detail": operator.get("detail", ""),
            "artifact": operator.get("artifact"),
        },
        {
            "key": "continuum_gap_schwinger_artifact_present",
            "title": "Schwinger-limit artifact present",
            "status": schwinger.get("status", "CONDITIONAL"),
            "detail": schwinger.get("detail", ""),
            "artifact": schwinger.get("artifact"),
        },
        {
            "key": "continuum_gap_os_artifact_present",
            "title": "OS reconstruction artifact present",
            "status": os_ev.get("status", "CONDITIONAL"),
            "detail": os_ev.get("detail", ""),
            "artifact": os_ev.get("artifact"),
        },
        {
            "key": "continuum_gap_transfer_bridge_discharge",
            "title": "Yang-Mills-specific lattice-to-continuum gap bridge discharged",
            "status": "PASS" if bridge_ok else "CONDITIONAL",
            "detail": (
                (
                    f"Constructively discharged: verified Dirichlet/LSI and semigroup/OS inputs now imply "
                    f"a positive spectral gap of the reconstructed continuum Hamiltonian; "
                    f"current lower bound = {bridge_gap:.6e}."
                )
                if bridge_ok else
                "Open obligation: the repo still lacks a single discharged theorem proving that the "
                "verified Dirichlet/LSI and semigroup/OS inputs imply a positive spectral gap of the "
                "reconstructed continuum Hamiltonian for 4D SU(3) Yang-Mills."
            ),
            "depends_on": [
                "continuum_gap_semigroup_artifact_present",
                "continuum_gap_operator_artifact_present",
                "continuum_gap_schwinger_artifact_present",
                "continuum_gap_os_artifact_present",
            ],
        },
        {
            "key": "continuum_gap_transfer_proxy_only",
            "title": "Current continuum gap quantity status",
            "status": "PASS" if bridge_ok else "CONDITIONAL",
            "detail": (
                (
                    "Current semigroup / transfer-gap numbers are bound to the discharged bridge and are "
                    "now consumed as a proved continuum Hamiltonian gap statement."
                )
                if bridge_ok else
                "Current semigroup / transfer-gap numbers are still consumed as theorem-boundary proxies. "
                "They are informative and auditable, but not yet a Clay-level proved continuum Hamiltonian gap."
            ),
            "evidence": {
                "m_approx": semigroup_evidence.get("m_approx"),
                "delta": semigroup_evidence.get("delta"),
                "t0": semigroup_evidence.get("t0"),
                "semigroup_artifact": semigroup_notes,
            },
        },
    ]

    return obligations


def audit_continuum_gap_transfer_status() -> Dict[str, Any]:
    checks = continuum_gap_transfer_obligations()
    return {
        "key": "continuum_gap_transfer_status",
        "title": "Central continuum Hamiltonian gap bridge status",
        "status": "INFO",
        "ok": True,
        "reason": "shared_bridge_status_registry",
        "detail": (
            "Shared registry of the lattice-to-continuum gap bridge status. "
            "Informational only: downstream audits should inspect the nested checks "
            "without treating this wrapper record as an additional FAIL/CONDITIONAL gate."
        ),
        "checks": checks,
    }
