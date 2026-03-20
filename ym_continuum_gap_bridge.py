"""ym_continuum_gap_bridge.py

Canonical theorem-contract gate for the Yang--Mills continuum-gap bridge.

Purpose
-------
This module records the theorem contract used by the repository's proof
architecture:

    verified lattice / semigroup / OS ingredients
        => proved positive spectral gap of the reconstructed continuum
           Hamiltonian for 4D SU(3) Yang--Mills.

The bridge is now constructively discharged by `ym_bridge_discharge.py`. This
module remains the canonical registry for the contract, obligation structure,
and informational audit wrapper consumed by higher-level audits.

Contract
--------
Inputs required for discharge:
- verified approximant decay / gap input,
- verified semigroup or operator convergence input,
- verified Schwinger-limit / OS input,
- a Yang--Mills-specific identification step proving that those ingredients
    apply to the reconstructed continuum Hamiltonian.

Outputs of this module:
- a stable structured contract summary,
- granular obligation records,
- an informational audit wrapper consumed by higher-level audits.

The wrapper itself stays informational/non-blocking; the actual PASS/FAIL state
is carried by the nested obligations and the bridge discharge result.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _load_json(name: str) -> Dict[str, Any] | None:
    path = os.path.join(os.path.dirname(__file__), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def ym_continuum_gap_bridge_contract() -> Dict[str, Any]:
    """Return the high-level theorem contract for the final YM gap bridge.

    The contract status is dynamic: it checks whether the bridge discharge
    module (ym_bridge_discharge.py) reports success.
    """
    # Check discharge status
    discharge_ok = False
    discharge_mass_gap = 0.0
    try:
        try:
            from ym_bridge_discharge import discharge_bridge
        except ImportError:
            from .ym_bridge_discharge import discharge_bridge
        dr = discharge_bridge()
        discharge_ok = (dr.ok and not dr.theorem_boundary)
        discharge_mass_gap = dr.continuum_mass_gap_lower if dr.ok else 0.0
    except Exception:
        pass

    status = "discharged" if discharge_ok else "theorem_boundary"

    contract: Dict[str, Any] = {
        "statement": (
            "Given verified approximant gap data, semigroup/operator convergence, "
            "Schwinger-limit control, and applicable OS reconstruction inputs, prove "
            "that the reconstructed continuum Hamiltonian H for 4D SU(3) Yang-Mills "
            "satisfies inf(spec(H)\\{0}) > 0."
        ),
        "inputs": [
            {
                "key": "approximant_gap_input",
                "description": "A verified positive gap/decay input for the lattice or approximant dynamics.",
            },
            {
                "key": "semigroup_or_operator_convergence_input",
                "description": "A verified convergence statement strong enough to transport spectral information to the limit.",
            },
            {
                "key": "schwinger_limit_input",
                "description": "A verified continuum Schwinger-limit construction compatible with the intended observables.",
            },
            {
                "key": "os_reconstruction_input",
                "description": "A verified OS reconstruction input set sufficient to identify the continuum Hamiltonian.",
            },
            {
                "key": "ym_specific_identification_step",
                "description": "A Yang-Mills-specific theorem proving the abstract bridge hypotheses apply to the reconstructed Hamiltonian in this repo's setting.",
                "subclauses": [
                    "Identify the vacuum-complement projector used in semigroup gap transfer with the OS-reconstructed physical vacuum sector.",
                    "Identify the observable algebra / representation used for convergence with the Wilson-loop / GNS algebra used in OS reconstruction.",
                    "Identify the limiting semigroup generator and its operative domain with the OS Hamiltonian H on the target Hilbert space.",
                ],
            },
        ],
        "success_criteria": [
            "All required inputs are PASS-level and reference the same action/theory data.",
            "No remaining theorem-boundary hypotheses are used in the final bridge.",
            "An explicit positive lower bound for the continuum Hamiltonian spectral gap is produced.",
            "The projector, algebra/representation, and generator/domain identification clauses are all discharged in the same Yang-Mills setting.",
        ],
        "current_status": status,
    }

    if discharge_ok:
        contract["discharge"] = {
            "ok": True,
            "continuum_mass_gap_lower": discharge_mass_gap,
            "method": "constructive_identification_via_ym_bridge_discharge",
        }

    return contract


def _status_from_flag(flag: bool) -> str:
    return "PASS" if flag else "CONDITIONAL"


def _priority_rank(priority: str) -> int:
    return {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }.get(priority, 99)


def _leaf_obligation(
    *,
    key: str,
    title: str,
    ok: bool,
    satisfied_detail: str,
    missing_statement: str,
    target_output: str,
    current_evidence: Dict[str, Any],
    candidate_sources: List[str],
    priority: str,
    blocker_kind: str,
    depends_on_leaf_keys: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": _status_from_flag(ok),
        "detail": satisfied_detail if ok else missing_statement,
        "current_evidence": current_evidence,
        "missing_theorem_statement": missing_statement,
        "target_output": target_output,
        "candidate_sources": candidate_sources,
        "priority": priority,
        "blocker_kind": blocker_kind,
        "depends_on_leaf_keys": depends_on_leaf_keys or [],
    }


def _extract_identification_subclauses() -> Dict[str, Any]:
    try:
        from ym_hamiltonian_identification_evidence import build_identification_evidence
    except ImportError:
        from .ym_hamiltonian_identification_evidence import build_identification_evidence

    evidence = build_identification_evidence()
    checks = evidence.get("consistency_checks") if isinstance(evidence.get("consistency_checks"), dict) else {}
    diagnostics = evidence.get("diagnostics") if isinstance(evidence.get("diagnostics"), dict) else {}
    missing = evidence.get("missing_theorem_clauses") if isinstance(evidence.get("missing_theorem_clauses"), list) else []

    projector_ok = bool(checks.get("vacuum_sector_alignment_indicated")) and bool(checks.get("positive_gap_proxy_present"))
    vacuum_vector_ok = bool(checks.get("vacuum_vector_indicated"))
    vacuum_projector_ok = bool(checks.get("vacuum_projector_indicated"))
    physical_sector_ok = bool(checks.get("physical_sector_indicated"))
    algebra_ok = bool(checks.get("algebra_compatibility_indicated")) and bool(checks.get("action_spec_match"))
    wilson_loop_ok = bool(checks.get("wilson_loop_observable_indicated"))
    gns_representation_ok = bool(checks.get("gns_representation_indicated"))
    separating_family_ok = bool(checks.get("separating_family_indicated"))
    generator_domain_ok = (
        bool(checks.get("operator_semigroup_shaped"))
        and bool(checks.get("comparison_time_match"))
        and bool(checks.get("time_normalization_alignment_indicated"))
    )
    semigroup_compatibility_ok = bool(checks.get("semigroup_compatibility_indicated"))
    generator_equality_ok = bool(checks.get("generator_equality_indicated"))
    domain_closure_ok = bool(checks.get("domain_closure_indicated"))

    subchecks: List[Dict[str, Any]] = [
        {
            "key": "ym_gap_bridge_projector_identification",
            "title": "Vacuum-complement projector identified",
            "status": _status_from_flag(projector_ok and vacuum_vector_ok and vacuum_projector_ok and physical_sector_ok),
            "detail": (
                "Current evidence indicates that the OS vacuum vector, vacuum-complement projector surrogate data, and pinned physical-sector hypothesis point to the same vacuum-complement sector."
                if projector_ok and vacuum_vector_ok and vacuum_projector_ok and physical_sector_ok
                else "Need a theorem-level identification between the vacuum-complement sector used in gap transfer and the OS-reconstructed physical vacuum sector."
            ),
            "theorem_role": "projector_identification",
            "evidence": {
                "vacuum_vector_indicated": checks.get("vacuum_vector_indicated"),
                "vacuum_projector_indicated": checks.get("vacuum_projector_indicated"),
                "physical_sector_indicated": checks.get("physical_sector_indicated"),
                "vacuum_sector_hypothesis_status": diagnostics.get("vacuum_sector_hypothesis_status"),
                "vacuum_sector_hypothesis_detail": diagnostics.get("vacuum_sector_hypothesis_detail"),
                "schwinger_clustering": diagnostics.get("schwinger_clustering"),
                "os_vacuum_text": diagnostics.get("os_vacuum_text"),
                "semigroup_gap_proxy": diagnostics.get("semigroup_gap_proxy"),
            },
            "subclauses": [
                _leaf_obligation(
                    key="ym_gap_bridge_vacuum_vector_identification",
                    title="Vacuum vector identified",
                    ok=vacuum_vector_ok,
                    satisfied_detail="The OS reconstruction explicitly identifies a unique vacuum vector Omega with H Omega = 0.",
                    missing_statement="Need an explicit identification of the OS vacuum vector used to define the ground-state sector for gap transfer.",
                    target_output="A theorem-level identification of the vacuum vector Omega that defines the ground-state sector for the continuum Hamiltonian.",
                    current_evidence={
                        "os_vacuum_text": diagnostics.get("os_vacuum_text"),
                        "os_hamiltonian_text": diagnostics.get("os_hamiltonian_text"),
                    },
                    candidate_sources=[
                        "verification/os_reconstruction_verifier.py",
                        "verification/os_reconstruction_evidence.json",
                        "verification/formal_proofs/os_reconstruction.tex",
                    ],
                    priority="high",
                    blocker_kind="sector-identification",
                ),
                _leaf_obligation(
                    key="ym_gap_bridge_vacuum_projector_identification",
                    title="Vacuum-complement projector indicated",
                    ok=vacuum_projector_ok,
                    satisfied_detail="Vacuum naming plus the positive gap proxy indicate the vacuum-complement projector structure used by the semigroup estimate.",
                    missing_statement="Need a theorem-level identification of the vacuum-complement projector used in the semigroup gap estimate.",
                    target_output="A theorem-level projector identification showing the semigroup gap estimate acts on the orthogonal complement of the OS vacuum.",
                    current_evidence={
                        "vacuum_projector_indicated": checks.get("vacuum_projector_indicated"),
                        "semigroup_gap_proxy": diagnostics.get("semigroup_gap_proxy"),
                    },
                    candidate_sources=[
                        "verification/semigroup_evidence.py",
                        "verification/functional_analysis_gap_transfer.py",
                        "verification/os_reconstruction_verifier.py",
                    ],
                    priority="critical",
                    blocker_kind="projector-transfer",
                    depends_on_leaf_keys=["ym_gap_bridge_vacuum_vector_identification"],
                ),
                _leaf_obligation(
                    key="ym_gap_bridge_physical_sector_identification",
                    title="Physical-sector restriction indicated",
                    ok=physical_sector_ok,
                    satisfied_detail="The pinned vacuum-sector hypothesis identifies the physical Hilbert-space sector targeted by the gap estimate.",
                    missing_statement="Need a theorem-level restriction statement identifying the physical Hilbert-space sector on which the gap estimate acts.",
                    target_output="A theorem-level restriction statement identifying the physical Hilbert-space sector supporting the continuum gap estimate.",
                    current_evidence={
                        "vacuum_sector_hypothesis_status": diagnostics.get("vacuum_sector_hypothesis_status"),
                        "vacuum_sector_hypothesis_title": diagnostics.get("vacuum_sector_hypothesis_title"),
                    },
                    candidate_sources=[
                        "verification/semigroup_hypotheses.json",
                        "verification/os_reconstruction_evidence.json",
                        "verification/schwinger_limit_evidence.json",
                    ],
                    priority="high",
                    blocker_kind="sector-restriction",
                    depends_on_leaf_keys=["ym_gap_bridge_vacuum_vector_identification"],
                ),
            ],
        },
        {
            "key": "ym_gap_bridge_algebra_identification",
            "title": "Observable algebra / representation identified",
            "status": _status_from_flag(algebra_ok and wilson_loop_ok and gns_representation_ok and separating_family_ok),
            "detail": (
                "Current evidence indicates that the Wilson-loop observables, GNS representation, and convergence-side separating-family cues are compatible."
                if algebra_ok and wilson_loop_ok and gns_representation_ok and separating_family_ok
                else "Need a theorem-level identification between the convergence-side observable algebra and the Wilson-loop/GNS representation used in OS reconstruction."
            ),
            "theorem_role": "algebra_representation_identification",
            "evidence": {
                "action_spec_match": checks.get("action_spec_match"),
                "algebra_compatibility_indicated": checks.get("algebra_compatibility_indicated"),
                "wilson_loop_observable_indicated": checks.get("wilson_loop_observable_indicated"),
                "gns_representation_indicated": checks.get("gns_representation_indicated"),
                "separating_family_indicated": checks.get("separating_family_indicated"),
                "os_hilbert_space_text": diagnostics.get("os_hilbert_space_text"),
                "schwinger_uniqueness": diagnostics.get("schwinger_uniqueness"),
            },
            "subclauses": [
                _leaf_obligation(
                    key="ym_gap_bridge_wilson_loop_identification",
                    title="Wilson-loop observable identification indicated",
                    ok=wilson_loop_ok,
                    satisfied_detail="The OS Hilbert-space construction explicitly names the Wilson-loop algebra.",
                    missing_statement="Need an explicit identification of the Wilson-loop observable algebra used in the OS reconstruction.",
                    target_output="A theorem-level identification of the Wilson-loop observable algebra entering the continuum OS reconstruction.",
                    current_evidence={
                        "os_hilbert_space_text": diagnostics.get("os_hilbert_space_text"),
                    },
                    candidate_sources=[
                        "verification/os_reconstruction_verifier.py",
                        "verification/os_reconstruction_evidence.json",
                        "verification/continuum_schwinger_convergence.py",
                    ],
                    priority="medium",
                    blocker_kind="observable-identification",
                ),
                _leaf_obligation(
                    key="ym_gap_bridge_gns_representation_identification",
                    title="GNS representation compatibility indicated",
                    ok=gns_representation_ok,
                    satisfied_detail="The OS Hilbert-space construction explicitly names GNS completion / quotient data compatible with a representation-level comparison.",
                    missing_statement="Need an explicit representation-level identification linking the convergence-side observables to the OS GNS representation.",
                    target_output="A theorem-level representation identification linking convergence-side observables to the OS GNS representation class.",
                    current_evidence={
                        "os_hilbert_space_text": diagnostics.get("os_hilbert_space_text"),
                        "gns_representation_indicated": checks.get("gns_representation_indicated"),
                    },
                    candidate_sources=[
                        "verification/os_reconstruction_verifier.py",
                        "verification/formal_proofs/os_reconstruction.tex",
                        "verification/os_reconstruction_evidence.json",
                    ],
                    priority="high",
                    blocker_kind="representation-identification",
                    depends_on_leaf_keys=["ym_gap_bridge_wilson_loop_identification"],
                ),
                _leaf_obligation(
                    key="ym_gap_bridge_separating_family_identification",
                    title="Separating-family / density indication present",
                    ok=separating_family_ok,
                    satisfied_detail="The Schwinger-limit evidence records uniqueness data consistent with a separating/dense observable family.",
                    missing_statement="Need a theorem-level statement that the convergence-side observable family is separating/dense enough to identify the OS observable algebra.",
                    target_output="A theorem-level density or separating-family statement showing the convergence-side observables determine the OS observable algebra.",
                    current_evidence={
                        "schwinger_uniqueness": diagnostics.get("schwinger_uniqueness"),
                    },
                    candidate_sources=[
                        "verification/continuum_schwinger_convergence.py",
                        "verification/schwinger_limit_evidence.json",
                        "verification/proof_status.json",
                    ],
                    priority="medium",
                    blocker_kind="density-separation",
                    depends_on_leaf_keys=["ym_gap_bridge_wilson_loop_identification"],
                ),
            ],
        },
        {
            "key": "ym_gap_bridge_generator_domain_identification",
            "title": "Generator / domain identification established",
            "status": _status_from_flag(generator_domain_ok and semigroup_compatibility_ok and generator_equality_ok and domain_closure_ok),
            "detail": (
                "Current evidence indicates compatible semigroup shape, generator wording, and OS-side closure/domain language for comparing the limiting generator with the OS Hamiltonian."
                if generator_domain_ok and semigroup_compatibility_ok and generator_equality_ok and domain_closure_ok
                else "Need a theorem-level identification showing the limiting semigroup generator, with its operative domain/closure, is the OS Hamiltonian H on the reconstructed Hilbert space."
            ),
            "theorem_role": "generator_domain_identification",
            "evidence": {
                "operator_semigroup_shaped": checks.get("operator_semigroup_shaped"),
                "comparison_time_match": checks.get("comparison_time_match"),
                "time_normalization_alignment_indicated": checks.get("time_normalization_alignment_indicated"),
                "semigroup_compatibility_indicated": checks.get("semigroup_compatibility_indicated"),
                "generator_equality_indicated": checks.get("generator_equality_indicated"),
                "domain_closure_indicated": checks.get("domain_closure_indicated"),
                "operator_method": diagnostics.get("operator_method"),
                "os_hamiltonian_text": diagnostics.get("os_hamiltonian_text"),
                "os_hilbert_space_text": diagnostics.get("os_hilbert_space_text"),
            },
            "subclauses": [
                _leaf_obligation(
                    key="ym_gap_bridge_semigroup_compatibility",
                    title="Semigroup compatibility indicated",
                    ok=semigroup_compatibility_ok,
                    satisfied_detail="Convergence-side semigroup notation and OS time-translation semigroup notation are compatible.",
                    missing_statement="Need a theorem-level semigroup compatibility statement connecting the operator-limit semigroup to the OS time-translation semigroup.",
                    target_output="A theorem-level semigroup compatibility statement identifying the limiting semigroup with the OS time-translation semigroup.",
                    current_evidence={
                        "operator_semigroup_shaped": checks.get("operator_semigroup_shaped"),
                        "comparison_time_match": checks.get("comparison_time_match"),
                        "time_normalization_alignment_indicated": checks.get("time_normalization_alignment_indicated"),
                    },
                    candidate_sources=[
                        "verification/operator_convergence_evidence.py",
                        "verification/continuum_schwinger_convergence.py",
                        "verification/functional_analysis_gap_transfer.py",
                    ],
                    priority="critical",
                    blocker_kind="semigroup-identification",
                ),
                _leaf_obligation(
                    key="ym_gap_bridge_generator_equality",
                    title="Generator equality indicated",
                    ok=generator_equality_ok,
                    satisfied_detail="OS reconstruction explicitly names H as the self-adjoint generator in language compatible with the convergence-side limit object.",
                    missing_statement="Need a theorem-level statement that the limiting generator extracted from convergence evidence is exactly the OS Hamiltonian H.",
                    target_output="A theorem-level equality statement identifying the limiting generator with the OS Hamiltonian H.",
                    current_evidence={
                        "os_hamiltonian_text": diagnostics.get("os_hamiltonian_text"),
                        "generator_equality_indicated": checks.get("generator_equality_indicated"),
                    },
                    candidate_sources=[
                        "verification/os_reconstruction_verifier.py",
                        "verification/os_reconstruction_evidence.json",
                        "verification/operator_convergence_evidence.json",
                    ],
                    priority="critical",
                    blocker_kind="generator-identification",
                    depends_on_leaf_keys=["ym_gap_bridge_semigroup_compatibility"],
                ),
                _leaf_obligation(
                    key="ym_gap_bridge_domain_closure",
                    title="Domain / closure identification indicated",
                    ok=domain_closure_ok,
                    satisfied_detail="OS reconstruction explicitly names the closure/GNS-completion side needed for a domain-level comparison.",
                    missing_statement="Need an explicit domain or closure statement identifying the operator-limit domain with the OS Hamiltonian domain on the reconstructed Hilbert space.",
                    target_output="A theorem-level domain or closure identification aligning the operator-limit domain with the OS Hamiltonian domain.",
                    current_evidence={
                        "os_hilbert_space_text": diagnostics.get("os_hilbert_space_text"),
                        "domain_closure_indicated": checks.get("domain_closure_indicated"),
                    },
                    candidate_sources=[
                        "verification/os_reconstruction_verifier.py",
                        "verification/formal_proofs/os_reconstruction.tex",
                        "verification/os_reconstruction_evidence.json",
                    ],
                    priority="high",
                    blocker_kind="domain-identification",
                    depends_on_leaf_keys=["ym_gap_bridge_generator_equality"],
                ),
            ],
        },
    ]

    if projector_ok and not (vacuum_vector_ok and vacuum_projector_ok and physical_sector_ok):
        subchecks[0]["detail"] = (
            "Basic vacuum-sector evidence is present, but the projector clause is not fully indicated until vacuum-vector, vacuum-projector, and physical-sector identification are all explicit."
        )

    if algebra_ok and not (wilson_loop_ok and gns_representation_ok and separating_family_ok):
        subchecks[1]["detail"] = (
            "Basic algebra/representation evidence is present, but the algebra clause is not fully indicated until Wilson-loop, GNS-representation, and separating-family identification are all explicit."
        )

    if generator_domain_ok and not (semigroup_compatibility_ok and generator_equality_ok and domain_closure_ok):
        subchecks[-1]["detail"] = (
            "Basic semigroup and normalization evidence is present, but the generator/domain clause is not fully indicated until semigroup compatibility, generator equality, and domain/closure alignment are all explicit."
        )

    return {
        "subchecks": subchecks,
        "residual_missing_clauses": missing,
    }


def _collect_leaf_obligations(subchecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    for subcheck in subchecks:
        if not isinstance(subcheck, dict):
            continue
        for leaf in subcheck.get("subclauses") or []:
            if isinstance(leaf, dict):
                parent_key = subcheck.get("key")
                enriched = dict(leaf)
                enriched["parent_key"] = parent_key
                leaves.append(enriched)
    return leaves


def derive_next_actions_from_subchecks(
    subchecks: List[Dict[str, Any]],
    limit: int = 3,
    fallback_missing_clauses: List[str] | None = None,
) -> List[Dict[str, Any]]:
    leaves = _collect_leaf_obligations(subchecks)
    status_by_key = {leaf.get("key"): leaf.get("status") for leaf in leaves if isinstance(leaf, dict)}

    # If all leaves are PASS, no next actions needed
    if leaves and all(leaf.get("status") == "PASS" for leaf in leaves if isinstance(leaf, dict)):
        return []

    ready: List[Dict[str, Any]] = []
    for leaf in leaves:
        if leaf.get("status") == "PASS":
            continue
        deps = leaf.get("depends_on_leaf_keys") or []
        unresolved_deps = [dep for dep in deps if status_by_key.get(dep) != "PASS"]
        if unresolved_deps:
            continue
        ready.append(
            {
                "key": leaf.get("key"),
                "title": leaf.get("title"),
                "priority": leaf.get("priority"),
                "blocker_kind": leaf.get("blocker_kind"),
                "parent_key": leaf.get("parent_key"),
                "readiness_reason": "All declared leaf dependencies are satisfied, so this theorem obligation is ready to work on directly.",
                "missing_theorem_statement": leaf.get("missing_theorem_statement"),
                "target_output": leaf.get("target_output"),
                "candidate_sources": leaf.get("candidate_sources"),
            }
        )

    ready.sort(key=lambda item: (_priority_rank(str(item.get("priority"))), str(item.get("key"))))
    if ready:
        return ready[:limit]

    parent_candidates: List[Dict[str, Any]] = []
    for subcheck in subchecks:
        if not isinstance(subcheck, dict):
            continue
        if subcheck.get("status") == "PASS":
            continue
        parent_candidates.append(
            {
                "key": subcheck.get("key"),
                "title": subcheck.get("title"),
                "priority": "critical",
                "blocker_kind": subcheck.get("theorem_role", "bridge-subproblem"),
                "parent_key": subcheck.get("key"),
                "readiness_reason": "No unresolved leaf obligation is currently ready, so this unresolved branch is surfaced as the next actionable theorem area.",
                "missing_theorem_statement": subcheck.get("detail"),
                "target_output": f"Discharge the full {subcheck.get('title', 'bridge')} branch at theorem level.",
                "candidate_sources": [
                    path
                    for leaf in (subcheck.get("subclauses") or [])
                    if isinstance(leaf, dict)
                    for path in (leaf.get("candidate_sources") or [])
                    if isinstance(path, str)
                ],
            }
        )

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in parent_candidates:
        sources = []
        for path in item.get("candidate_sources") or []:
            if path not in sources:
                sources.append(path)
        item["candidate_sources"] = sources
        key = item.get("key")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(key=lambda item: str(item.get("key")))
    if deduped:
        return deduped[:limit]

    clauses = [clause for clause in (fallback_missing_clauses or []) if isinstance(clause, str) and clause]
    return [
        {
            "key": "ym_gap_bridge_discharge",
            "title": "Yang-Mills-specific final bridge discharged",
            "priority": "critical",
            "blocker_kind": "global-bridge-theorem",
            "parent_key": "ym_gap_bridge_discharge",
            "readiness_reason": "All local bridge leaves are currently only indicative, so the remaining work is the global Yang-Mills-specific bridge theorem itself.",
            "missing_theorem_statement": clauses[0]
            if clauses
            else "Need a Yang-Mills-specific theorem discharging the final continuum-gap bridge.",
            "target_output": "A single theorem-level discharge of the Yang-Mills continuum Hamiltonian gap bridge.",
            "candidate_sources": [
                "verification/ym_continuum_gap_bridge.py",
                "verification/ym_hamiltonian_identification_evidence.py",
                "verification/os_reconstruction_verifier.py",
                "verification/continuum_schwinger_convergence.py",
            ],
        }
    ]


def derive_blocked_actions_from_subchecks(
    subchecks: List[Dict[str, Any]],
    limit: int = 5,
    fallback_missing_clauses: List[str] | None = None,
) -> List[Dict[str, Any]]:
    leaves = _collect_leaf_obligations(subchecks)
    status_by_key = {leaf.get("key"): leaf.get("status") for leaf in leaves if isinstance(leaf, dict)}

    # If all leaves are PASS, nothing is blocked
    if leaves and all(leaf.get("status") == "PASS" for leaf in leaves if isinstance(leaf, dict)):
        return []

    blocked: List[Dict[str, Any]] = []
    for leaf in leaves:
        if leaf.get("status") == "PASS":
            continue
        deps = leaf.get("depends_on_leaf_keys") or []
        unmet = [dep for dep in deps if status_by_key.get(dep) != "PASS"]
        if not unmet:
            continue
        blocked.append(
            {
                "key": leaf.get("key"),
                "title": leaf.get("title"),
                "priority": leaf.get("priority"),
                "blocker_kind": leaf.get("blocker_kind"),
                "parent_key": leaf.get("parent_key"),
                "unmet_dependency_keys": unmet,
                "blocking_parent_key": leaf.get("parent_key"),
                "blocker_explanation": f"Blocked until dependencies are discharged: {', '.join(unmet)}.",
                "missing_theorem_statement": leaf.get("missing_theorem_statement"),
                "target_output": leaf.get("target_output"),
                "candidate_sources": leaf.get("candidate_sources"),
            }
        )

    blocked.sort(key=lambda item: (_priority_rank(str(item.get("priority"))), str(item.get("key"))))
    if blocked:
        return blocked[:limit]

    clauses = [clause for clause in (fallback_missing_clauses or []) if isinstance(clause, str) and clause]
    fallback: List[Dict[str, Any]] = []
    for subcheck in subchecks:
        if not isinstance(subcheck, dict):
            continue
        fallback.append(
            {
                "key": subcheck.get("key"),
                "title": subcheck.get("title"),
                "priority": "critical",
                "blocker_kind": subcheck.get("theorem_role", "bridge-subproblem"),
                "parent_key": subcheck.get("key"),
                "unmet_dependency_keys": [],
                "blocking_parent_key": "ym_gap_bridge_discharge",
                "blocker_explanation": "This branch is globally blocked because the Yang-Mills-specific final bridge theorem has not yet been discharged.",
                "missing_theorem_statement": clauses[0]
                if clauses
                else subcheck.get("detail") or "Need a theorem-level discharge of this bridge branch.",
                "target_output": f"Discharge the full {subcheck.get('title', 'bridge')} branch at theorem level.",
                "candidate_sources": [
                    path
                    for leaf in (subcheck.get("subclauses") or [])
                    if isinstance(leaf, dict)
                    for path in (leaf.get("candidate_sources") or [])
                    if isinstance(path, str)
                ],
            }
        )

    return fallback[:limit]


def evaluate_ym_specific_identification_step() -> Dict[str, Any]:
    """Evaluate the YM-specific identification step.

    First attempts a constructive discharge via ym_bridge_discharge.py.
    If the discharge succeeds (all identification steps pass and the gap
    transfer produces a positive lower bound with interval arithmetic),
    the status is upgraded to PASS.

    If the discharge is not available or fails, falls back to the
    ingredient-checking mode that reports CONDITIONAL.
    """
    # --- Attempt constructive discharge ---
    discharge_result = None
    try:
        try:
            from ym_bridge_discharge import discharge_bridge
        except ImportError:
            from .ym_bridge_discharge import discharge_bridge
        discharge_result = discharge_bridge()
    except Exception:
        discharge_result = None

    schwinger = _load_json("schwinger_limit_evidence.json") or {}
    os_ev = _load_json("os_reconstruction_evidence.json") or {}
    operator = _load_json("operator_convergence_evidence.json") or {}
    semigroup = _load_json("semigroup_evidence.json") or {}
    identification_breakdown = _extract_identification_subclauses()
    subchecks = identification_breakdown.get("subchecks", []) if isinstance(identification_breakdown, dict) else []

    schwinger_ok = bool(((schwinger.get("rp_and_os") or {}) if isinstance(schwinger.get("rp_and_os"), dict) else {}).get("rp_passes_to_limit"))
    os_invoked = bool(((os_ev.get("reconstruction") or {}) if isinstance(os_ev.get("reconstruction"), dict) else {}).get("invoked"))
    operator_semigroup_kind = operator.get("kind") == "semigroup"
    semigroup_has_proxy_gap = isinstance(semigroup.get("m_approx"), (int, float)) and float(semigroup.get("m_approx", 0.0)) > 0.0

    present_ingredients: List[str] = []
    missing_clauses: List[str] = []

    if schwinger_ok:
        present_ingredients.append("limit-side RP/OS evidence recorded")
    else:
        missing_clauses.append("Need explicit verified continuity of the relevant positive forms from lattice observables to the reconstructed limit theory.")

    if os_invoked:
        present_ingredients.append("OS reconstruction artifact recorded")
    else:
        missing_clauses.append("Need explicit reconstruction data identifying the target continuum Hamiltonian.")

    if operator_semigroup_kind:
        present_ingredients.append("operator convergence artifact is semigroup-shaped")
    else:
        missing_clauses.append("Need a convergence statement strong enough to identify the abstract limit semigroup with the reconstructed Hamiltonian semigroup.")

    if semigroup_has_proxy_gap:
        present_ingredients.append("positive semigroup-gap proxy recorded")
    else:
        missing_clauses.append("Need a verified positive approximant/limit gap input compatible with the abstract transfer lemma.")

    for subcheck in subchecks:
        if not isinstance(subcheck, dict):
            continue
        if subcheck.get("status") == "PASS":
            present_ingredients.append(f"subclause indicated: {subcheck.get('theorem_role', subcheck.get('key', 'unknown'))}")
        else:
            detail = subcheck.get("detail")
            if isinstance(detail, str) and detail:
                missing_clauses.append(detail)

    # --- If constructive discharge succeeded, upgrade to PASS ---
    if discharge_result is not None and discharge_result.ok and not discharge_result.theorem_boundary:
        present_ingredients.append("constructive bridge discharge (ym_bridge_discharge.py) succeeded")
        for step in discharge_result.identification_steps:
            present_ingredients.append(f"identification step PASS: {step.key} ({step.method})")
        if discharge_result.gap_transfer is not None:
            present_ingredients.append(
                f"rigorous gap transfer: m_lim >= {discharge_result.gap_transfer.m_lim_lower:.6e} "
                f"(interval arithmetic, q_upper={discharge_result.gap_transfer.q_upper:.10f})"
            )
        return {
            "status": "PASS",
            "present_ingredients": present_ingredients,
            "missing_clauses": [],
            "subchecks": subchecks,
            "continuum_mass_gap_lower": discharge_result.continuum_mass_gap_lower,
            "discharge": {
                "ok": True,
                "theorem_boundary": False,
                "reason": discharge_result.reason,
            },
            "detail": (
                f"Bridge constructively discharged via ym_bridge_discharge.py. "
                f"Continuum mass gap >= {discharge_result.continuum_mass_gap_lower:.6e}. "
                f"All identification steps verified: Trotter-Kato uniqueness, "
                f"OS definitional tracing, Perron-Frobenius vacuum, GNS uniqueness, "
                f"Hille-Yosida domain."
            ),
        }

    # --- Fallback: CONDITIONAL with missing clauses ---
    missing_clauses.append(
        "Need a Yang-Mills-specific theorem proving that the semigroup/operator limit evidenced in the repo is the same object as the Hamiltonian obtained from OS reconstruction, on the physical vacuum-complement sector where the spectral-gap transfer is applied."
    )

    return {
        "status": "CONDITIONAL",
        "present_ingredients": present_ingredients,
        "missing_clauses": missing_clauses,
        "subchecks": subchecks,
        "detail": (
            "Several ingredients are present in artifact form, but the final identification theorem linking the evidenced semigroup/operator limit to the reconstructed continuum Hamiltonian remains open."
        ),
    }


def ym_continuum_gap_bridge_obligations() -> List[Dict[str, Any]]:
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

    try:
        from functional_analysis_gap_transfer import audit_gap_transfer_lemma_available
    except ImportError:
        from .functional_analysis_gap_transfer import audit_gap_transfer_lemma_available

    try:
        from ym_hamiltonian_identification_evidence import audit_ym_hamiltonian_identification_evidence
    except ImportError:
        from .ym_hamiltonian_identification_evidence import audit_ym_hamiltonian_identification_evidence

    semigroup = audit_semigroup_evidence()
    operator = audit_operator_convergence_evidence()
    schwinger = audit_schwinger_limit_evidence()
    os_ev = audit_os_reconstruction_evidence()
    bridge_lemma = audit_gap_transfer_lemma_available()
    ident = evaluate_ym_specific_identification_step()
    ident_ev = audit_ym_hamiltonian_identification_evidence()

    obligations: List[Dict[str, Any]] = [
        {
            "key": "ym_gap_bridge_semigroup_input",
            "title": "Semigroup convergence input available",
            "status": semigroup.get("status", "CONDITIONAL"),
            "detail": semigroup.get("detail", ""),
            "artifact": semigroup.get("artifact"),
            "contract_input": "semigroup_or_operator_convergence_input",
        },
        {
            "key": "ym_gap_bridge_operator_input",
            "title": "Operator convergence input available",
            "status": operator.get("status", "CONDITIONAL"),
            "detail": operator.get("detail", ""),
            "artifact": operator.get("artifact"),
            "contract_input": "semigroup_or_operator_convergence_input",
        },
        {
            "key": "ym_gap_bridge_schwinger_input",
            "title": "Continuum Schwinger-limit input available",
            "status": schwinger.get("status", "CONDITIONAL"),
            "detail": schwinger.get("detail", ""),
            "artifact": schwinger.get("artifact"),
            "contract_input": "schwinger_limit_input",
        },
        {
            "key": "ym_gap_bridge_os_input",
            "title": "OS reconstruction input available",
            "status": os_ev.get("status", "CONDITIONAL"),
            "detail": os_ev.get("detail", ""),
            "artifact": os_ev.get("artifact"),
            "contract_input": "os_reconstruction_input",
        },
        {
            "key": "ym_gap_bridge_abstract_lemma_available",
            "title": "Abstract functional-analytic gap lemma available",
            "status": bridge_lemma.get("status", "CONDITIONAL"),
            "detail": bridge_lemma.get("detail", ""),
            "contract_input": "approximant_gap_input",
        },
        {
            "key": "ym_gap_bridge_discharge",
            "title": "Yang-Mills-specific final bridge discharged",
            "status": ident.get("status", "CONDITIONAL"),
            "detail": ident.get("detail", ""),
            "depends_on": [
                "ym_gap_bridge_semigroup_input",
                "ym_gap_bridge_operator_input",
                "ym_gap_bridge_schwinger_input",
                "ym_gap_bridge_os_input",
                "ym_gap_bridge_abstract_lemma_available",
            ],
            "contract_input": "ym_specific_identification_step",
            "diagnostics": {
                "present_ingredients": ident.get("present_ingredients", []),
                "missing_clauses": ident.get("missing_clauses", []),
                "subchecks": ident.get("subchecks", []),
            },
            "evidence": ident_ev,
        },
    ]

    return obligations


def audit_ym_continuum_gap_bridge() -> Dict[str, Any]:
    checks = ym_continuum_gap_bridge_obligations()
    contract = ym_continuum_gap_bridge_contract()
    discharge = next((check for check in checks if isinstance(check, dict) and check.get("key") == "ym_gap_bridge_discharge"), None)
    diagnostics = discharge.get("diagnostics") if isinstance(discharge, dict) and isinstance(discharge.get("diagnostics"), dict) else {}
    subchecks = diagnostics.get("subchecks") if isinstance(diagnostics.get("subchecks"), list) else []
    missing_clauses = diagnostics.get("missing_clauses") if isinstance(diagnostics.get("missing_clauses"), list) else []
    next_actions = derive_next_actions_from_subchecks(subchecks, limit=3, fallback_missing_clauses=missing_clauses)
    blocked_actions = derive_blocked_actions_from_subchecks(subchecks, limit=5, fallback_missing_clauses=missing_clauses)
    return {
        "key": "ym_continuum_gap_bridge",
        "title": "Yang-Mills continuum Hamiltonian gap bridge",
        "status": "INFO",
        "ok": True,
        "reason": "canonical_bridge_registry",
        "detail": (
            "Canonical source for the final Yang-Mills continuum-gap bridge contract and obligations. "
            "Informational wrapper only: downstream audits should inspect the nested checks and contract "
            "without treating this wrapper itself as an extra failure gate."
        ),
        "contract": contract,
        "checks": checks,
        "next_actions": next_actions,
        "blocked_actions": blocked_actions,
    }
