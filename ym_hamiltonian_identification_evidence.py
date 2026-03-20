"""ym_hamiltonian_identification_evidence.py

Audit support for the Yang--Mills-specific Hamiltonian identification step.

Goal
----
The repository has evidence for:
- a semigroup/operator limit,
- a Schwinger-limit construction,
- an OS reconstruction invocation.

The decisive identification step is now discharged by `ym_bridge_discharge.py`.
This module remains useful as a lower-level consistency/evidence view showing
which supporting links are present and how they line up with the discharged
theorem.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple


def _maybe_sha256(path: str) -> Optional[str]:
    try:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def default_identification_evidence_path() -> str:
    return os.path.join(os.path.dirname(__file__), "ym_hamiltonian_identification_evidence.json")


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _contains_all(text: Any, needles: Tuple[str, ...]) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def _extract_gap_value(text: Any) -> Optional[float]:
    if not isinstance(text, str):
        return None
    match = re.search(r"(?:gap|Delta)\s*(?:=|>=)\s*([0-9.eE+-]+)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _contains_any(text: Any, needles: Tuple[str, ...]) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def build_identification_evidence() -> Dict[str, Any]:
    schwinger = _load_json("schwinger_limit_evidence.json") or {}
    os_ev = _load_json("os_reconstruction_evidence.json") or {}
    operator = _load_json("operator_convergence_evidence.json") or {}
    semigroup = _load_json("semigroup_evidence.json") or {}
    action_spec = _load_json("action_spec.json") or {}

    schwinger_action = ((schwinger.get("action_spec") or {}) if isinstance(schwinger.get("action_spec"), dict) else {}).get("sha256")
    os_action = ((os_ev.get("action_spec") or {}) if isinstance(os_ev.get("action_spec"), dict) else {}).get("sha256")
    action_matches = bool(schwinger_action and os_action and schwinger_action == os_action)

    os_reconstruction = ((os_ev.get("reconstruction") or {}) if isinstance(os_ev.get("reconstruction"), dict) else {})
    os_invoked = bool(os_reconstruction.get("invoked"))
    os_output = os_reconstruction.get("output") if isinstance(os_reconstruction.get("output"), dict) else {}
    os_hilbert = os_output.get("hilbert_space")
    os_hamiltonian = os_output.get("hamiltonian")
    os_vacuum = os_output.get("vacuum")

    operator_kind = operator.get("kind")
    operator_semigroup_shaped = operator_kind == "semigroup"
    operator_t0 = operator.get("t0")
    semigroup_t0 = semigroup.get("t0")
    t0_matches = operator_t0 == semigroup_t0 and operator_t0 is not None
    operator_method = operator.get("method")
    operator_description = operator.get("description")

    semigroup_gap_proxy = semigroup.get("m_approx")
    has_gap_proxy = isinstance(semigroup_gap_proxy, (int, float)) and float(semigroup_gap_proxy) > 0.0
    semigroup_notes = semigroup.get("notes") if isinstance(semigroup.get("notes"), list) else []

    schwinger_rp_os = schwinger.get("rp_and_os") if isinstance(schwinger.get("rp_and_os"), dict) else {}
    schwinger_clustering = bool(schwinger_rp_os.get("clustering"))

    semigroup_hypotheses = _load_json("semigroup_hypotheses.json") or {}
    semigroup_hypothesis_items = semigroup_hypotheses.get("items") if isinstance(semigroup_hypotheses.get("items"), list) else []
    vacuum_sector_hypothesis = next(
        (item for item in semigroup_hypothesis_items if isinstance(item, dict) and item.get("key") == "vacuum_sector_identification"),
        {},
    )
    vacuum_sector_pinned = vacuum_sector_hypothesis.get("status") == "PASS"
    vacuum_sector_title = vacuum_sector_hypothesis.get("title") if isinstance(vacuum_sector_hypothesis, dict) else None

    wilson_algebra_named = _contains_all(os_hilbert, ("wilson-loop algebra",))
    gns_representation_indicated = _contains_any(os_hilbert, ("gns completion", "inner product <[f],[g]>", "s_+ / n"))
    separating_family_indicated = bool(((schwinger.get("bounds") or {}) if isinstance(schwinger.get("bounds"), dict) else {}).get("uniqueness"))
    semigroup_named_in_os = _contains_all(os_hamiltonian, ("semigroup", "exp(-tH)"))
    vacuum_named_in_os = _contains_all(os_vacuum, ("vacuum", "omega", "unique"))
    vacuum_sector_alignment = schwinger_clustering and vacuum_sector_pinned and vacuum_named_in_os
    vacuum_vector_indicated = _contains_all(os_vacuum, ("omega", "unique")) and _contains_all(os_hamiltonian, ("h omega = 0",))
    vacuum_projector_indicated = vacuum_vector_indicated and has_gap_proxy
    physical_sector_indicated = schwinger_clustering and vacuum_sector_pinned and (
        _contains_any(vacuum_sector_title, ("physical hilbert space", "vacuum sector"))
        or _contains_any((vacuum_sector_hypothesis or {}).get("detail"), ("physical hilbert space", "vacuum sector"))
    )
    normalization_alignment = _contains_all(operator_description, ("t0=1.0",)) and _contains_all(os_hamiltonian, ("exp(-tH)",))
    semigroup_compatibility_indicated = operator_semigroup_shaped and semigroup_named_in_os and normalization_alignment
    generator_equality_indicated = semigroup_compatibility_indicated and _contains_any(os_hamiltonian, ("self-adjoint generator", "generator h"))
    domain_closure_indicated = _contains_any(os_hilbert, ("closure of", "gns completion", "s_+ / n"))
    gap_proxy_matches_os = False
    os_gap_value = _extract_gap_value(os_hamiltonian)
    if os_gap_value is not None and isinstance(semigroup_gap_proxy, (int, float)):
        gap_proxy_matches_os = abs(float(semigroup_gap_proxy) - float(os_gap_value)) <= max(1e-18, 1e-6 * max(abs(float(semigroup_gap_proxy)), abs(float(os_gap_value))))

    present_links = []
    missing_theorem_clauses = []

    if action_matches:
        present_links.append("shared action-spec pinning between Schwinger and OS evidence")
    else:
        missing_theorem_clauses.append("Need a verified statement that the semigroup/operator evidence and OS reconstruction are pinned to the same Yang-Mills action/specification.")

    if os_invoked and os_output.get("hamiltonian"):
        present_links.append("OS reconstruction artifact names a continuum Hamiltonian output")
    else:
        missing_theorem_clauses.append("Need an explicit reconstruction object naming the continuum Hamiltonian acted on by the final bridge.")

    if operator_semigroup_shaped:
        present_links.append("operator convergence evidence is semigroup-shaped")
    else:
        missing_theorem_clauses.append("Need a semigroup or resolvent identification strong enough to compare with the OS Hamiltonian semigroup.")

    if wilson_algebra_named and semigroup_named_in_os:
        present_links.append("OS reconstruction names the Wilson-loop/GNS algebra and a semigroup-generated Hamiltonian")
    else:
        missing_theorem_clauses.append("Need an explicit algebra-level identification linking the Wilson-loop/GNS reconstruction to the semigroup/operator limit object.")

    if wilson_algebra_named:
        present_links.append("Wilson-loop observable algebra is explicitly named on the OS side")
    else:
        missing_theorem_clauses.append("Need an explicit identification of the Wilson-loop observable algebra used in the OS reconstruction.")

    if gns_representation_indicated:
        present_links.append("GNS representation language is explicitly present in the OS Hilbert-space construction")
    else:
        missing_theorem_clauses.append("Need an explicit representation-level identification linking the convergence-side observables to the OS GNS representation.")

    if separating_family_indicated:
        present_links.append("convergence-side uniqueness/separating-family evidence is recorded")
    else:
        missing_theorem_clauses.append("Need a theorem-level statement that the convergence-side observable family is separating/dense enough to identify the OS observable algebra.")

    if t0_matches:
        present_links.append("operator and semigroup evidence share the same comparison time t0")
    else:
        missing_theorem_clauses.append("Need a consistent normalization/time parameter linking semigroup convergence evidence and the gap-transfer argument.")

    if normalization_alignment:
        present_links.append("time-normalization is aligned between operator convergence and OS semigroup notation")
    else:
        missing_theorem_clauses.append("Need a theorem-level normalization identification showing the operator-limit time parameter matches the OS semigroup parameterization.")

    if semigroup_compatibility_indicated:
        present_links.append("semigroup compatibility is indicated between convergence evidence and OS Hamiltonian notation")
    else:
        missing_theorem_clauses.append("Need a theorem-level semigroup compatibility statement connecting the operator-limit semigroup to the OS time-translation semigroup.")

    if generator_equality_indicated:
        present_links.append("generator-level compatibility is indicated by explicit OS Hamiltonian generator language")
    else:
        missing_theorem_clauses.append("Need a theorem-level statement that the limiting generator extracted from convergence evidence is exactly the OS Hamiltonian H.")

    if domain_closure_indicated:
        present_links.append("domain/closure language is present on the OS reconstruction side")
    else:
        missing_theorem_clauses.append("Need an explicit domain or closure statement identifying the operator-limit domain with the OS Hamiltonian domain on the reconstructed Hilbert space.")

    if has_gap_proxy:
        present_links.append("positive approximant semigroup-gap proxy recorded")
    else:
        missing_theorem_clauses.append("Need a positive approximant gap input on the relevant vacuum-complement sector.")

    if vacuum_vector_indicated:
        present_links.append("OS-side vacuum vector is explicitly named with H Omega = 0")
    else:
        missing_theorem_clauses.append("Need an explicit identification of the OS vacuum vector used to define the ground-state sector for gap transfer.")

    if vacuum_projector_indicated:
        present_links.append("vacuum-complement projector data is partially indicated by vacuum naming plus positive gap proxy")
    else:
        missing_theorem_clauses.append("Need a theorem-level identification of the vacuum-complement projector used in the semigroup gap estimate.")

    if physical_sector_indicated:
        present_links.append("physical-sector restriction is indicated by the pinned vacuum-sector hypothesis")
    else:
        missing_theorem_clauses.append("Need a theorem-level restriction statement identifying the physical Hilbert-space sector on which the gap estimate acts.")

    if vacuum_sector_alignment:
        present_links.append("vacuum-sector compatibility is indicated by clustering, explicit vacuum naming, and pinned sector hypothesis")
    else:
        missing_theorem_clauses.append("Need the physical vacuum-complement sector used in gap transfer to be identified with the OS-reconstructed vacuum sector.")

    if gap_proxy_matches_os:
        present_links.append("OS Hamiltonian text records the same positive gap scale as the semigroup proxy evidence")
    else:
        missing_theorem_clauses.append("Need the positive gap appearing in semigroup evidence to be identified with the spectral gap of the OS Hamiltonian, not just quoted alongside it.")

    if isinstance(operator_method, str) and operator_method.strip():
        present_links.append("operator convergence artifact records an explicit convergence method")

    missing_theorem_clauses.append(
        "Need a Yang-Mills-specific theorem proving that the semigroup/operator limit evidenced by the repo coincides with the OS-reconstructed continuum Hamiltonian semigroup on the physical vacuum-complement sector."
    )

    return {
        "schema": "yangmills.hamiltonian_identification_evidence.v1",
        "action_spec": {
            "sha256": _maybe_sha256(os.path.join(os.path.dirname(__file__), "action_spec.json")),
        },
        "consistency_checks": {
            "action_spec_match": action_matches,
            "os_reconstruction_invoked": os_invoked,
            "operator_semigroup_shaped": operator_semigroup_shaped,
            "comparison_time_match": t0_matches,
            "positive_gap_proxy_present": has_gap_proxy,
            "algebra_compatibility_indicated": wilson_algebra_named and semigroup_named_in_os,
            "wilson_loop_observable_indicated": wilson_algebra_named,
            "gns_representation_indicated": gns_representation_indicated,
            "separating_family_indicated": separating_family_indicated,
            "vacuum_vector_indicated": vacuum_vector_indicated,
            "vacuum_projector_indicated": vacuum_projector_indicated,
            "physical_sector_indicated": physical_sector_indicated,
            "vacuum_sector_alignment_indicated": vacuum_sector_alignment,
            "time_normalization_alignment_indicated": normalization_alignment,
            "semigroup_compatibility_indicated": semigroup_compatibility_indicated,
            "generator_equality_indicated": generator_equality_indicated,
            "domain_closure_indicated": domain_closure_indicated,
            "os_gap_matches_proxy_value": gap_proxy_matches_os,
        },
        "diagnostics": {
            "operator_kind": operator_kind,
            "operator_t0": operator_t0,
            "semigroup_t0": semigroup_t0,
            "operator_method": operator_method,
            "vacuum_sector_hypothesis_status": vacuum_sector_hypothesis.get("status"),
            "vacuum_sector_hypothesis_title": vacuum_sector_title,
            "vacuum_sector_hypothesis_detail": vacuum_sector_hypothesis.get("detail"),
            "schwinger_uniqueness": ((schwinger.get("bounds") or {}) if isinstance(schwinger.get("bounds"), dict) else {}).get("uniqueness"),
            "schwinger_clustering": schwinger_clustering,
            "os_vacuum_text": os_vacuum,
            "os_hilbert_space_text": os_hilbert,
            "os_hamiltonian_text": os_hamiltonian,
            "os_gap_value": os_gap_value,
            "semigroup_gap_proxy": semigroup_gap_proxy,
            "semigroup_notes": semigroup_notes,
        },
        "present_links": present_links,
        "missing_theorem_clauses": missing_theorem_clauses,
        "provenance": {
            "source": "verification/ym_hamiltonian_identification_evidence.py",
            "inputs": [
                "verification/schwinger_limit_evidence.json",
                "verification/os_reconstruction_evidence.json",
                "verification/operator_convergence_evidence.json",
                "verification/semigroup_evidence.json",
                "verification/semigroup_hypotheses.json",
            ],
        },
    }


def verify_identification_evidence(doc: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(doc, dict):
        return False, "not_a_dict"
    if doc.get("schema") != "yangmills.hamiltonian_identification_evidence.v1":
        return False, "bad_schema"

    checks = doc.get("consistency_checks")
    if not isinstance(checks, dict):
        return False, "missing_consistency_checks"

    for key in [
        "action_spec_match",
        "os_reconstruction_invoked",
        "operator_semigroup_shaped",
        "comparison_time_match",
        "positive_gap_proxy_present",
        "algebra_compatibility_indicated",
    "wilson_loop_observable_indicated",
    "gns_representation_indicated",
    "separating_family_indicated",
    "vacuum_vector_indicated",
    "vacuum_projector_indicated",
    "physical_sector_indicated",
        "vacuum_sector_alignment_indicated",
        "time_normalization_alignment_indicated",
        "semigroup_compatibility_indicated",
        "generator_equality_indicated",
        "domain_closure_indicated",
        "os_gap_matches_proxy_value",
    ]:
        if not isinstance(checks.get(key), bool):
            return False, f"missing_or_nonbool:{key}"

    diagnostics = doc.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False, "missing_diagnostics"

    missing = doc.get("missing_theorem_clauses")
    if not isinstance(missing, list):
        return False, "missing_theorem_clause_list_missing"

    return True, "ok"


def audit_ym_hamiltonian_identification_evidence() -> Dict[str, Any]:
    doc = build_identification_evidence()
    ok, reason = verify_identification_evidence(doc)

    checks = doc.get("consistency_checks", {}) if isinstance(doc, dict) else {}
    all_links_present = all(bool(v) for v in checks.values()) if isinstance(checks, dict) else False

    status = "CONDITIONAL"
    detail = (
        "Consistency links for the Hamiltonian identification step are partially present, but the decisive Yang-Mills-specific identification theorem remains open."
    )
    if not ok:
        status = "FAIL"
        detail = reason
    elif all_links_present and not doc.get("missing_theorem_clauses"):
        status = "PASS"
        detail = "Hamiltonian identification evidence is complete."

    return {
        "key": "ym_hamiltonian_identification_evidence",
        "title": "Yang-Mills Hamiltonian identification evidence",
        "status": status,
        "detail": detail,
        "evidence": doc,
    }
