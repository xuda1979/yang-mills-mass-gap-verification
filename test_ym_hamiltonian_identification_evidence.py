def _import_module_symbols():
    try:
        from ym_hamiltonian_identification_evidence import (
            audit_ym_hamiltonian_identification_evidence,
            build_identification_evidence,
        )
    except Exception:
        from verification.ym_hamiltonian_identification_evidence import (
            audit_ym_hamiltonian_identification_evidence,
            build_identification_evidence,
        )
    return audit_ym_hamiltonian_identification_evidence, build_identification_evidence


def test_identification_evidence_exposes_richer_checks():
    audit_fn, build_fn = _import_module_symbols()

    doc = build_fn()
    checks = doc["consistency_checks"]
    diagnostics = doc["diagnostics"]

    assert checks["action_spec_match"] is True
    assert checks["operator_semigroup_shaped"] is True
    assert checks["comparison_time_match"] is True
    assert checks["positive_gap_proxy_present"] is True
    assert checks["algebra_compatibility_indicated"] is True
    assert checks["wilson_loop_observable_indicated"] is True
    assert checks["gns_representation_indicated"] is True
    assert checks["separating_family_indicated"] is True
    assert checks["vacuum_vector_indicated"] is True
    assert checks["vacuum_projector_indicated"] is True
    assert checks["physical_sector_indicated"] is True
    assert checks["vacuum_sector_alignment_indicated"] is True
    assert checks["time_normalization_alignment_indicated"] is True
    assert checks["semigroup_compatibility_indicated"] is True
    assert checks["generator_equality_indicated"] is True
    assert checks["domain_closure_indicated"] is True
    assert checks["os_gap_matches_proxy_value"] is False

    assert diagnostics["operator_t0"] == 1.0
    assert diagnostics["semigroup_t0"] == 1.0
    assert diagnostics["vacuum_sector_hypothesis_status"] == "PASS"
    assert "physical hilbert space" in (diagnostics["vacuum_sector_hypothesis_title"] or "").lower()
    assert diagnostics["schwinger_uniqueness"] is True
    assert diagnostics["schwinger_clustering"] is True
    assert "h omega = 0" in (diagnostics["os_hamiltonian_text"] or "").lower()
    assert "unique cyclic vector omega" in (diagnostics["os_vacuum_text"] or "").lower()
    assert "closure of" in (diagnostics["os_hilbert_space_text"] or "").lower()
    assert "wilson-loop algebra" in (diagnostics["os_hilbert_space_text"] or "").lower()
    assert "self-adjoint generator" in (diagnostics["os_hamiltonian_text"] or "").lower()

    audit = audit_fn()
    assert audit["status"] in {"CONDITIONAL", "PASS"}
    if audit["status"] == "PASS":
        assert "complete" in audit["detail"].lower()
    else:
        assert "identification theorem remains open" in audit["detail"].lower()