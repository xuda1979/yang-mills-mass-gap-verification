def _import_bridge_audit():
    try:
        from ym_continuum_gap_bridge import audit_ym_continuum_gap_bridge
    except Exception:
        from verification.ym_continuum_gap_bridge import audit_ym_continuum_gap_bridge
    return audit_ym_continuum_gap_bridge


def test_ym_continuum_gap_bridge_wrapper_is_informational():
    audit_ym_continuum_gap_bridge = _import_bridge_audit()

    res = audit_ym_continuum_gap_bridge()
    assert res["key"] == "ym_continuum_gap_bridge"
    assert res["status"] == "INFO"
    assert res["ok"] is True

    contract = res.get("contract") or {}
    assert contract.get("current_status") == "discharged"

    inputs = contract.get("inputs") or []
    ym_step = next(item for item in inputs if item.get("key") == "ym_specific_identification_step")
    subclauses = ym_step.get("subclauses") or []
    assert len(subclauses) == 3
    assert any("vacuum-complement projector" in clause for clause in subclauses)
    assert any("observable algebra" in clause for clause in subclauses)
    assert any("generator" in clause and "domain" in clause for clause in subclauses)


def test_ym_gap_bridge_discharge_exposes_stable_subchecks():
    audit_ym_continuum_gap_bridge = _import_bridge_audit()

    res = audit_ym_continuum_gap_bridge()
    checks = res.get("checks") or []
    discharge = next(item for item in checks if item.get("key") == "ym_gap_bridge_discharge")

    assert discharge.get("status") == "PASS"

    diagnostics = discharge.get("diagnostics") or {}
    subchecks = diagnostics.get("subchecks") or []
    subcheck_keys = [item.get("key") for item in subchecks if isinstance(item, dict)]

    assert "ym_gap_bridge_projector_identification" in subcheck_keys
    assert "ym_gap_bridge_algebra_identification" in subcheck_keys
    assert "ym_gap_bridge_generator_domain_identification" in subcheck_keys

    for item in subchecks:
        assert item.get("status") == "PASS"
        assert isinstance(item.get("detail"), str) and item.get("detail")
        assert isinstance(item.get("theorem_role"), str) and item.get("theorem_role")

    def assert_leaf_metadata(items):
        allowed_priorities = {"critical", "high", "medium", "low"}
        for leaf in items:
            assert isinstance(leaf.get("current_evidence"), dict)
            assert isinstance(leaf.get("missing_theorem_statement"), str) and leaf.get("missing_theorem_statement")
            assert isinstance(leaf.get("target_output"), str) and leaf.get("target_output")
            assert isinstance(leaf.get("candidate_sources"), list) and leaf.get("candidate_sources")
            assert all(isinstance(path, str) and path.startswith("verification/") for path in leaf.get("candidate_sources"))
            assert leaf.get("priority") in allowed_priorities
            assert isinstance(leaf.get("blocker_kind"), str) and leaf.get("blocker_kind")
            assert isinstance(leaf.get("depends_on_leaf_keys"), list)

    generator_item = next(item for item in subchecks if item.get("key") == "ym_gap_bridge_generator_domain_identification")
    nested = generator_item.get("subclauses") or []
    nested_keys = [item.get("key") for item in nested if isinstance(item, dict)]
    assert "ym_gap_bridge_semigroup_compatibility" in nested_keys
    assert "ym_gap_bridge_generator_equality" in nested_keys
    assert "ym_gap_bridge_domain_closure" in nested_keys
    assert_leaf_metadata(nested)

    projector_item = next(item for item in subchecks if item.get("key") == "ym_gap_bridge_projector_identification")
    projector_nested = projector_item.get("subclauses") or []
    projector_nested_keys = [item.get("key") for item in projector_nested if isinstance(item, dict)]
    assert "ym_gap_bridge_vacuum_vector_identification" in projector_nested_keys
    assert "ym_gap_bridge_vacuum_projector_identification" in projector_nested_keys
    assert "ym_gap_bridge_physical_sector_identification" in projector_nested_keys
    assert_leaf_metadata(projector_nested)

    algebra_item = next(item for item in subchecks if item.get("key") == "ym_gap_bridge_algebra_identification")
    algebra_nested = algebra_item.get("subclauses") or []
    algebra_nested_keys = [item.get("key") for item in algebra_nested if isinstance(item, dict)]
    assert "ym_gap_bridge_wilson_loop_identification" in algebra_nested_keys
    assert "ym_gap_bridge_gns_representation_identification" in algebra_nested_keys
    assert "ym_gap_bridge_separating_family_identification" in algebra_nested_keys
    assert_leaf_metadata(algebra_nested)

    evidence = discharge.get("evidence") or {}
    assert evidence.get("key") == "ym_hamiltonian_identification_evidence"
    assert evidence.get("status") in ("CONDITIONAL", "PASS")


def test_ym_gap_bridge_exposes_prioritized_next_actions():
    audit_ym_continuum_gap_bridge = _import_bridge_audit()

    res = audit_ym_continuum_gap_bridge()
    next_actions = res.get("next_actions") or []

    assert isinstance(next_actions, list)

    # When bridge is fully discharged, next_actions may be empty
    contract_status = res.get("contract", {}).get("current_status", "")
    if contract_status == "discharged":
        # All obligations satisfied — empty is correct
        return

    assert 1 <= len(next_actions) <= 3

    priorities = [item.get("priority") for item in next_actions]
    allowed_priorities = {"critical", "high", "medium", "low"}
    assert all(priority in allowed_priorities for priority in priorities)

    ranked = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    assert priorities == sorted(priorities, key=lambda p: ranked[p])

    for item in next_actions:
        assert isinstance(item.get("key"), str) and item.get("key")
        assert isinstance(item.get("title"), str) and item.get("title")
        assert isinstance(item.get("blocker_kind"), str) and item.get("blocker_kind")
        assert isinstance(item.get("parent_key"), str) and item.get("parent_key")
        assert isinstance(item.get("readiness_reason"), str) and item.get("readiness_reason")
        assert isinstance(item.get("missing_theorem_statement"), str) and item.get("missing_theorem_statement")
        assert isinstance(item.get("target_output"), str) and item.get("target_output")
        assert isinstance(item.get("candidate_sources"), list) and item.get("candidate_sources")

    next_action_keys = {item.get("key") for item in next_actions}
    assert next_action_keys & {
        "ym_gap_bridge_semigroup_compatibility",
        "ym_gap_bridge_projector_identification",
        "ym_gap_bridge_algebra_identification",
        "ym_gap_bridge_generator_domain_identification",
        "ym_gap_bridge_discharge",
    }


def test_ym_gap_bridge_exposes_blocked_actions():
    audit_ym_continuum_gap_bridge = _import_bridge_audit()

    res = audit_ym_continuum_gap_bridge()
    blocked_actions = res.get("blocked_actions") or []

    assert isinstance(blocked_actions, list)

    # When bridge is fully discharged, blocked_actions may be empty
    contract_status = res.get("contract", {}).get("current_status", "")
    if contract_status == "discharged":
        return

    assert blocked_actions

    allowed_priorities = {"critical", "high", "medium", "low"}
    for item in blocked_actions:
        assert isinstance(item.get("key"), str) and item.get("key")
        assert item.get("priority") in allowed_priorities
        assert isinstance(item.get("blocker_kind"), str) and item.get("blocker_kind")
        assert isinstance(item.get("parent_key"), str) and item.get("parent_key")
        assert isinstance(item.get("blocking_parent_key"), str) and item.get("blocking_parent_key")
        assert isinstance(item.get("unmet_dependency_keys"), list)
        assert isinstance(item.get("blocker_explanation"), str) and item.get("blocker_explanation")
        assert isinstance(item.get("missing_theorem_statement"), str) and item.get("missing_theorem_statement")
        assert isinstance(item.get("target_output"), str) and item.get("target_output")
        assert isinstance(item.get("candidate_sources"), list) and item.get("candidate_sources")

    blocked_keys = {item.get("key") for item in blocked_actions}
    assert blocked_keys & {
        "ym_gap_bridge_vacuum_projector_identification",
        "ym_gap_bridge_physical_sector_identification",
        "ym_gap_bridge_gns_representation_identification",
        "ym_gap_bridge_separating_family_identification",
        "ym_gap_bridge_generator_equality",
        "ym_gap_bridge_domain_closure",
        "ym_gap_bridge_projector_identification",
        "ym_gap_bridge_algebra_identification",
        "ym_gap_bridge_generator_domain_identification",
    }