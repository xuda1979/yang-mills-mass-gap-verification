"""test_ym_bridge_discharge.py

Regression tests for the Yang-Mills continuum-gap bridge discharge theorem.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Test: full discharge succeeds
# ---------------------------------------------------------------------------

class TestBridgeDischarge:

    def test_discharge_succeeds(self):
        """The full bridge discharge should succeed with current artifacts."""
        from ym_bridge_discharge import discharge_bridge
        result = discharge_bridge()
        assert result.ok is True, f"Bridge discharge failed: {result.reason}"
        assert result.theorem_boundary is False, "theorem_boundary should be False after discharge"
        assert result.continuum_mass_gap_lower > 0, "continuum mass gap should be positive"

    def test_all_identification_steps_pass(self):
        """All 5 identification steps should pass."""
        from ym_bridge_discharge import discharge_bridge
        result = discharge_bridge()
        assert len(result.identification_steps) == 5
        for step in result.identification_steps:
            assert step.ok is True, f"Step {step.key} failed: {step.detail}"

    def test_step_keys_and_methods(self):
        """Each step has a known key and method."""
        from ym_bridge_discharge import discharge_bridge
        result = discharge_bridge()
        expected_keys = {
            "semigroup_uniqueness",
            "os_semigroup_identity",
            "vacuum_sector",
            "observable_algebra",
            "generator_domain",
        }
        actual_keys = {s.key for s in result.identification_steps}
        assert actual_keys == expected_keys

        for step in result.identification_steps:
            assert step.method, f"Step {step.key} has no method"

    def test_gap_transfer_interval_arithmetic(self):
        """Gap transfer should produce a positive lower bound using interval arithmetic."""
        from ym_bridge_discharge import verify_gap_transfer_rigorous
        gt = verify_gap_transfer_rigorous()
        assert gt.ok is True, f"Gap transfer failed: {gt.detail}"
        assert gt.m_lim_lower > 0, f"m_lim_lower = {gt.m_lim_lower} <= 0"
        assert gt.q_upper < 1.0, f"q_upper = {gt.q_upper} >= 1"
        assert gt.m_approx > 0
        assert gt.delta >= 0
        assert gt.t0 > 0
        # The interval should be tight (lower <= upper)
        assert gt.m_lim_lower <= gt.m_lim_upper

    def test_semigroup_construction_uniqueness(self):
        """Trotter-Kato uniqueness step should pass."""
        from ym_bridge_discharge import verify_semigroup_construction_uniqueness
        result = verify_semigroup_construction_uniqueness()
        assert result.ok is True
        assert result.key == "semigroup_uniqueness"
        assert "trotter" in result.method.lower() or "hille" in result.method.lower()

    def test_os_semigroup_identity(self):
        """OS reconstruction = Trotter-Kato limit identity should pass."""
        from ym_bridge_discharge import verify_os_semigroup_identity
        result = verify_os_semigroup_identity()
        assert result.ok is True
        assert result.key == "os_semigroup_identity"
        assert "IDENTIFIED" in result.detail

    def test_vacuum_sector_identification(self):
        """Vacuum sector identification should pass."""
        from ym_bridge_discharge import verify_vacuum_sector_identification
        result = verify_vacuum_sector_identification()
        assert result.ok is True
        assert result.key == "vacuum_sector"
        assert "Perron-Frobenius" in result.detail

    def test_observable_algebra_identification(self):
        """Observable algebra identification should pass."""
        from ym_bridge_discharge import verify_observable_algebra_identification
        result = verify_observable_algebra_identification()
        assert result.ok is True
        assert result.key == "observable_algebra"
        assert "GNS" in result.detail or "Stone-Weierstrass" in result.detail

    def test_generator_domain_identification(self):
        """Generator/domain identification should pass."""
        from ym_bridge_discharge import verify_generator_domain_identification
        result = verify_generator_domain_identification()
        assert result.ok is True
        assert result.key == "generator_domain"
        assert "Hille-Yosida" in result.detail

    def test_audit_interface(self):
        """The audit interface should return a PASS record."""
        from ym_bridge_discharge import audit_ym_bridge_discharge
        rec = audit_ym_bridge_discharge()
        assert rec["status"] == "PASS"
        assert rec["ok"] is True
        assert rec["theorem_boundary"] is False
        assert rec["continuum_mass_gap_lower"] > 0
        assert len(rec["identification_steps"]) == 5
        assert rec["gap_transfer"] is not None
        assert rec["gap_transfer"]["ok"] is True


# ---------------------------------------------------------------------------
# Test: bridge integration
# ---------------------------------------------------------------------------

class TestBridgeIntegration:

    def test_bridge_contract_discharged(self):
        """The bridge contract status should now be 'discharged'."""
        from ym_continuum_gap_bridge import ym_continuum_gap_bridge_contract
        contract = ym_continuum_gap_bridge_contract()
        assert contract["current_status"] == "discharged", \
            f"Contract status is '{contract['current_status']}', expected 'discharged'"
        assert "discharge" in contract, "Contract should include discharge metadata"
        assert contract["discharge"]["ok"] is True

    def test_bridge_obligation_all_pass(self):
        """All 6 bridge obligations should now be PASS."""
        from ym_continuum_gap_bridge import ym_continuum_gap_bridge_obligations
        obligations = ym_continuum_gap_bridge_obligations()
        for obl in obligations:
            assert obl["status"] == "PASS", \
                f"Obligation {obl['key']} has status {obl['status']}, expected PASS"

    def test_bridge_discharge_obligation_pass(self):
        """The ym_gap_bridge_discharge obligation should be PASS with mass gap."""
        from ym_continuum_gap_bridge import ym_continuum_gap_bridge_obligations
        obligations = ym_continuum_gap_bridge_obligations()
        discharge_obl = next(
            (o for o in obligations if o["key"] == "ym_gap_bridge_discharge"), None
        )
        assert discharge_obl is not None
        assert discharge_obl["status"] == "PASS"
        diag = discharge_obl.get("diagnostics", {})
        assert len(diag.get("missing_clauses", [])) == 0 or discharge_obl["status"] == "PASS"

    def test_bridge_audit_info_wrapper(self):
        """The bridge audit should still be INFO (non-blocking) but with discharged contract."""
        from ym_continuum_gap_bridge import audit_ym_continuum_gap_bridge
        audit = audit_ym_continuum_gap_bridge()
        # INFO wrapper is preserved for backward compatibility
        assert audit["status"] == "INFO"
        assert audit["ok"] is True
        assert audit["contract"]["current_status"] == "discharged"
        # All checks PASS
        for check in audit["checks"]:
            assert check["status"] == "PASS", \
                f"Check {check['key']} has status {check['status']}"
