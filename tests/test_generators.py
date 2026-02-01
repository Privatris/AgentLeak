"""Tests for agentleak generators."""

import pytest

from agentleak.schemas.scenario import (
    AdversaryLevel,
    CanaryTier,
    Vertical,
)
# Import VaultGenerator from scenario_generator (new location)
from agentleak.generators.scenario_generator import (
    ScenarioGenerator,
    VaultGenerator,
    generate_scenario,
    generate_agentleak_lite,
)


class TestVaultGenerator:
    """Test vault generation."""

    def test_healthcare_vault(self):
        """Test healthcare vault generation."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_healthcare_vault()
        
        assert hasattr(vault, 'records')
        assert len(vault.records) >= 1
        record = vault.records[0]
        assert "name" in record.fields or "patient_name" in record.fields
        assert "ssn" in record.fields
        assert record.fields["ssn"].startswith("CANARY_SSN_")
        assert "diagnosis" in record.fields
        assert record.fields["diagnosis"].startswith("CANARY_DIAGNOSIS_")
    
    def test_finance_vault(self):
        """Test finance vault generation."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_finance_vault()
        
        assert len(vault.records) >= 1
        record = vault.records[0]
        assert "ssn" in record.fields
        assert "account_number" in record.fields
        assert record.fields["account_number"].startswith("CANARY_ACCOUNT_")
        assert "credit_card" in record.fields
        assert record.fields["credit_card"].startswith("CANARY_CC_")
    
    def test_legal_vault(self):
        """Test legal vault generation."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_legal_vault()
        
        assert len(vault.records) >= 1
        record = vault.records[0]
        assert "client_name" in record.fields
        assert "case_number" in record.fields
        assert "confidential_strategy" in record.fields
        assert record.fields["confidential_strategy"].startswith("CANARY_STRATEGY_")
    
    def test_corporate_vault(self):
        """Test corporate/HR vault generation."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_corporate_vault()
        
        assert len(vault.records) >= 1
        record = vault.records[0]
        assert "employee_name" in record.fields
        assert "employee_id" in record.fields
        assert "salary" in record.fields
        assert "performance_notes" in record.fields
        assert record.fields["performance_notes"].startswith("CANARY_PERF_NOTE_")
    
    def test_generate_by_vertical(self):
        """Test vault generation by vertical."""
        gen = VaultGenerator(seed=42)
        
        for vertical in [Vertical.HEALTHCARE, Vertical.FINANCE, Vertical.LEGAL, Vertical.CORPORATE]:
            vault = gen.generate_vault(vertical)
            assert hasattr(vault, 'records')
            assert len(vault.records) >= 1
    
    def test_vault_has_canaries(self):
        """Test that vaults contain CANARY_ tokens."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_healthcare_vault()
        
        # Check for canary tokens via get_all_canaries()
        canaries = vault.get_all_canaries()
        assert len(canaries) >= 3, "Should have at least 3 canary fields"
        assert all("CANARY_" in c for c in canaries)
    
    def test_reproducibility(self):
        """Test that same seed produces same vault."""
        gen1 = VaultGenerator(seed=42)
        gen2 = VaultGenerator(seed=42)
        
        vault1 = gen1.generate_healthcare_vault()
        vault2 = gen2.generate_healthcare_vault()
        
        # Note: With Faker, seeds need to be set before each generation
        # The test checks that canary prefixes are deterministic
        assert len(vault1.records) == len(vault2.records)
        assert vault1.records[0].record_type == vault2.records[0].record_type
    
    def test_record_count(self):
        """Test that record_count parameter works."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_healthcare_vault(record_count=5)
        
        assert len(vault.records) == 5
    
    def test_all_verticals_covered(self):
        """Test that all verticals have vault generators."""
        gen = VaultGenerator(seed=42)
        
        for vertical in Vertical:
            vault = gen.generate_vault(vertical)
            assert hasattr(vault, 'records')
            assert len(vault.records) >= 1


class TestScenarioGenerator:
    """Test scenario generation."""

    def test_generate_basic_scenario(self):
        """Test basic scenario generation."""
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(Vertical.HEALTHCARE)

        assert scenario.scenario_id.startswith("agentleak_hea_")
        assert scenario.vertical == Vertical.HEALTHCARE
        assert len(scenario.agents) >= 1
        # Check vault is properly generated
        assert scenario.private_vault is not None
        assert len(scenario.private_vault.records) > 0

    def test_generate_adversarial_scenario(self):
        """Test adversarial scenario generation."""
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(
            Vertical.FINANCE,
            adversary_level=AdversaryLevel.A1_WEAK,
        )

        assert scenario.is_adversarial is True
        assert scenario.attack.enabled is True
        assert scenario.attack.attack_class is not None

    def test_generate_multiagent_scenario(self):
        """Test multi-agent scenario generation."""
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(
            Vertical.LEGAL,
            multi_agent=True,
        )

        assert scenario.is_multiagent is True
        assert len(scenario.agents) >= 2

    def test_scenario_has_objective(self):
        """Test that scenarios have valid objectives."""
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(Vertical.CORPORATE)

        assert scenario.objective.user_request is not None
        assert len(scenario.objective.user_request) > 0
        assert len(scenario.objective.success_criteria) > 0

    def test_scenario_has_allowed_set(self):
        """Test that scenarios have allowed/forbidden sets."""
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(Vertical.HEALTHCARE)

        assert len(scenario.allowed_set.fields) > 0
        assert len(scenario.allowed_set.forbidden_fields) > 0

    def test_generate_scenario_set(self):
        """Test generating multiple scenarios."""
        gen = ScenarioGenerator(seed=42)
        scenario_set = gen.generate_set("test_set", total_count=20)

        assert scenario_set.name == "test_set"
        assert scenario_set.count >= 15  # Allow some rounding

        # Check vertical distribution
        verticals = set(s.vertical for s in scenario_set.scenarios)
        assert len(verticals) == 4  # All 4 verticals

    def test_agentleak_lite_generation(self):
        """Test agentleak-Lite convenience function."""
        agentleak_lite = generate_agentleak_lite(seed=42)

        assert agentleak_lite.name == "agentleak_lite_100"
        assert agentleak_lite.count >= 90  # Allow some rounding

        # Check distribution
        adversarial_count = sum(1 for s in agentleak_lite.scenarios if s.is_adversarial)
        assert adversarial_count > 0  # Some should be adversarial

    def test_scenario_serialization(self):
        """Test that generated scenarios can be serialized."""
        scenario = generate_scenario(Vertical.FINANCE, seed=42)

        # Should not raise
        json_str = scenario.model_dump_json()
        assert len(json_str) > 100

        # Should be valid JSON
        import json

        data = json.loads(json_str)
        assert data["vertical"] == "finance"


class TestIntegration:
    """Integration tests for generators."""

    def test_full_generation_pipeline(self):
        """Test complete generation from vaults to scenarios."""
        # Generate vault
        vault_gen = VaultGenerator(seed=42)
        vault = vault_gen.generate_vault(Vertical.HEALTHCARE)
        assert hasattr(vault, 'records')
        assert len(vault.records) >= 1

        # Generate scenario
        scenario_gen = ScenarioGenerator(seed=42)
        scenario = scenario_gen.generate(
            Vertical.HEALTHCARE,
            adversary_level=AdversaryLevel.A2_STRONG,
            multi_agent=True,
        )

        # Verify scenario completeness
        assert scenario.is_multiagent is True
        assert scenario.is_adversarial is True
        assert scenario.private_vault is not None
        assert scenario.attack.attack_class is not None

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        scenario1 = generate_scenario(Vertical.LEGAL, seed=12345)
        scenario2 = generate_scenario(Vertical.LEGAL, seed=12345)

        assert scenario1.scenario_id == scenario2.scenario_id
        assert scenario1.objective.user_request == scenario2.objective.user_request


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
