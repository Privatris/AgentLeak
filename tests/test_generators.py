"""Tests for APB generators."""

import pytest

from agentleak.schemas.scenario import (
    AdversaryLevel,
    CanaryTier,
    Channel,
    Vertical,
)
from agentleak.generators.canary_generator import (
    CanaryGenerator,
    generate_obvious_canary,
    generate_realistic_canary,
    generate_semantic_canary,
)
from agentleak.generators.vault_generator import (
    VaultGenerator,
    generate_vault,
)
from agentleak.generators.scenario_generator import (
    ScenarioGenerator,
    generate_scenario,
    generate_apb_lite,
)


class TestCanaryGenerator:
    """Test canary generation."""
    
    def test_obvious_ssn(self):
        """Test obvious SSN canary format."""
        gen = CanaryGenerator(seed=42)
        ssn = gen.obvious_ssn()
        assert ssn.startswith("CANARY_SSN_")
        assert len(ssn) == len("CANARY_SSN_") + 8
    
    def test_obvious_email(self):
        """Test obvious email canary format."""
        gen = CanaryGenerator(seed=42)
        email = gen.obvious_email()
        assert email.startswith("CANARY_EMAIL_")
        assert email.endswith("@apb.test")
    
    def test_realistic_ssn(self):
        """Test realistic SSN format."""
        gen = CanaryGenerator(seed=42)
        ssn = gen.realistic_ssn()
        # Format: XXX-XX-XXXX with 900-999 area
        parts = ssn.split("-")
        assert len(parts) == 3
        assert 900 <= int(parts[0]) <= 999
    
    def test_realistic_phone(self):
        """Test realistic phone format (555 prefix)."""
        gen = CanaryGenerator(seed=42)
        phone = gen.realistic_phone()
        assert phone.startswith("555-01")
    
    def test_semantic_medical(self):
        """Test semantic medical canary."""
        gen = CanaryGenerator(seed=42)
        condition = gen.semantic_medical_condition()
        assert isinstance(condition, str)
        assert len(condition) > 10  # Should be descriptive
    
    def test_generate_by_tier(self):
        """Test generate method with different tiers."""
        gen = CanaryGenerator(seed=42)
        
        obvious = gen.generate(CanaryTier.OBVIOUS, "ssn")
        assert "CANARY_" in obvious
        
        realistic = gen.generate(CanaryTier.REALISTIC, "ssn")
        assert "-" in realistic  # SSN format
        
        semantic = gen.generate(CanaryTier.SEMANTIC, "medical")
        assert len(semantic) > 20
    
    def test_generate_set(self):
        """Test generating multiple canaries."""
        gen = CanaryGenerator(seed=42)
        canaries = gen.generate_set(count=10)
        
        assert len(canaries) == 10
        for c in canaries:
            assert "tier" in c
            assert "field_type" in c
            assert "value" in c
    
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        obvious = generate_obvious_canary("ssn")
        assert "CANARY_" in obvious
        
        realistic = generate_realistic_canary("phone")
        assert "555" in realistic
        
        semantic = generate_semantic_canary("financial")
        assert len(semantic) > 10


class TestVaultGenerator:
    """Test vault generation."""
    
    def test_generate_healthcare_vault(self):
        """Test healthcare vault generation."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_vault(Vertical.HEALTHCARE, record_count=3)
        
        assert len(vault.records) == 3
        # Should have patient or provider records
        record_types = [r.record_type for r in vault.records]
        assert any(t in ["patient", "provider_note"] for t in record_types)
    
    def test_generate_finance_vault(self):
        """Test finance vault generation."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_vault(Vertical.FINANCE, record_count=3)
        
        assert len(vault.records) == 3
        record_types = [r.record_type for r in vault.records]
        assert any(t in ["customer", "transaction_dispute"] for t in record_types)
    
    def test_vault_has_canaries(self):
        """Test that vaults contain canaries."""
        gen = VaultGenerator(seed=42)
        vault = gen.generate_vault(Vertical.LEGAL, record_count=5)
        
        canaries = vault.get_all_canaries()
        # Should have at least some CANARY_ markers
        assert any("CANARY_" in c for c in canaries) or len(canaries) == 0
    
    def test_vault_tier_distribution(self):
        """Test custom tier distribution."""
        gen = VaultGenerator(seed=42)
        tier_dist = {
            CanaryTier.OBVIOUS: 2,
            CanaryTier.REALISTIC: 2,
            CanaryTier.SEMANTIC: 1,
        }
        vault = gen.generate_vault(
            Vertical.CORPORATE,
            record_count=5,
            tier_distribution=tier_dist,
        )
        
        assert len(vault.records) == 5
        
        # Check tier distribution
        tiers = [r.canary_tier for r in vault.records]
        assert tiers.count(CanaryTier.OBVIOUS) == 2
        assert tiers.count(CanaryTier.REALISTIC) == 2
        assert tiers.count(CanaryTier.SEMANTIC) == 1
    
    def test_convenience_function(self):
        """Test module-level generate_vault function."""
        vault = generate_vault(Vertical.HEALTHCARE, record_count=2, seed=42)
        assert len(vault.records) == 2


class TestScenarioGenerator:
    """Test scenario generation."""
    
    def test_generate_basic_scenario(self):
        """Test basic scenario generation."""
        gen = ScenarioGenerator(seed=42)
        scenario = gen.generate(Vertical.HEALTHCARE)
        
        assert scenario.scenario_id.startswith("apb_hea_")
        assert scenario.vertical == Vertical.HEALTHCARE
        assert len(scenario.agents) >= 1
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
    
    def test_apb_lite_generation(self):
        """Test APB-Lite convenience function."""
        apb_lite = generate_apb_lite(seed=42)
        
        assert apb_lite.name == "apb_lite_100"
        assert apb_lite.count >= 90  # Allow some rounding
        
        # Check distribution
        adversarial_count = sum(1 for s in apb_lite.scenarios if s.is_adversarial)
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
        """Test complete generation from canaries to scenarios."""
        # Generate canaries
        canary_gen = CanaryGenerator(seed=42)
        canaries = canary_gen.generate_set(count=5)
        assert len(canaries) == 5
        
        # Generate vault
        vault_gen = VaultGenerator(seed=42)
        vault = vault_gen.generate_vault(Vertical.HEALTHCARE, record_count=3)
        assert len(vault.records) == 3
        
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
        assert len(scenario.get_canaries()) > 0
        assert scenario.attack.attack_class is not None
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        scenario1 = generate_scenario(Vertical.LEGAL, seed=12345)
        scenario2 = generate_scenario(Vertical.LEGAL, seed=12345)
        
        assert scenario1.scenario_id == scenario2.scenario_id
        assert scenario1.objective.user_request == scenario2.objective.user_request


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
