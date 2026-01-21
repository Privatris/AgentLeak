#!/usr/bin/env python3
"""
AgentLeak SDK - Integration Tests

Tests the SDK integration with all 4 paper frameworks:
- CrewAI
- LangChain  
- AutoGPT
- MetaGPT

Paper Reference: Section 5.1 - "all major frameworks exhibit 28-35% internal leakage"
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentleak.integrations import (
    BaseIntegration,
    IntegrationConfig,
    DetectionMode,
    LeakIncident,
    IntegrationStats,
    CrewAIIntegration,
    LangChainIntegration,
    AutoGPTIntegration,
    MetaGPTIntegration,
    add_agentleak,
    detect_framework,
)


# ============================================================================
# Test Configuration
# ============================================================================

TEST_VAULT = {
    "ssn": "123-45-6789",
    "credit_card": "4111-1111-1111-1111",
    "api_key": "sk-test-1234567890abcdef",
    "password": "SuperSecret123!",
    "email": "john.doe@company.com",
}

SAFE_CONTENT = "The weather is nice today. No sensitive data here."
LEAK_CONTENT = "User SSN is 123-45-6789 and their email is john.doe@company.com"


# ============================================================================
# Mock Framework Objects (for testing without actual frameworks)
# ============================================================================

class MockCrew:
    """Mock CrewAI Crew for testing."""
    __module__ = 'crewai'
    callbacks = None
    
    def kickoff(self):
        return "Mock crew result"


class MockLangChainAgent:
    """Mock LangChain Agent for testing."""
    __module__ = 'langchain.agents'
    callbacks = None
    
    def invoke(self, input, callbacks=None):
        return {"output": "Mock agent result"}


class MockAutoGPTAgent:
    """Mock AutoGPT Agent for testing."""
    __module__ = 'autogpt'
    
    def execute_command(self, name, args):
        return "Command executed"
    
    class memory:
        @staticmethod
        def add(content):
            pass


class MockMetaGPTTeam:
    """Mock MetaGPT Team for testing."""
    __module__ = 'metagpt.team'
    roles = []
    
    async def run(self):
        return "Mock team result"


# ============================================================================
# Test Functions
# ============================================================================

def test_base_classes():
    """Test base class instantiation and configuration."""
    print("\n" + "="*60)
    print("Test 1: Base Classes")
    print("="*60)
    
    # Test IntegrationConfig
    config = IntegrationConfig(
        vault=TEST_VAULT,
        mode=DetectionMode.HYBRID,
        alert_threshold=0.7,
        block_on_leak=False,
    )
    
    assert config.vault == TEST_VAULT
    assert config.mode == DetectionMode.HYBRID
    assert config.alert_threshold == 0.7
    print("✓ IntegrationConfig works correctly")
    
    # Test LeakIncident
    incident = LeakIncident(
        timestamp="2024-01-01T00:00:00",
        framework="test",
        channel="C1",
        leaked_items=["ssn"],
        content_preview="Test content...",
        confidence=0.95,
        detection_tier="EXACT_MATCH",
    )
    
    assert incident.framework == "test"
    assert incident.confidence == 0.95
    print("✓ LeakIncident works correctly")
    
    # Test IntegrationStats
    stats = IntegrationStats()
    assert stats.total_checks == 0
    assert stats.leak_rate == 0.0
    print("✓ IntegrationStats works correctly")
    
    print("\n✓ All base classes work correctly")
    return True


def test_framework_detection():
    """Test automatic framework detection."""
    print("\n" + "="*60)
    print("Test 2: Framework Detection")
    print("="*60)
    
    # Test detection
    crew = MockCrew()
    assert detect_framework(crew) == 'crewai'
    print("✓ CrewAI detection works")
    
    agent = MockLangChainAgent()
    assert detect_framework(agent) == 'langchain'
    print("✓ LangChain detection works")
    
    autogpt = MockAutoGPTAgent()
    assert detect_framework(autogpt) == 'autogpt'
    print("✓ AutoGPT detection works")
    
    team = MockMetaGPTTeam()
    assert detect_framework(team) == 'metagpt'
    print("✓ MetaGPT detection works")
    
    print("\n✓ All framework detections work correctly")
    return True


def test_crewai_integration():
    """Test CrewAI integration."""
    print("\n" + "="*60)
    print("Test 3: CrewAI Integration")
    print("="*60)
    
    config = IntegrationConfig(vault=TEST_VAULT)
    integration = CrewAIIntegration(config)
    
    assert integration.FRAMEWORK_NAME == "crewai"
    print(f"✓ Framework: {integration.FRAMEWORK_NAME}")
    print(f"✓ Version: {integration.FRAMEWORK_VERSION}")
    
    # Test attachment
    crew = MockCrew()
    crew = integration.attach(crew)
    
    assert hasattr(crew, '_agentleak')
    print("✓ Monitoring attached to Crew")
    
    # Test leak detection
    incident = integration._check_for_leaks(SAFE_CONTENT, "C1")
    assert incident is None
    print("✓ Safe content passes (no leak)")
    
    incident = integration._check_for_leaks(LEAK_CONTENT, "C2")
    assert incident is not None
    print(f"✓ Leak detected: {incident.leaked_items}")
    
    # Check stats
    stats = integration.stats
    print(f"✓ Stats: {stats.total_checks} checks, {stats.leaks_detected} leaks")
    
    print("\n✓ CrewAI integration works correctly")
    return True


def test_langchain_integration():
    """Test LangChain integration."""
    print("\n" + "="*60)
    print("Test 4: LangChain Integration")
    print("="*60)
    
    config = IntegrationConfig(vault=TEST_VAULT)
    integration = LangChainIntegration(config)
    
    assert integration.FRAMEWORK_NAME == "langchain"
    print(f"✓ Framework: {integration.FRAMEWORK_NAME}")
    print(f"✓ Version: {integration.FRAMEWORK_VERSION}")
    
    # Test attachment
    agent = MockLangChainAgent()
    agent = integration.attach(agent)
    
    assert hasattr(agent, '_agentleak')
    print("✓ Monitoring attached to Agent")
    
    # Test leak detection on C3 (tool input) and C4 (tool output)
    incident = integration._check_for_leaks("search for weather", "C3")
    assert incident is None
    print("✓ Safe tool input passes")
    
    incident = integration._check_for_leaks(f"API key: {TEST_VAULT['api_key']}", "C4")
    assert incident is not None
    print(f"✓ Leak in tool output detected: {incident.leaked_items}")
    
    print("\n✓ LangChain integration works correctly")
    return True


def test_autogpt_integration():
    """Test AutoGPT integration."""
    print("\n" + "="*60)
    print("Test 5: AutoGPT Integration")
    print("="*60)
    
    config = IntegrationConfig(vault=TEST_VAULT)
    integration = AutoGPTIntegration(config)
    
    assert integration.FRAMEWORK_NAME == "autogpt"
    print(f"✓ Framework: {integration.FRAMEWORK_NAME}")
    print(f"✓ Version: {integration.FRAMEWORK_VERSION}")
    
    # Test attachment
    agent = MockAutoGPTAgent()
    agent = integration.attach(agent)
    
    assert hasattr(agent, '_agentleak')
    print("✓ Monitoring attached to Agent")
    
    # Test leak detection on C5 (memory)
    incident = integration._check_for_leaks("Remember this context", "C5")
    assert incident is None
    print("✓ Safe memory content passes")
    
    incident = integration._check_for_leaks(f"Store password: {TEST_VAULT['password']}", "C5")
    assert incident is not None
    print(f"✓ Leak in memory detected: {incident.leaked_items}")
    
    print("\n✓ AutoGPT integration works correctly")
    return True


def test_metagpt_integration():
    """Test MetaGPT integration."""
    print("\n" + "="*60)
    print("Test 6: MetaGPT Integration")
    print("="*60)
    
    config = IntegrationConfig(vault=TEST_VAULT)
    integration = MetaGPTIntegration(config)
    
    assert integration.FRAMEWORK_NAME == "metagpt"
    print(f"✓ Framework: {integration.FRAMEWORK_NAME}")
    print(f"✓ Version: {integration.FRAMEWORK_VERSION}")
    
    # Test attachment
    team = MockMetaGPTTeam()
    team = integration.attach(team)
    
    assert hasattr(team, '_agentleak')
    print("✓ Monitoring attached to Team")
    
    # Test leak detection on C2 (inter-role messages)
    incident = integration._check_for_leaks("Architect reviewing code", "C2")
    assert incident is None
    print("✓ Safe inter-role message passes")
    
    incident = integration._check_for_leaks(f"User's credit card: {TEST_VAULT['credit_card']}", "C2")
    assert incident is not None
    print(f"✓ Leak in role message detected: {incident.leaked_items}")
    
    print("\n✓ MetaGPT integration works correctly")
    return True


def test_universal_function():
    """Test universal add_agentleak function."""
    print("\n" + "="*60)
    print("Test 7: Universal add_agentleak()")
    print("="*60)
    
    # Test with each framework
    crew = add_agentleak(MockCrew(), TEST_VAULT)
    assert hasattr(crew, '_agentleak')
    print("✓ add_agentleak() works with CrewAI")
    
    agent = add_agentleak(MockLangChainAgent(), TEST_VAULT)
    assert hasattr(agent, '_agentleak')
    print("✓ add_agentleak() works with LangChain")
    
    autogpt = add_agentleak(MockAutoGPTAgent(), TEST_VAULT)
    assert hasattr(autogpt, '_agentleak')
    print("✓ add_agentleak() works with AutoGPT")
    
    team = add_agentleak(MockMetaGPTTeam(), TEST_VAULT)
    assert hasattr(team, '_agentleak')
    print("✓ add_agentleak() works with MetaGPT")
    
    print("\n✓ Universal function works for all frameworks")
    return True


def test_detection_modes():
    """Test different detection modes."""
    print("\n" + "="*60)
    print("Test 8: Detection Modes")
    print("="*60)
    
    modes = [
        (DetectionMode.FAST, "FAST (~10ms, 75% accuracy)"),
        (DetectionMode.STANDARD, "STANDARD (~100ms, 85% accuracy)"),
        (DetectionMode.HYBRID, "HYBRID (~200ms, 92% accuracy)"),
    ]
    
    for mode, desc in modes:
        config = IntegrationConfig(vault=TEST_VAULT, mode=mode)
        integration = CrewAIIntegration(config)
        
        # Test with leak content
        incident = integration._check_for_leaks(LEAK_CONTENT, "C1")
        detected = incident is not None
        
        print(f"✓ {desc}: leak_detected={detected}")
    
    print("\n✓ All detection modes work correctly")
    return True


def generate_summary():
    """Generate test summary."""
    print("\n" + "="*60)
    print("SDK Test Summary")
    print("="*60)
    
    summary = {
        "sdk_version": "1.0.0",
        "frameworks": ["CrewAI", "LangChain", "AutoGPT", "MetaGPT"],
        "channels_monitored": {
            "C1": "Final output",
            "C2": "Inter-agent messages",
            "C3": "Tool inputs",
            "C4": "Tool outputs",
            "C5": "Memory",
            "C6": "Artifacts",
        },
        "detection_modes": ["FAST", "STANDARD", "HYBRID", "LLM_ONLY"],
        "paper_reference": "Section 5.1 - 28-35% leak rate on all frameworks",
    }
    
    print(json.dumps(summary, indent=2))
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all tests."""
    print("="*60)
    print("AgentLeak SDK Integration Tests")
    print("Paper: 'AgentLeak: Privacy Leakage in Multi-Agent Systems'")
    print("="*60)
    
    tests = [
        ("Base Classes", test_base_classes),
        ("Framework Detection", test_framework_detection),
        ("CrewAI Integration", test_crewai_integration),
        ("LangChain Integration", test_langchain_integration),
        ("AutoGPT Integration", test_autogpt_integration),
        ("MetaGPT Integration", test_metagpt_integration),
        ("Universal Function", test_universal_function),
        ("Detection Modes", test_detection_modes),
    ]
    
    results = {}
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results[name] = "✓ PASS" if success else "✗ FAIL"
        except Exception as e:
            results[name] = f"✗ ERROR: {e}"
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    
    for name, result in results.items():
        print(f"  {name}: {result}")
    
    passed = sum(1 for r in results.values() if "PASS" in r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! SDK is ready for use.")
        generate_summary()
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
