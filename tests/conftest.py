"""
Pytest configuration and markers for AgentLeak tests.

Organize tests by category, difficulty, and characteristics.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests (moderate speed)"
    )
    config.addinivalue_line(
        "markers", "features: feature tests (full scenarios)"
    )
    config.addinivalue_line(
        "markers", "production: production tests (full validation)"
    )
    
    # Difficulty markers
    config.addinivalue_line(
        "markers", "easy: easy tests (< 5s)"
    )
    config.addinivalue_line(
        "markers", "medium: medium difficulty tests (5-30s)"
    )
    config.addinivalue_line(
        "markers", "hard: hard tests (> 30s)"
    )
    
    # Characteristics
    config.addinivalue_line(
        "markers", "quick: quick execution (< 5s)"
    )
    config.addinivalue_line(
        "markers", "slow: slow execution (> 30s)"
    )
    config.addinivalue_line(
        "markers", "flaky: intermittent failures"
    )
    config.addinivalue_line(
        "markers", "requires_api: requires external API"
    )
    config.addinivalue_line(
        "markers", "requires_ml: requires ML dependencies"
    )


# Pytest hooks for better organization
def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test names and files."""
    for item in items:
        # Add markers based on filename
        if "unit" in item.nodeid or item.fspath.basename.startswith("test_"):
            if item.fspath.basename in ["test_schemas.py", "test_generators.py", 
                                        "test_metrics.py", "test_attack_families.py",
                                        "test_defenses.py"]:
                item.add_marker(pytest.mark.unit)
                item.add_marker(pytest.mark.quick)
            elif item.fspath.basename == "test_detection.py":
                item.add_marker(pytest.mark.unit)
                item.add_marker(pytest.mark.requires_ml)
        
        if item.fspath.basename in ["test_all_channels.py", "test_harness.py",
                                     "test_multiagent_evaluator.py", "test_sdk_integrations.py",
                                     "test_privacylens_integration.py"]:
            item.add_marker(pytest.mark.integration)
        
        if item.fspath.basename in ["test_all_frameworks.py", "test_crewai_5scenarios.py",
                                     "test_crewai_adapter_fix.py", "test_crewai_zero_modification.py",
                                     "test_strict_evaluator.py"]:
            item.add_marker(pytest.mark.features)
        
        if item.fspath.basename in ["test_agentleak_v2.py", "test_production.py"]:
            item.add_marker(pytest.mark.production)
