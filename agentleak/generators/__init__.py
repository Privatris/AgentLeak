"""
AgentLeak Generators - Scenario and data generation.

This module provides generators for creating privacy evaluation scenarios,
loading real-world PII data from HuggingFace, and generating attack templates.

Main components:
- RealDataLoader: Load pre-anonymized PII from HuggingFace datasets
- ScenarioGenerator: Generate benchmark scenarios  
- HardScenarioGenerator: Complex scenarios for adversarial testing

Deprecated (archived in _archive/):
- ComplexTaskGenerator, SeedDataGenerator, VaultGenerator, CanaryGenerator
"""

import warnings

# Real data loader - primary method for loading PII
from .real_data_loader import (
    RealDataLoader,
    RealDataScenarioGenerator,
)

# Contextual Integrity framework for privacy norms
from .contextual_integrity import (
    CROWDSOURCED_SEEDS,
    REGULATION_SEEDS,
    AgentTrajectory,
    ContextualizedDataPoint,
    NormSource,
    PrivacySeed,
    RelationshipType,
    TransmissionPrinciple,
    Vignette,
    get_all_seeds,
    get_seeds_by_source,
    get_seeds_by_vertical,
)

# Hard scenarios for adversarial testing
from .hard_scenarios import (
    HardScenarioGenerator,
    HardScenarioTemplate,
    generate_complex_financial_record,
    generate_complex_patient_record,
    generate_hard_scenarios_jsonl,
)

# Main scenario generation
from .scenario_generator import (
    ScenarioGenerator,
    generate_agentleak_full,
    generate_agentleak_lite,
    generate_scenario,
)

# Vignette generation with iterative refinement
from .vignette_generator import (
    HAS_ALL_FIELDS_TEST,
    NO_EXPLICIT_WORDS_TEST,
    VIGNETTE_LENGTH_TEST,
    SurgeryKitModule,
    SurgeryKitUnitTest,
    VignetteGenerator,
    create_contextualized_data_point,
    load_data_points_from_json,
    save_data_points_to_json,
)


# Lazy loading for deprecated modules
def __getattr__(name):
    """Lazy-load deprecated generators with deprecation warnings."""
    _deprecated = {
        'CanaryGenerator', 'VaultGenerator', 'SeedDataGenerator', 
        'ComplexTaskGenerator', 'DataSeed', 'TaskConfig', 'TaskType',
        'CrossAgentAttackTemplate', 'ComplexTask', 'TaskStep',
    }
    
    if name in _deprecated:
        warnings.warn(
            f"'{name}' is deprecated and has been moved to _archive/. "
            "Consider using RealDataLoader or ScenarioGenerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise AttributeError(f"'{name}' is no longer available. "
                           f"Use RealDataLoader from agentleak.generators instead.")
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Real data (recommended)
    "RealDataLoader",
    "RealDataScenarioGenerator",
    # Scenario generation
    "ScenarioGenerator",
    "generate_scenario",
    "generate_agentleak_lite",
    "generate_agentleak_full",
    # Contextual Integrity
    "PrivacySeed",
    "Vignette",
    "AgentTrajectory",
    "ContextualizedDataPoint",
    "NormSource",
    "TransmissionPrinciple",
    "RelationshipType",
    "REGULATION_SEEDS",
    "CROWDSOURCED_SEEDS",
    "get_all_seeds",
    "get_seeds_by_vertical",
    "get_seeds_by_source",
    # Vignette Generator
    "VignetteGenerator",
    "SurgeryKitModule",
    "SurgeryKitUnitTest",
    "NO_EXPLICIT_WORDS_TEST",
    "HAS_ALL_FIELDS_TEST",
    "VIGNETTE_LENGTH_TEST",
    "create_contextualized_data_point",
    "load_data_points_from_json",
    "save_data_points_to_json",
    # Hard scenarios
    "HardScenarioGenerator",
    "HardScenarioTemplate",
    "generate_hard_scenarios_jsonl",
    "generate_complex_patient_record",
    "generate_complex_financial_record",
]
