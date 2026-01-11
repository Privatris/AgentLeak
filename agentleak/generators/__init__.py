"""AgentLeak Generators - Scenario and data generation."""

from .canary_generator import (
    CanaryGenerator,
    generate_obvious_canary,
    generate_realistic_canary,
    generate_semantic_canary,
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

# Hard scenarios for addressing reviewer concerns
from .hard_scenarios import (
    HardScenarioGenerator,
    HardScenarioTemplate,
    generate_complex_financial_record,
    generate_complex_patient_record,
    generate_hard_scenarios_jsonl,
)
from .scenario_generator import (
    ScenarioGenerator,
    generate_agentleak_full,
    generate_agentleak_lite,
    generate_scenario,
)
from .seed_generator import (
    DataSeed,
    PlotType,
    SeedDataGenerator,
    create_sample_seeds,
)
from .task_config import (
    CATEGORY_EXAMPLES,
    SensitiveCategory,
    TaskConfig,
    TaskConfigCollection,
    TaskType,
)
from .vault_generator import (
    VaultGenerator,
    generate_vault,
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

# Cross-agent attack scenarios
try:
    from .cross_agent_scenarios import (
        COLLUSION_ATTACKS,
        DELEGATION_ATTACKS,
        IPI_CROSSAGENT_ATTACKS,
        ROLE_BOUNDARY_ATTACKS,
        CrossAgentAttackTemplate,
        generate_all_cross_agent_scenarios,
        generate_cross_agent_scenario,
        generate_cross_agent_scenarios_jsonl,
    )
except ImportError:
    pass  # Optional, depends on multiagent_evaluator

# Complex multi-step tasks (inspired by WebArena, GAIA, AgentBench)
from .complex_tasks import (
    ALL_COMPLEX_TASKS,
    CORPORATE_COMPLEX_TASKS,
    FINANCE_COMPLEX_TASKS,
    HEALTHCARE_COMPLEX_TASKS,
    LEGAL_COMPLEX_TASKS,
    ComplexTask,
    ComplexTaskGenerator,
    DependencyType,
    TaskComplexity,
    TaskStep,
    generate_complex_scenarios,
    get_expected_tsr_by_complexity,
)

__all__ = [
    # Canary
    "CanaryGenerator",
    "generate_obvious_canary",
    "generate_realistic_canary",
    "generate_semantic_canary",
    # Vault
    "VaultGenerator",
    "generate_vault",
    # Scenario
    "ScenarioGenerator",
    "generate_scenario",
    "generate_agentleak_lite",
    "generate_agentleak_full",
    # TaskConfig (AgentDAM-style)
    "TaskConfig",
    "TaskType",
    "SensitiveCategory",
    "TaskConfigCollection",
    "CATEGORY_EXAMPLES",
    # SeedGenerator
    "SeedDataGenerator",
    "DataSeed",
    "PlotType",
    "create_sample_seeds",
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
    # Hard scenarios (reviewer concerns)
    "HardScenarioGenerator",
    "HardScenarioTemplate",
    "generate_hard_scenarios_jsonl",
    "generate_complex_patient_record",
    "generate_complex_financial_record",
    # Cross-agent attack scenarios
    "CrossAgentAttackTemplate",
    "COLLUSION_ATTACKS",
    "DELEGATION_ATTACKS",
    "ROLE_BOUNDARY_ATTACKS",
    "IPI_CROSSAGENT_ATTACKS",
    "generate_cross_agent_scenario",
    "generate_all_cross_agent_scenarios",
    "generate_cross_agent_scenarios_jsonl",
    # Complex multi-step tasks (WebArena/GAIA/AgentBench inspired)
    "ComplexTaskGenerator",
    "ComplexTask",
    "TaskStep",
    "TaskComplexity",
    "DependencyType",
    "ALL_COMPLEX_TASKS",
    "HEALTHCARE_COMPLEX_TASKS",
    "FINANCE_COMPLEX_TASKS",
    "LEGAL_COMPLEX_TASKS",
    "CORPORATE_COMPLEX_TASKS",
    "generate_complex_scenarios",
    "get_expected_tsr_by_complexity",
]
