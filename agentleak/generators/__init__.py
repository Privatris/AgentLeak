"""AgentLeak Generators - Scenario and data generation."""

from .canary_generator import (
    CanaryGenerator,
    generate_obvious_canary,
    generate_realistic_canary,
    generate_semantic_canary,
)
from .vault_generator import (
    VaultGenerator,
    generate_vault,
)
from .scenario_generator import (
    ScenarioGenerator,
    generate_scenario,
    generate_agentleak_lite,
    generate_agentleak_full,
)
from .task_config import (
    TaskConfig,
    TaskType,
    SensitiveCategory,
    TaskConfigCollection,
    CATEGORY_EXAMPLES,
)
from .seed_generator import (
    SeedDataGenerator,
    DataSeed,
    PlotType,
    create_sample_seeds,
)

# Contextual Integrity framework for privacy norms
from .contextual_integrity import (
    PrivacySeed,
    Vignette,
    AgentTrajectory,
    ContextualizedDataPoint,
    NormSource,
    TransmissionPrinciple,
    RelationshipType,
    REGULATION_SEEDS,
    CROWDSOURCED_SEEDS,
    get_all_seeds,
    get_seeds_by_vertical,
    get_seeds_by_source,
)

# Vignette generation with iterative refinement
from .vignette_generator import (
    VignetteGenerator,
    SurgeryKitModule,
    SurgeryKitUnitTest,
    NO_EXPLICIT_WORDS_TEST,
    HAS_ALL_FIELDS_TEST,
    VIGNETTE_LENGTH_TEST,
    create_contextualized_data_point,
    load_data_points_from_json,
    save_data_points_to_json,
)

# Hard scenarios for addressing reviewer concerns
from .hard_scenarios import (
    HardScenarioGenerator,
    HardScenarioTemplate,
    generate_hard_scenarios_jsonl,
    generate_complex_patient_record,
    generate_complex_financial_record,
)

# Cross-agent attack scenarios
try:
    from .cross_agent_scenarios import (
        CrossAgentAttackTemplate,
        COLLUSION_ATTACKS,
        DELEGATION_ATTACKS,
        ROLE_BOUNDARY_ATTACKS,
        IPI_CROSSAGENT_ATTACKS,
        generate_cross_agent_scenario,
        generate_all_cross_agent_scenarios,
        generate_cross_agent_scenarios_jsonl,
    )
except ImportError:
    pass  # Optional, depends on multiagent_evaluator

# Complex multi-step tasks (inspired by WebArena, GAIA, AgentBench)
from .complex_tasks import (
    ComplexTaskGenerator,
    ComplexTask,
    TaskStep,
    TaskComplexity,
    DependencyType,
    ALL_COMPLEX_TASKS,
    HEALTHCARE_COMPLEX_TASKS,
    FINANCE_COMPLEX_TASKS,
    LEGAL_COMPLEX_TASKS,
    CORPORATE_COMPLEX_TASKS,
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
