"""APB Generators - Scenario and data generation."""

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
    generate_apb_full,
)
from agentleak.generators.task_config import (
    TaskConfig,
    TaskType,
    SensitiveCategory,
    TaskConfigCollection,
    CATEGORY_EXAMPLES,
)
from agentleak.generators.seed_generator import (
    SeedDataGenerator,
    DataSeed,
    PlotType,
    create_sample_seeds,
)
# PrivacyLens-inspired: Contextual Integrity framework
from agentleak.generators.contextual_integrity import (
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
# PrivacyLens-inspired: Vignette generation with SurgeryKit
from agentleak.generators.vignette_generator import (
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
    "generate_apb_lite",
    "generate_apb_full",
    # TaskConfig (AgentDAM-style)
    "TaskConfig",
    "TaskType",
    "SensitiveCategory",
    "TaskConfigCollection",
    "CATEGORY_EXAMPLES",
    # SeedGenerator (AgentDAM-style)
    "SeedDataGenerator",
    "DataSeed",
    "PlotType",
    "create_sample_seeds",
    # PrivacyLens: Contextual Integrity
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
    # PrivacyLens: Vignette Generator
    "VignetteGenerator",
    "SurgeryKitModule",
    "SurgeryKitUnitTest",
    "NO_EXPLICIT_WORDS_TEST",
    "HAS_ALL_FIELDS_TEST",
    "VIGNETTE_LENGTH_TEST",
    "create_contextualized_data_point",
    "load_data_points_from_json",
    "save_data_points_to_json",
]
