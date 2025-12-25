"""
AgentLeak - A Full-Stack Benchmark for Privacy Leakage
in Tool-Using and Multi-Agent LLM Systems.

This package provides:
- Scenario generation with privacy-sensitive data
- Attack implementations (15 classes in 4 families)
- Multi-agent execution harness
- Privacy leakage detection pipeline (3-stage: canary, pattern, semantic)
- Metrics computation (ELR, WLS, CLR, ASR, TSR)
- Defense implementations (LCF - Latent Compliance Firewall)
"""

__version__ = "0.1.0"
__author__ = "Faouzi EL YAGOUBI, Ranwa AL MALLAH"

from agentleak.schemas.scenario import (
    AdversaryLevel,
    AttackClass,
    AttackConfig,
    AttackFamily,
    Channel,
    Difficulty,
    Scenario,
    ScenarioSet,
    Vertical,
)
from agentleak.schemas.trace import (
    EventType,
    ExecutionTrace,
    TraceEvent,
)
from agentleak.schemas.results import (
    BenchmarkResults,
    DetectionResult,
    FieldLeak,
)

__all__ = [
    # Version
    "__version__",
    # Scenario
    "Scenario",
    "ScenarioSet",
    "Vertical",
    "Difficulty",
    "AdversaryLevel",
    "AttackFamily",
    "AttackClass",
    "AttackConfig",
    "Channel",
    # Trace
    "EventType",
    "TraceEvent",
    "ExecutionTrace",
    # Results
    "FieldLeak",
    "DetectionResult",
    "BenchmarkResults",
]
