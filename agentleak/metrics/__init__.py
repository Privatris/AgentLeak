"""
AgentLeak Metrics Module - Privacy and utility metrics for agent evaluation.

Metrics:
- ELR: Exact Leakage Rate (proportion of private fields leaked)
- WLS: Weighted Leakage Score (sensitivity-weighted leakage)
- CLR: Channel Leakage Rate (per-channel breakdown)
- ASR: Attack Success Rate (proportion of successful attacks)
- TSR: Task Success Rate (proportion of successfully completed tasks)
- Pareto AUC: Area under privacy-utility tradeoff curve
"""

from .aggregator import (
    AggregatedMetrics,
    MetricsAggregator,
    VerticalBreakdown,
)
from .core import (
    AttackMetrics,
    BenchmarkMetrics,
    ChannelMetrics,
    LeakageMetrics,
    MetricsCalculator,
    ParetoMetrics,
    TaskMetrics,
)
from .pareto import (
    ParetoCalculator,
    ParetoFrontier,
    ParetoPoint,
)

__all__ = [
    # Core metrics
    "MetricsCalculator",
    "LeakageMetrics",
    "ChannelMetrics",
    "AttackMetrics",
    "TaskMetrics",
    "ParetoMetrics",
    "BenchmarkMetrics",
    # Aggregation
    "MetricsAggregator",
    "AggregatedMetrics",
    "VerticalBreakdown",
    # Pareto
    "ParetoCalculator",
    "ParetoPoint",
    "ParetoFrontier",
    # Utility evaluator (WebArena-style)
    "CombinedEvaluator",
    "EvalConfig",
    "EvalResult",
    "EvalType",
    "evaluator_router",
]

# Import utility evaluator
try:
    from .utility_evaluator import (
        CombinedEvaluator,
        ContentCheckEvaluator,
        EvalConfig,
        EvalResult,
        EvalType,
        HTMLContentEvaluator,
        LLMJudgeEvaluator,
        StringMatchEvaluator,
        evaluator_router,
    )
except ImportError:
    pass  # Optional module

# Import strict evaluator ()
try:
    from .strict_evaluator import (
        FailureMode,
        StrictEvalResult,
        StrictTaskEvaluator,
    )
except ImportError:
    pass  # Optional module

# Import multi-agent evaluator
try:
    from .multiagent_evaluator import (
        AgentEnvironmentState,
        CrossAgentLeakageResult,
        DataFlow,
        ExpectedToolCall,
        MultiAgentEnvironment,
        MultiAgentEvalResult,
        MultiAgentEvaluator,
        MultiAgentGroundTruth,
        ProbingLevel,
        ProbingResult,
        clearance_allows,
        create_test_environment,
        detect_clearance_violation,
        generate_multi_agent_ground_truth,
        generate_seed_probing_prompt,
        generate_trajectory_probing_prompt,
        generate_vignette_probing_prompt,
    )
except ImportError:
    pass  # Optional module
