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

from .core import (
    MetricsCalculator,
    LeakageMetrics,
    ChannelMetrics,
    AttackMetrics,
    TaskMetrics,
    ParetoMetrics,
    BenchmarkMetrics,
)

from .aggregator import (
    MetricsAggregator,
    AggregatedMetrics,
    VerticalBreakdown,
)

from .pareto import (
    ParetoCalculator,
    ParetoPoint,
    ParetoFrontier,
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
        EvalConfig,
        EvalResult,
        EvalType,
        evaluator_router,
        StringMatchEvaluator,
        ContentCheckEvaluator,
        HTMLContentEvaluator,
        LLMJudgeEvaluator,
    )
except ImportError:
    pass  # Optional module

# Import strict evaluator ()
try:
    from .strict_evaluator import (
        StrictTaskEvaluator,
        StrictEvalResult,
        FailureMode,
    )
except ImportError:
    pass  # Optional module

# Import multi-agent evaluator 
try:
    from .multiagent_evaluator import (
        MultiAgentEvaluator,
        MultiAgentEnvironment,
        AgentEnvironmentState,
        MultiAgentEvalResult,
        MultiAgentGroundTruth,
        ExpectedToolCall,
        DataFlow,
        ProbingLevel,
        ProbingResult,
        CrossAgentLeakageResult,
        clearance_allows,
        detect_clearance_violation,
        generate_seed_probing_prompt,
        generate_vignette_probing_prompt,
        generate_trajectory_probing_prompt,
        generate_multi_agent_ground_truth,
        create_test_environment,
    )
except ImportError:
    pass  # Optional module
