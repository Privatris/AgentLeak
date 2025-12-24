"""
APB Metrics Module - Privacy and utility metrics for agent evaluation.

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
