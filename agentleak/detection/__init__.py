# AgentLeak Detection Pipeline
"""
3-Stage Leakage Detection Pipeline for agentleak.

The detection pipeline operates in three stages to handle the 3-tier canary system:

Stage 1 - CanaryMatcher (Tier 1: Obvious canaries)
  - Exact string matching for CANARY_* tokens
  - Fast, high precision, 100% recall for T1

Stage 2 - PatternAuditor (Tier 2: Realistic canaries)
  - Regex-based detection for SSN, phone, credit card patterns
  - Reserved ranges (SSN 900-999, phone 555-01xx)
  - High precision via allowlist filtering

Stage 3 - SemanticDetector (Tier 3: Semantic canaries)
  - Embedding similarity for natural language facts
  - Configurable threshold (default 0.72)
  - Handles paraphrasing and reformulation

 additions:
  - Multi-level probing evaluation (seed, vignette, trajectory)
  - Two-stage leakage detection (extract → judge)
  - Helpfulness evaluation for utility-privacy tradeoff
"""

from .canary_matcher import CanaryMatcher
from .pattern_auditor import PatternAuditor
from .semantic_detector import SemanticDetector
from .pipeline import DetectionPipeline, DetectionConfig
from .privacy_evaluator import (
    PrivacyEvaluator,
    PrivacyJudgeResult,
    JudgeConfig,
    JudgeModel,
    DataMinimizationEvaluator,
)

# : Multi-level probing
from .probing_evaluation import (
    ProbingEvaluator,
    ProbingLevel,
    ProbingResult,
    ProbingMetrics,
    compute_probing_metrics,
    compute_knowledge_action_gap,
    export_probing_results,
)

# : Two-stage leakage detection
from .leakage_detector import (
    LeakageDetector,
    LeakageVerdict,
    HelpfulnessRating,
    SensitiveInfoExtraction,
    LeakageJudgment,
    HelpfulnessJudgment,
    FullEvaluationResult,
    AggregateMetrics,
    compute_aggregate_metrics,
    export_results,
)

__all__ = [
    "CanaryMatcher",
    "PatternAuditor",
    "SemanticDetector",
    "DetectionPipeline",
    "DetectionConfig",
    # LLM-based evaluator (AgentDAM-style)
    "PrivacyEvaluator",
    "PrivacyJudgeResult",
    "JudgeConfig",
    "JudgeModel",
    "DataMinimizationEvaluator",
    # Multi-level probing
    "ProbingEvaluator",
    "ProbingLevel",
    "ProbingResult",
    "ProbingMetrics",
    "compute_probing_metrics",
    "compute_knowledge_action_gap",
    "export_probing_results",
    # Leakage detection
    "LeakageDetector",
    "LeakageVerdict",
    "HelpfulnessRating",
    "SensitiveInfoExtraction",
    "LeakageJudgment",
    "HelpfulnessJudgment",
    "FullEvaluationResult",
    "AggregateMetrics",
    "compute_aggregate_metrics",
    "export_results",
]
