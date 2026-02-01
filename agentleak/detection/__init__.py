# AgentLeak Detection Pipeline
"""
Detection Architecture (IEEE Paper Section 7)
==============================================

Three-tier hybrid approach: Presidio (Tier 1-2) + LLM-as-Judge (Tier 3)

TIER 1 & 2: PRESIDIO (Pattern + NER)
├── Exact canary token matching
├── Standard PII: SSN, Credit Card, IBAN, Email, Phone
├── Custom recognizers: Patient ID, IMEI, VIN, Crypto, etc.
├── ✅ Fast, deterministic, no API calls
└── Detects: ~18% of leaks (direct pattern matches)

TIER 3: LLM-as-Judge (Model-Agnostic)
├── Semantic paraphrase detection
├── Inference & derivation analysis  
├── Context-aware evaluation
├── ✅ Handles complex semantic leakage
├── Supports: OpenRouter, OpenAI, Anthropic, Google
└── Detects: ~82% of leaks (semantic matches)

Quick Start:
    from agentleak.detection import HybridPipeline
    
    pipeline = HybridPipeline()
    result = pipeline.analyze(text, vault)
    print(f"Leaks detected: {result.all_leaks}")

Reference: AgentLeak IEEE Paper Section 7 (Detection Pipeline)
"""

# =============================================================================
# 🎯 RECOMMENDED: HYBRID PIPELINE (Presidio + LLM-as-Judge)
# =============================================================================
from .hybrid_pipeline import (
    HybridPipeline,
    HybridConfig,
    HybridResult,
    create_hybrid_pipeline,
    detect_leakage,
)

# =============================================================================
# TIER 1 & 2: PRESIDIO DETECTOR (Pattern + NER)
# =============================================================================
from .presidio_detector import (
    PresidioDetector,
    PresidioConfig,
    PresidioMatch,
    PIICategory,
    analyze_text,
)

# =============================================================================
# TIER 3: LLM JUDGE (Model-Agnostic LLM-as-Judge)
# =============================================================================
from .llm_judge import (
    LLMJudge,
    LLMJudgeConfig,
    LLMConfig,  # Legacy alias
    LLMProvider,
    JudgmentResult,
    SemanticLeak,
    LeakageSeverity,
    evaluate_semantic_leakage,
    quick_evaluate,
)

# =============================================================================
# CORE TYPES (From detector module)
# =============================================================================
from ..core.detector import (
    DetectionTier,
    DetectionResult,
)


__all__ = [
    # ========== RECOMMENDED API ==========
    # Hybrid Pipeline (entry point)
    "HybridPipeline",
    "HybridConfig",
    "HybridResult",
    "create_hybrid_pipeline",
    "detect_leakage",
    
    # ========== TIER 1 & 2 API ==========
    # Presidio Detector
    "PresidioDetector",
    "PresidioConfig",
    "PresidioMatch",
    "PIICategory",
    "analyze_text",
    
    # ========== TIER 3 API ==========
    # LLM Judge (Model-Agnostic)
    "LLMJudge",
    "LLMJudgeConfig",
    "LLMConfig",  # Legacy alias
    "LLMProvider",
    "JudgmentResult",
    "SemanticLeak",
    "LeakageSeverity",
    "evaluate_semantic_leakage",
    "quick_evaluate",
    
    # ========== CORE TYPES ==========
    "DetectionTier",
    "DetectionResult",
]
