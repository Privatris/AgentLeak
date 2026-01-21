# AgentLeak Detection Pipeline
"""
MODERN Detection Architecture (Recommended)
==============================================

Two-tier hybrid approach optimized for performance and accuracy:

TIER 1 & 2: PRESIDIO (Pattern + NER)
├── Exact canary token matching
├── Standard PII: SSN, Credit Card, IBAN, Email, Phone
├── Custom recognizers: Patient ID, IMEI, VIN, Crypto, etc.
├── ✅ Fast, deterministic, no API calls
└── Detects: ~18% of leaks (direct pattern matches)

TIER 3: GEMINI 2.0 FLASH (LLM-as-Judge)
├── Semantic paraphrase detection
├── Inference & derivation analysis  
├── Context-aware evaluation
├── ✅ Handles complex semantic leakage
└── Detects: ~82% of leaks (semantic matches)

Quick Start:
    from agentleak.detection import HybridPipeline
    
    pipeline = HybridPipeline()
    result = pipeline.analyze(text, vault)
    print(f"Leaks detected: {result.all_leaks}")

================================================================================
LEGACY API (For backward compatibility, use modern API instead)
================================================================================
"""

# =============================================================================
# 🎯 RECOMMENDED: HYBRID PIPELINE (Presidio + Gemini)
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
# TIER 3: GEMINI JUDGE (LLM-as-Judge)
# =============================================================================
from .gemini_judge import (
    GeminiJudge,
    GeminiJudgeConfig,
    JudgmentResult,
    SemanticLeak,
    LeakageSeverity,
    evaluate_semantic_leakage,
)

# =============================================================================
# LEGACY API: PrivacyJudge (For backward compatibility with benchmarks)
# =============================================================================
from .privacy_judge import (
    PrivacyJudge,
    JudgeConfiguration,
    LLMProvider,
    PrivacyJudgment,
    quick_evaluate,
    evaluate_trace,
    evaluate_scenario,
    evaluate_benchmark,
)

# =============================================================================
# LEGACY API: Basic Detectors (For backward compatibility)
# =============================================================================
from .basic_detectors import (
    ExactDetector,
    PatternDetector,
    SemanticDetector,
    DetectionTier,
    DetectionResult,
)
from .llm_judge import LLMJudge, LLMConfig


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
    # Gemini Judge
    "GeminiJudge",
    "GeminiJudgeConfig",
    "JudgmentResult",
    "SemanticLeak",
    "LeakageSeverity",
    "evaluate_semantic_leakage",
    
    # ========== LEGACY API (Benchmarks) ==========
    "PrivacyJudge",
    "JudgeConfiguration",
    "LLMProvider",
    "PrivacyJudgment",
    "quick_evaluate",
    "evaluate_trace",
    "evaluate_scenario",
    "evaluate_benchmark",
    
    # ========== LEGACY COMPATIBILITY ==========
    "ExactDetector",
    "PatternDetector",
    "SemanticDetector",
    "DetectionTier",
    "DetectionResult",
    "LLMJudge",
    "LLMConfig",
]
