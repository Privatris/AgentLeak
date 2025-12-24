"""
APB Defenses Module - Privacy defenses for agentic systems.

Implements:
- LCF: Learned Content Filter (primary defense)
- OutputSanitizer: Rule-based output filtering
- MemoryGuard: Memory access control
- Mitigation: AgentDAM-style privacy prompts and CoT
"""

from .lcf import (
    LearnedContentFilter,
    LCFConfig,
    LCFTrainer,
    FilterDecision,
)

from .sanitizer import (
    OutputSanitizer,
    SanitizerConfig,
    SanitizationResult,
)

from .base import (
    BaseDefense,
    DefenseResult,
    DefenseConfig,
)

from .mitigation import (
    MitigationType,
    MitigationPipeline,
    MitigationResult,
    PromptBuilder,
    PromptConfig,
    PreFilterMitigation,
    PostFilterMitigation,
    create_mitigation,
    COT_PRIVACY_PROMPT,
    PRIVACY_SYSTEM_PROMPT,
)

__all__ = [
    # LCF
    "LearnedContentFilter",
    "LCFConfig",
    "LCFTrainer",
    "FilterDecision",
    # Sanitizer
    "OutputSanitizer",
    "SanitizerConfig",
    "SanitizationResult",
    # Base
    "BaseDefense",
    "DefenseResult",
    "DefenseConfig",
    # Mitigation (AgentDAM-style)
    "MitigationType",
    "MitigationPipeline",
    "MitigationResult",
    "PromptBuilder",
    "PromptConfig",
    "PreFilterMitigation",
    "PostFilterMitigation",
    "create_mitigation",
    "COT_PRIVACY_PROMPT",
    "PRIVACY_SYSTEM_PROMPT",
]
