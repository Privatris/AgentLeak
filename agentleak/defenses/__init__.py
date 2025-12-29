"""
AgentLeak Defenses Module - Privacy defenses for agentic systems.

Implements baseline defenses as described in the paper (Section 8):

1. Output Filtering: PII scrubbers (regex + NER) on final outputs
2. Input Sanitization: Filter sensitive data before agent execution
3. System Prompt Hardening: Privacy-aware prompts with CoT reasoning
4. Role Separation: Multi-agent clearance levels (handled in harness)
5. Memory Minimization: Disable persistent memory (config option)

External Guardrail Systems (for comparison):
- PromptGuard (Meta)
- NeMo Guardrails (NVIDIA)
- LlamaGuard 3 (Meta)
- Lakera Guard (commercial)
- Rebuff (open-source)
"""

from .sanitizer import (
    OutputSanitizer,
    SanitizerConfig,
    SanitizationResult,
)

from .base import (
    BaseDefense,
    DefenseResult,
    DefenseConfig,
    FilterAction,
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
    # Base
    "BaseDefense",
    "DefenseResult",
    "DefenseConfig",
    "FilterAction",
    # Sanitizer (Output Filtering)
    "OutputSanitizer",
    "SanitizerConfig",
    "SanitizationResult",
    # Mitigation (System Prompt Hardening + CoT)
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
