"""
AgentLeak Defenses Module - Privacy defenses for agentic systems.

================================================================================
DEFENSE FAMILIES (Based on Paper Section 9)
================================================================================

The paper identifies that existing defenses have a CRITICAL GAP:
- External channels (C1 output, C3 tools): 3.8% leak rate (defended)
- Internal channels (C2 inter-agent, C5 memory): 31.5% leak rate (UNDEFENDED)
- 8.3x higher leak rate on internal channels

This module provides defenses for ALL channels, addressing the gap.

FAMILY 1: OUTPUT FILTERING (C1)
├── OutputSanitizer: Regex-based PII redaction
├── PresidioDefense: NER-based PII detection + redaction
└── GuardrailDefense: External guardrail integration

FAMILY 2: INTERNAL CHANNEL PROTECTION (C2, C5) ⭐ KEY DIFFERENTIATOR
├── InternalChannelDefense: Message + memory filtering
├── PresidioDefense.protect_message(): Inter-agent PII filter
└── PresidioDefense.protect_memory(): Memory write filter

FAMILY 3: PROMPT ENGINEERING (System-level)
├── PromptBuilder: Privacy-aware prompt construction
├── COT_PRIVACY_PROMPT: Chain-of-thought privacy reasoning
└── PRIVACY_SYSTEM_PROMPT: Basic privacy instructions

FAMILY 4: PRE/POST FILTERS (Input/Output)
├── PreFilterMitigation: Filter sensitive data before agent
├── PostFilterMitigation: Check outputs for leaks after agent
└── MitigationPipeline: Combined pre+post filtering

FAMILY 5: EXTERNAL GUARDRAILS (Third-party)
├── LLMGuardIntegration: ProtectAI LLM Guard
├── GuardrailsAIIntegration: Guardrails AI
└── PresidioFallback: Internal fallback

================================================================================
USAGE
================================================================================

Quick Start (Recommended):
    from agentleak.defenses import PresidioDefense, create_presidio_defense
    
    # Full channel protection
    defense = PresidioDefense()
    result = defense.filter("SSN: 123-45-6789", Channel.C1_FINAL_OUTPUT)
    
    # Internal channel protection (addresses paper's key gap)
    defense = create_internal_channel_defense()
    msg = defense.protect_message("Patient SSN is 123-45-6789", "Agent1", "Agent2")

With External Guardrails:
    from agentleak.defenses import GuardrailDefense, get_best_available_guardrail
    
    defense = get_best_available_guardrail()  # Auto-selects best available
    result = defense.filter("Email: john@example.com", Channel.C1_FINAL_OUTPUT)

Prompt Engineering:
    from agentleak.defenses import PromptBuilder, COT_PRIVACY_PROMPT
    
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt()

================================================================================
"""

# =============================================================================
# FAMILY 1: BASE CLASSES
# =============================================================================
from .base import (
    BaseDefense,
    DefenseConfig,
    DefenseResult,
    FilterAction,
)

# =============================================================================
# FAMILY 2: OUTPUT FILTERING (C1)
# =============================================================================
from .sanitizer import (
    OutputSanitizer,
    SanitizerConfig,
    SanitizationResult,
)

from .presidio_defense import (
    PresidioDefense,
    PresidioDefenseConfig,
    PresidioDefenseResult,
    RedactionStyle,
    create_presidio_defense,
    create_internal_channel_defense as create_presidio_internal_defense,
)

# =============================================================================
# FAMILY 3: INTERNAL CHANNEL PROTECTION (C2, C5) - KEY DIFFERENTIATOR
# =============================================================================
from .internal_channel import (
    InternalChannelDefense,
    InternalChannelConfig,
    MessageFilter,
    MemoryFilter,
    ClearanceLevel,
    AgentProfile,
    create_internal_defense,
    wrap_message_handler,
    # Framework hooks
    FrameworkHook,
    CrewAIHook,
    LangChainHook,
)

# =============================================================================
# FAMILY 4: PROMPT ENGINEERING + PRE/POST FILTERS
# =============================================================================
from .mitigation import (
    # Prompt engineering
    PromptBuilder,
    PromptConfig,
    COT_PRIVACY_PROMPT,
    PRIVACY_SYSTEM_PROMPT,
    COT_EXAMPLES,
    # Mitigation types
    MitigationType,
    # Pre/Post filters
    PreFilterMitigation,
    PostFilterMitigation,
    # Pipeline
    MitigationPipeline,
    MitigationResult,
    create_mitigation,
)

# =============================================================================
# FAMILY 5: EXTERNAL GUARDRAILS
# =============================================================================
from .guardrails import (
    GuardrailDefense,
    GuardrailConfig,
    GuardrailResult,
    GuardrailType,
    # Integrations
    LLMGuardIntegration,
    GuardrailsAIIntegration,
    PresidioFallback,
    GuardrailBase,
    # Factory functions
    create_guardrail_defense,
    get_best_available_guardrail,
)


# =============================================================================
# CONVENIENCE: Create internal channel defense (addresses paper's key gap)
# =============================================================================
def create_internal_channel_defense():
    """
    Create defense for internal channels (C2, C5).
    
    This addresses the KEY FINDING from the paper:
    - Internal channels have 8.3x higher leak rates than external
    - 31.5% vs 3.8% leak rate
    - NO existing defense covers C2/C5
    
    Returns:
        PresidioDefense configured for internal channels
    """
    return create_presidio_internal_defense()


# =============================================================================
# ALL EXPORTS
# =============================================================================
__all__ = [
    # ========== BASE ==========
    "BaseDefense",
    "DefenseConfig",
    "DefenseResult",
    "FilterAction",
    
    # ========== OUTPUT FILTERING (C1) ==========
    # Sanitizer (regex-based)
    "OutputSanitizer",
    "SanitizerConfig",
    "SanitizationResult",
    # Presidio (NER-based)
    "PresidioDefense",
    "PresidioDefenseConfig",
    "PresidioDefenseResult",
    "RedactionStyle",
    "create_presidio_defense",
    
    # ========== INTERNAL CHANNEL PROTECTION (C2, C5) ==========
    "InternalChannelDefense",
    "InternalChannelConfig",
    "MessageFilter",
    "MemoryFilter",
    "ClearanceLevel",
    "AgentProfile",
    "create_internal_defense",
    "create_internal_channel_defense",
    "wrap_message_handler",
    # Framework hooks
    "FrameworkHook",
    "CrewAIHook",
    "LangChainHook",
    
    # ========== PROMPT ENGINEERING ==========
    "PromptBuilder",
    "PromptConfig",
    "COT_PRIVACY_PROMPT",
    "PRIVACY_SYSTEM_PROMPT",
    "COT_EXAMPLES",
    "MitigationType",
    
    # ========== PRE/POST FILTERS ==========
    "PreFilterMitigation",
    "PostFilterMitigation",
    "MitigationPipeline",
    "MitigationResult",
    "create_mitigation",
    
    # ========== EXTERNAL GUARDRAILS ==========
    "GuardrailDefense",
    "GuardrailConfig",
    "GuardrailResult",
    "GuardrailType",
    "LLMGuardIntegration",
    "GuardrailsAIIntegration",
    "PresidioFallback",
    "GuardrailBase",
    "create_guardrail_defense",
    "get_best_available_guardrail",
]
