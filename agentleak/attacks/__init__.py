"""APB Attacks - 15 attack classes across 4 families."""

from agentleak.attacks.attack_module import (
    AttackPayload,
    BaseAttack,
    ATTACK_REGISTRY,
    get_attack,
    generate_attack_payload,
    # F1: Prompt & Instruction
    DirectPromptInjection,
    RoleConfusion,
    ContextOverride,
    FormatCoercion,
    # F2: Tool Surface
    IndirectPromptInjection,
    ToolOutputPoisoning,
    RetrievalTrap,
    LinkFollowingExfiltration,
    # F3: Memory & Persistence
    MemoryWriteExfiltration,
    VectorStoreLeakage,
    LogLeakage,
    ArtifactLeakage,
    # F4: Multi-Agent
    CrossAgentCollusion,
    RoleBoundaryViolation,
    DelegationExploit,
)

__all__ = [
    # Core
    "AttackPayload",
    "BaseAttack",
    "ATTACK_REGISTRY",
    "get_attack",
    "generate_attack_payload",
    # F1
    "DirectPromptInjection",
    "RoleConfusion",
    "ContextOverride",
    "FormatCoercion",
    # F2
    "IndirectPromptInjection",
    "ToolOutputPoisoning",
    "RetrievalTrap",
    "LinkFollowingExfiltration",
    # F3
    "MemoryWriteExfiltration",
    "VectorStoreLeakage",
    "LogLeakage",
    "ArtifactLeakage",
    # F4
    "CrossAgentCollusion",
    "RoleBoundaryViolation",
    "DelegationExploit",
]
