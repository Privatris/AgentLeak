"""
AgentLeak Core Module - Fundamental data structures and abstractions.

Contains:
- channels: The 7 leakage channels (C1-C7)
- attacks: 32 attack classes in 6 families
- scenarios: Benchmark scenario definitions
- traces: Execution trace structures
"""

from .attacks import AttackClass, AttackFamily, AttackPayload
from .channels import Channel, ChannelManager
from .scenarios import AllowedSet, PrivateVault, Scenario

__all__ = [
    "Channel",
    "ChannelManager",
    "AttackClass",
    "AttackFamily",
    "AttackPayload",
    "Scenario",
    "PrivateVault",
    "AllowedSet",
]
