"""
AgentLeak Catalog - Centralized definitions for attacks, defenses, and frameworks.

This module provides the canonical taxonomy from the paper:
- 32 attack classes across 6 families (F1-F6)
- 7 leakage channels (C1-C7)
- 3 adversary levels (A0-A2)
- 4 defense categories (D1-D4)
- Framework adapters

Usage:
    from agentleak.catalog import ATTACKS, DEFENSES, FRAMEWORKS, CHANNELS
    
    # Get attack info
    attack = ATTACKS.get("F1.1")  # Direct Prompt Injection
    
    # Get defense
    defense = DEFENSES.get("D1")  # Output Sanitizer
    
    # List all attacks in family
    f4_attacks = ATTACKS.by_family("F4")
"""

from .attacks import ATTACKS, AttackClass, AttackFamily, AdversaryLevel
from .defenses import DEFENSES, DefenseCategory
from .channels import CHANNELS, Channel
from .frameworks import FRAMEWORKS, FrameworkAdapter

__all__ = [
    "ATTACKS",
    "DEFENSES", 
    "CHANNELS",
    "FRAMEWORKS",
    "AttackClass",
    "AttackFamily",
    "AdversaryLevel",
    "DefenseCategory",
    "Channel",
    "FrameworkAdapter",
]
