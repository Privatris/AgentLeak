"""
Legacy generators - archived for compatibility.

These generators are deprecated and archived.
Use RealDataLoader and HardScenarioGenerator instead.
"""

from .canary_generator import (
    CanaryGenerator,
    generate_obvious_canary,
    generate_realistic_canary,
    generate_semantic_canary,
)
from .vault_generator import (
    VaultGenerator,
    generate_vault,
)

__all__ = [
    "CanaryGenerator",
    "generate_obvious_canary",
    "generate_realistic_canary",
    "generate_semantic_canary",
    "VaultGenerator",
    "generate_vault",
]
