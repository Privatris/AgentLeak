"""
AgentLeak - A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems

Based on the paper: "AgentLeak: A Full-Stack Benchmark for Privacy Leakage
in Multi-Agent LLM Systems" (December 2025)

Features:
- 7 leakage channels (C1-C7)
- 19 attack classes in 5 families (F1-F5)
- Multi-tier detection pipeline
- Output sanitizer and regex-based defenses
- Interactive CLI with real-time visualization

Modules:
    - core: Core data structures (channels, attacks, scenarios)
    - schemas: Pydantic models for scenarios, traces, results
    - harness: Framework adapters and trace collection
    - detection: Multi-tier leakage detection
    - defenses: Privacy defense mechanisms
    - metrics: ELR, WLS, TSR computation
    - generators: Scenario and payload generation
    - cli: Interactive command-line interface
    - config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Faouzi EL YAGOUBI, Ranwa AL MALLAH"

# Core structures
# Configuration
from .config import Config, load_config
from .core.attacks import AdversaryLevel, AttackClass, AttackFamily, AttackManager
from .core.channels import Channel, ChannelManager
from .core.scenarios import Scenario, ScenarioGenerator

__all__ = [
    # Version
    "__version__",
    # Core
    "Channel",
    "ChannelManager",
    "AttackClass",
    "AttackFamily",
    "AttackManager",
    "AdversaryLevel",
    "Scenario",
    "ScenarioGenerator",
    # Config
    "Config",
    "load_config",
    # Main entry point
    "main",
]


def main():
    """Main entry point for agentleak command."""
    from .__main__ import main as _main

    _main()
