"""
AgentLeak - Professional Multi-Agent Privacy Testing Framework

Clean API for testing privacy leaks in multi-agent LLM systems.

Quick Start:
    >>> from agentleak import AgentLeakTester, DetectionMode
    >>> 
    >>> # Simple privacy check
    >>> from agentleak import quick_check
    >>> result = quick_check(
    ...     vault={"ssn": "123-45-6789"},
    ...     output="User SSN is 123-45-6789",
    ...     mode="fast"
    ... )
    >>> print(f"Leaked: {result.leaked}")
    
    >>> # Advanced testing
    >>> tester = AgentLeakTester(mode=DetectionMode.HYBRID)
    >>> result = tester.check(
    ...     vault=sensitive_data,
    ...     output=agent_output,
    ...     channel="C1"
    ... )

Features:
    - 4-tier detection: exact → pattern → semantic → LLM
    - Multi-channel support (C1-C7)
    - LLM-as-Judge integrated natively
    - Early stopping for efficiency
    - Clean, professional API
"""

__version__ = "1.0.0"
__author__ = "Faouzi EL YAGOUBI, Ranwa AL MALLAH"

# Main API
from .tester import (
    AgentLeakTester,
    DetectionMode,
    TestResult,
    quick_check
)

# Core detection
from .core.detector import (
    Detector,
    DetectionResult,
    DetectionTier,
    DetectorPipeline
)

# Individual detectors
from .detection.basic_detectors import (
    ExactDetector,
    PatternDetector,
    SemanticDetector
)
from .detection.llm_judge import (
    LLMJudge,
    LLMConfig
)

# Legacy core (for backwards compatibility)
from .core.attacks import AdversaryLevel, AttackClass, AttackFamily
from .core.channels import Channel
from .config import Config, load_config


__all__ = [
    # Version
    "__version__",
    
    # Main API (recommended)
    "AgentLeakTester",
    "DetectionMode", 
    "TestResult",
    "quick_check",
    
    # Core detection (advanced)
    "Detector",
    "DetectionResult",
    "DetectionTier",
    "DetectorPipeline",
    
    # Individual detectors (advanced)
    "ExactDetector",
    "PatternDetector",
    "SemanticDetector",
    "LLMJudge",
    "LLMConfig",
    
    # Legacy (backwards compatibility)
    "Channel",
    "AttackClass",
    "AttackFamily",
    "AdversaryLevel",
    "Config",
    "load_config",
]


def main():
    """Main entry point for agentleak command."""
    from .__main__ import main as _main

    _main()
