"""
AgentLeak Defense Catalog - Defense mechanisms for privacy protection.

From the paper "AgentLeak: Privacy Benchmark for Multi-Agent LLM Systems":

Defense Categories:
    D1: Output Sanitization - Filter PII from final outputs
    D2: Input Validation - Detect and block malicious inputs
    D3: Memory Guard - Protect persistent storage
    D4: Channel Isolation - Isolate inter-agent communication

Defense Strategies:
    - Pattern-based: Regex, keyword matching
    - ML-based: NER, classifier models
    - LLM-based: Judge models for semantic analysis
    - Hybrid: Combination of above
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union

from .channels import Channel


class DefenseCategory(str, Enum):
    """The 4 defense categories from the paper."""
    D1 = "D1"  # Output Sanitization
    D2 = "D2"  # Input Validation
    D3 = "D3"  # Memory Guard
    D4 = "D4"  # Channel Isolation

    @property
    def name(self) -> str:
        return {
            "D1": "Output Sanitization",
            "D2": "Input Validation",
            "D3": "Memory Guard",
            "D4": "Channel Isolation",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "D1": "Filter PII and sensitive data from outputs before delivery",
            "D2": "Detect and block prompt injection and malicious inputs",
            "D3": "Protect memory writes and prevent cross-session leakage",
            "D4": "Isolate communication channels between agents",
        }[self.value]

    @property
    def target_channels(self) -> List[str]:
        """Which channels this defense protects."""
        return {
            "D1": ["C1", "C7"],       # Final output, artifacts
            "D2": ["C3", "C4"],       # Tool inputs/outputs
            "D3": ["C5", "C6"],       # Memory, logs
            "D4": ["C2"],             # Inter-agent messages
        }[self.value]


class DefenseStrategy(str, Enum):
    """Defense implementation strategies."""
    PATTERN = "pattern"     # Regex, keyword matching
    ML = "ml"               # Machine learning models (NER, classifiers)
    LLM = "llm"             # LLM-based judge/filter
    HYBRID = "hybrid"       # Combination of strategies
    RULE = "rule"           # Rule-based logic


class FilterAction(str, Enum):
    """Actions a defense can take."""
    ALLOW = "allow"         # Pass through unchanged
    BLOCK = "block"         # Block entirely
    REDACT = "redact"       # Redact sensitive parts
    WARN = "warn"           # Allow but log warning
    TRANSFORM = "transform" # Transform/anonymize


@dataclass
class DefenseResult:
    """Result from applying a defense."""
    action: FilterAction
    original: str
    filtered: Optional[str] = None
    confidence: float = 1.0
    detections: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    reason: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return self.action == FilterAction.BLOCK

    @property
    def modified(self) -> bool:
        return self.action in [FilterAction.REDACT, FilterAction.TRANSFORM]


@dataclass
class DefenseConfig:
    """Configuration for a defense."""
    enabled: bool = True
    strategy: DefenseStrategy = DefenseStrategy.PATTERN
    strict_mode: bool = False  # Block on any suspicion
    log_decisions: bool = True
    channels: List[Channel] = field(default_factory=list)
    custom_patterns: List[str] = field(default_factory=list)
    threshold: float = 0.5  # Confidence threshold for ML/LLM


@dataclass
class DefenseSpec:
    """Specification of a defense mechanism."""
    id: str                          # e.g., "D1.1"
    name: str                        # e.g., "PII Regex Filter"
    category: DefenseCategory        # e.g., D1
    strategy: DefenseStrategy        # e.g., PATTERN
    description: str
    target_channels: List[str]
    effectiveness: Dict[str, float] = field(default_factory=dict)  # By attack family
    latency_ms: float = 0.0          # Average latency
    false_positive_rate: float = 0.0
    implementation: Optional[str] = None  # Python class/function path


# =============================================================================
# D1: Output Sanitization Defenses
# =============================================================================

D1_DEFENSES = [
    DefenseSpec(
        id="D1.1",
        name="PII Regex Filter",
        category=DefenseCategory.D1,
        strategy=DefenseStrategy.PATTERN,
        description="Regex-based detection and redaction of common PII patterns (SSN, email, phone, etc.)",
        target_channels=["C1", "C7"],
        effectiveness={"F1": 0.65, "F2": 0.55, "F3": 0.40, "F4": 0.35, "F5": 0.50, "F6": 0.30},
        latency_ms=5.0,
        false_positive_rate=0.15,
        implementation="agentleak.defenses.sanitizer.PIIRegexFilter",
    ),
    DefenseSpec(
        id="D1.2",
        name="NER-based Sanitizer",
        category=DefenseCategory.D1,
        strategy=DefenseStrategy.ML,
        description="Named Entity Recognition to detect and redact sensitive entities",
        target_channels=["C1", "C7"],
        effectiveness={"F1": 0.75, "F2": 0.70, "F3": 0.55, "F4": 0.50, "F5": 0.60, "F6": 0.45},
        latency_ms=50.0,
        false_positive_rate=0.10,
        implementation="agentleak.defenses.sanitizer.NERSanitizer",
    ),
    DefenseSpec(
        id="D1.3",
        name="LLM Judge Filter",
        category=DefenseCategory.D1,
        strategy=DefenseStrategy.LLM,
        description="LLM-based semantic analysis to detect privacy violations",
        target_channels=["C1", "C7"],
        effectiveness={"F1": 0.85, "F2": 0.80, "F3": 0.70, "F4": 0.65, "F5": 0.75, "F6": 0.60},
        latency_ms=500.0,
        false_positive_rate=0.05,
        implementation="agentleak.defenses.sanitizer.LLMJudgeFilter",
    ),
    DefenseSpec(
        id="D1.4",
        name="Canary Token Detector",
        category=DefenseCategory.D1,
        strategy=DefenseStrategy.PATTERN,
        description="Detect planted canary tokens in outputs to identify leakage",
        target_channels=["C1", "C2", "C7"],
        effectiveness={"F1": 0.95, "F2": 0.90, "F3": 0.85, "F4": 0.80, "F5": 0.85, "F6": 0.70},
        latency_ms=2.0,
        false_positive_rate=0.01,
        implementation="agentleak.defenses.sanitizer.CanaryDetector",
    ),
]

# =============================================================================
# D2: Input Validation Defenses
# =============================================================================

D2_DEFENSES = [
    DefenseSpec(
        id="D2.1",
        name="Prompt Injection Detector",
        category=DefenseCategory.D2,
        strategy=DefenseStrategy.HYBRID,
        description="Detect and block prompt injection attempts in user inputs",
        target_channels=["C3"],
        effectiveness={"F1": 0.70, "F2": 0.60, "F3": 0.30, "F4": 0.25, "F5": 0.55, "F6": 0.40},
        latency_ms=20.0,
        false_positive_rate=0.12,
        implementation="agentleak.defenses.mitigation.PromptInjectionDetector",
    ),
    DefenseSpec(
        id="D2.2",
        name="Input Sanitizer",
        category=DefenseCategory.D2,
        strategy=DefenseStrategy.PATTERN,
        description="Sanitize and normalize user inputs to remove malicious content",
        target_channels=["C3", "C4"],
        effectiveness={"F1": 0.55, "F2": 0.50, "F3": 0.20, "F4": 0.15, "F5": 0.40, "F6": 0.60},
        latency_ms=3.0,
        false_positive_rate=0.08,
        implementation="agentleak.defenses.mitigation.InputSanitizer",
    ),
    DefenseSpec(
        id="D2.3",
        name="Tool Output Validator",
        category=DefenseCategory.D2,
        strategy=DefenseStrategy.HYBRID,
        description="Validate tool outputs for injection attacks before processing",
        target_channels=["C4"],
        effectiveness={"F1": 0.30, "F2": 0.75, "F3": 0.35, "F4": 0.30, "F5": 0.25, "F6": 0.35},
        latency_ms=15.0,
        false_positive_rate=0.10,
        implementation="agentleak.defenses.mitigation.ToolOutputValidator",
    ),
]

# =============================================================================
# D3: Memory Guard Defenses
# =============================================================================

D3_DEFENSES = [
    DefenseSpec(
        id="D3.1",
        name="Memory Write Filter",
        category=DefenseCategory.D3,
        strategy=DefenseStrategy.HYBRID,
        description="Filter sensitive data before writing to persistent memory",
        target_channels=["C5"],
        effectiveness={"F1": 0.40, "F2": 0.45, "F3": 0.80, "F4": 0.60, "F5": 0.35, "F6": 0.30},
        latency_ms=10.0,
        false_positive_rate=0.08,
        implementation="agentleak.defenses.mitigation.MemoryWriteFilter",
    ),
    DefenseSpec(
        id="D3.2",
        name="Log Redactor",
        category=DefenseCategory.D3,
        strategy=DefenseStrategy.PATTERN,
        description="Automatically redact sensitive data from logs",
        target_channels=["C6"],
        effectiveness={"F1": 0.50, "F2": 0.45, "F3": 0.70, "F4": 0.40, "F5": 0.35, "F6": 0.25},
        latency_ms=5.0,
        false_positive_rate=0.12,
        implementation="agentleak.defenses.mitigation.LogRedactor",
    ),
    DefenseSpec(
        id="D3.3",
        name="Session Isolation",
        category=DefenseCategory.D3,
        strategy=DefenseStrategy.RULE,
        description="Isolate session state to prevent cross-session leakage",
        target_channels=["C5", "C6"],
        effectiveness={"F1": 0.20, "F2": 0.25, "F3": 0.85, "F4": 0.55, "F5": 0.20, "F6": 0.15},
        latency_ms=1.0,
        false_positive_rate=0.02,
        implementation="agentleak.defenses.mitigation.SessionIsolation",
    ),
]

# =============================================================================
# D4: Channel Isolation Defenses
# =============================================================================

D4_DEFENSES = [
    DefenseSpec(
        id="D4.1",
        name="Inter-Agent Message Filter",
        category=DefenseCategory.D4,
        strategy=DefenseStrategy.HYBRID,
        description="Filter sensitive data from inter-agent communications",
        target_channels=["C2"],
        effectiveness={"F1": 0.35, "F2": 0.40, "F3": 0.50, "F4": 0.75, "F5": 0.30, "F6": 0.25},
        latency_ms=15.0,
        false_positive_rate=0.10,
        implementation="agentleak.defenses.mitigation.InterAgentFilter",
    ),
    DefenseSpec(
        id="D4.2",
        name="Role-Based Access Control",
        category=DefenseCategory.D4,
        strategy=DefenseStrategy.RULE,
        description="Enforce role-based access control for agent data access",
        target_channels=["C2", "C5"],
        effectiveness={"F1": 0.25, "F2": 0.30, "F3": 0.45, "F4": 0.70, "F5": 0.20, "F6": 0.15},
        latency_ms=2.0,
        false_positive_rate=0.05,
        implementation="agentleak.defenses.mitigation.RBACController",
    ),
    DefenseSpec(
        id="D4.3",
        name="Delegation Validator",
        category=DefenseCategory.D4,
        strategy=DefenseStrategy.RULE,
        description="Validate task delegations to prevent privilege escalation",
        target_channels=["C2"],
        effectiveness={"F1": 0.20, "F2": 0.25, "F3": 0.35, "F4": 0.80, "F5": 0.15, "F6": 0.10},
        latency_ms=3.0,
        false_positive_rate=0.03,
        implementation="agentleak.defenses.mitigation.DelegationValidator",
    ),
]


# =============================================================================
# Defense Registry
# =============================================================================

class DefenseRegistry:
    """Central registry of all defense mechanisms."""
    
    def __init__(self):
        self._defenses: Dict[str, DefenseSpec] = {}
        self._by_category: Dict[DefenseCategory, List[DefenseSpec]] = {c: [] for c in DefenseCategory}
        
        # Register all defenses
        for defense_list in [D1_DEFENSES, D2_DEFENSES, D3_DEFENSES, D4_DEFENSES]:
            for defense in defense_list:
                self._defenses[defense.id] = defense
                self._defenses[defense.name.lower().replace(" ", "_")] = defense
                self._by_category[defense.category].append(defense)
    
    def get(self, id_or_name: str) -> Optional[DefenseSpec]:
        """Get defense by ID (e.g., 'D1.1') or name."""
        return self._defenses.get(id_or_name) or self._defenses.get(id_or_name.lower().replace(" ", "_"))
    
    def by_category(self, category: Union[str, DefenseCategory]) -> List[DefenseSpec]:
        """Get all defenses in a category."""
        if isinstance(category, str):
            category = DefenseCategory(category)
        return self._by_category.get(category, [])
    
    def by_channel(self, channel: str) -> List[DefenseSpec]:
        """Get all defenses that protect a specific channel."""
        channel = channel.upper()
        return [d for d in self.all() if channel in d.target_channels]
    
    def by_strategy(self, strategy: DefenseStrategy) -> List[DefenseSpec]:
        """Get all defenses using a specific strategy."""
        return [d for d in self.all() if d.strategy == strategy]
    
    def all(self) -> List[DefenseSpec]:
        """Get all defense specs."""
        return [d for d in self._defenses.values() if isinstance(d, DefenseSpec)]
    
    @property
    def total_count(self) -> int:
        """Total number of defense specs."""
        return len([d for d in self._defenses.values() if isinstance(d, DefenseSpec)]) // 2
    
    def effectiveness_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get defense effectiveness matrix (defense x attack family)."""
        matrix = {}
        for defense in self.all():
            matrix[defense.id] = defense.effectiveness
        return matrix
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        all_defenses = self.all()
        return {
            "total_defenses": len(all_defenses),
            "by_category": {
                c.value: {
                    "name": c.name,
                    "count": len(self._by_category[c]),
                    "channels": c.target_channels,
                }
                for c in DefenseCategory
            },
            "by_strategy": {
                s.value: len(self.by_strategy(s))
                for s in DefenseStrategy
            },
            "avg_latency_ms": sum(d.latency_ms for d in all_defenses) / len(all_defenses),
            "avg_fpr": sum(d.false_positive_rate for d in all_defenses) / len(all_defenses),
        }


# =============================================================================
# Defense Base Class (for implementations)
# =============================================================================

class BaseDefense(ABC):
    """Abstract base class for defense implementations."""
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        self._decisions: List[DefenseResult] = []
    
    @abstractmethod
    def filter(self, content: str, channel: Channel) -> DefenseResult:
        """Apply defense filter to content."""
        pass
    
    def filter_batch(self, items: List[tuple]) -> List[DefenseResult]:
        """Filter multiple items."""
        return [self.filter(content, channel) for content, channel in items]
    
    def should_monitor(self, channel: Channel) -> bool:
        """Check if this channel should be monitored."""
        if not self.config.channels:
            return True
        return channel in self.config.channels
    
    @property
    def decision_log(self) -> List[DefenseResult]:
        return self._decisions.copy()


# Global registry instance
DEFENSES = DefenseRegistry()
