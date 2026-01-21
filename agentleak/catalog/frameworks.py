"""
AgentLeak Framework Catalog - Multi-agent framework adapters.

Supported frameworks from the paper:
    - LangChain (langchain.com)
    - CrewAI (crewai.io)
    - MetaGPT (github.com/geekan/MetaGPT)
    - AutoGPT (github.com/Significant-Gravitas/Auto-GPT)
    - Custom (user-defined frameworks)

Each adapter provides:
    - Trace collection from framework internals
    - Channel mapping (framework events → C1-C7)
    - Scenario execution
    - Defense integration points
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Type

from .channels import Channel


class FrameworkType(str, Enum):
    """Supported multi-agent frameworks."""
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    METAGPT = "metagpt"
    AUTOGPT = "autogpt"
    CUSTOM = "custom"

    @property
    def display_name(self) -> str:
        return {
            "langchain": "LangChain",
            "crewai": "CrewAI",
            "metagpt": "MetaGPT",
            "autogpt": "AutoGPT",
            "custom": "Custom Framework",
        }[self.value]


@dataclass
class FrameworkSpec:
    """Specification of a multi-agent framework."""
    id: str                          # e.g., "langchain"
    name: str                        # e.g., "LangChain"
    framework_type: FrameworkType
    description: str
    supported_channels: List[str]    # Which channels can be traced
    multiagent_support: bool         # Native multi-agent support
    memory_support: bool             # Native memory/persistence
    tool_support: bool               # Native tool calling
    version_tested: str              # Version used in paper experiments
    adapter_class: str               # Python class path for adapter
    install_command: str             # pip install command
    docs_url: str
    paper_results: Dict[str, float] = field(default_factory=dict)  # Attack success rates


# =============================================================================
# Framework Specifications
# =============================================================================

LANGCHAIN_SPEC = FrameworkSpec(
    id="langchain",
    name="LangChain",
    framework_type=FrameworkType.LANGCHAIN,
    description="Popular framework for building LLM applications with chains and agents",
    supported_channels=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    multiagent_support=True,
    memory_support=True,
    tool_support=True,
    version_tested="0.1.0",
    adapter_class="agentleak.harness.adapters.langchain_adapter.LangChainAdapter",
    install_command="pip install langchain langchain-openai",
    docs_url="https://python.langchain.com/docs/",
    paper_results={
        "F1": 0.622, "F2": 0.717, "F3": 0.627,
        "F4": 0.800, "F5": 0.627, "F6": 0.550,
    },
)

CREWAI_SPEC = FrameworkSpec(
    id="crewai",
    name="CrewAI",
    framework_type=FrameworkType.CREWAI,
    description="Framework for orchestrating role-playing autonomous AI agents",
    supported_channels=["C1", "C2", "C3", "C4", "C5"],
    multiagent_support=True,
    memory_support=True,
    tool_support=True,
    version_tested="0.28.0",
    adapter_class="agentleak.harness.adapters.crewai_adapter.CrewAIAdapter",
    install_command="pip install crewai",
    docs_url="https://docs.crewai.com/",
    paper_results={
        "F1": 0.650, "F2": 0.740, "F3": 0.610,
        "F4": 0.850, "F5": 0.600, "F6": 0.520,
    },
)

METAGPT_SPEC = FrameworkSpec(
    id="metagpt",
    name="MetaGPT",
    framework_type=FrameworkType.METAGPT,
    description="Multi-agent framework that assigns different roles to GPTs",
    supported_channels=["C1", "C2", "C3", "C4", "C5", "C7"],
    multiagent_support=True,
    memory_support=True,
    tool_support=True,
    version_tested="0.6.0",
    adapter_class="agentleak.harness.adapters.metagpt_adapter.MetaGPTAdapter",
    install_command="pip install metagpt",
    docs_url="https://docs.deepwisdom.ai/",
    paper_results={
        "F1": 0.610, "F2": 0.700, "F3": 0.650,
        "F4": 0.780, "F5": 0.590, "F6": 0.480,
    },
)

AUTOGPT_SPEC = FrameworkSpec(
    id="autogpt",
    name="AutoGPT",
    framework_type=FrameworkType.AUTOGPT,
    description="Autonomous AI agent for complex task completion",
    supported_channels=["C1", "C3", "C4", "C5", "C6", "C7"],
    multiagent_support=False,
    memory_support=True,
    tool_support=True,
    version_tested="0.5.0",
    adapter_class="agentleak.harness.adapters.autogpt_adapter.AutoGPTAdapter",
    install_command="pip install auto-gpt",
    docs_url="https://docs.agpt.co/",
    paper_results={
        "F1": 0.580, "F2": 0.680, "F3": 0.700,
        "F4": 0.450, "F5": 0.550, "F6": 0.500,
    },
)

CUSTOM_SPEC = FrameworkSpec(
    id="custom",
    name="Custom Framework",
    framework_type=FrameworkType.CUSTOM,
    description="User-defined custom framework with manual trace collection",
    supported_channels=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    multiagent_support=True,
    memory_support=True,
    tool_support=True,
    version_tested="N/A",
    adapter_class="agentleak.harness.adapters.base_adapter.CustomAdapter",
    install_command="",
    docs_url="",
    paper_results={},
)


# =============================================================================
# Framework Adapter Base Class
# =============================================================================

@dataclass
class TraceEvent:
    """A single trace event captured from framework execution."""
    timestamp: float
    channel: Channel
    event_type: str       # e.g., "llm_call", "tool_input", "agent_message"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class ExecutionTrace:
    """Complete trace of a scenario execution."""
    scenario_id: str
    framework: str
    events: List[TraceEvent] = field(default_factory=list)
    final_output: str = ""
    success: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def by_channel(self, channel: Channel) -> List[TraceEvent]:
        """Get events for a specific channel."""
        return [e for e in self.events if e.channel == channel]
    
    def content_by_channel(self, channel: Channel) -> List[str]:
        """Get content strings for a specific channel."""
        return [e.content for e in self.by_channel(channel)]


class FrameworkAdapter(ABC):
    """
    Abstract base class for framework adapters.
    
    Implementations must:
    1. Set up trace collection hooks
    2. Execute scenarios
    3. Map framework events to channels (C1-C7)
    4. Integrate defense mechanisms
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._trace_events: List[TraceEvent] = []
    
    @property
    @abstractmethod
    def framework_type(self) -> FrameworkType:
        """Return the framework type."""
        pass
    
    @property
    @abstractmethod
    def supported_channels(self) -> List[Channel]:
        """Return list of channels this adapter can trace."""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize the framework and set up trace hooks."""
        pass
    
    @abstractmethod
    def execute(self, scenario: Dict) -> ExecutionTrace:
        """
        Execute a scenario and collect traces.
        
        Args:
            scenario: Scenario dict with agents, tools, objective, etc.
            
        Returns:
            ExecutionTrace with all captured events
        """
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources after execution."""
        pass
    
    def add_trace_event(
        self,
        channel: Channel,
        event_type: str,
        content: str,
        **kwargs
    ) -> None:
        """Add a trace event during execution."""
        import time
        event = TraceEvent(
            timestamp=time.time(),
            channel=channel,
            event_type=event_type,
            content=content,
            **kwargs
        )
        self._trace_events.append(event)
    
    def clear_trace(self) -> None:
        """Clear collected trace events."""
        self._trace_events.clear()
    
    def get_trace(self) -> List[TraceEvent]:
        """Get collected trace events."""
        return self._trace_events.copy()


# =============================================================================
# Framework Registry
# =============================================================================

class FrameworkRegistry:
    """Central registry of all framework adapters."""
    
    def __init__(self):
        self._frameworks: Dict[str, FrameworkSpec] = {
            "langchain": LANGCHAIN_SPEC,
            "crewai": CREWAI_SPEC,
            "metagpt": METAGPT_SPEC,
            "autogpt": AUTOGPT_SPEC,
            "custom": CUSTOM_SPEC,
        }
        self._adapters: Dict[str, Type[FrameworkAdapter]] = {}
    
    def get(self, framework_id: str) -> Optional[FrameworkSpec]:
        """Get framework spec by ID."""
        return self._frameworks.get(framework_id.lower())
    
    def all(self) -> List[FrameworkSpec]:
        """Get all framework specs."""
        return list(self._frameworks.values())
    
    def multiagent_frameworks(self) -> List[FrameworkSpec]:
        """Get frameworks with native multi-agent support."""
        return [f for f in self.all() if f.multiagent_support]
    
    def register_adapter(self, framework_id: str, adapter_class: Type[FrameworkAdapter]) -> None:
        """Register a custom adapter class."""
        self._adapters[framework_id.lower()] = adapter_class
    
    def create_adapter(self, framework_id: str, config: Optional[Dict] = None) -> Optional[FrameworkAdapter]:
        """Create an adapter instance for a framework."""
        framework_id = framework_id.lower()
        
        # Check for registered custom adapter
        if framework_id in self._adapters:
            return self._adapters[framework_id](config)
        
        # Try to import default adapter
        spec = self.get(framework_id)
        if spec and spec.adapter_class:
            try:
                module_path, class_name = spec.adapter_class.rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_path)
                adapter_class = getattr(module, class_name)
                return adapter_class(config)
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load adapter for {framework_id}: {e}")
                return None
        
        return None
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        all_frameworks = self.all()
        return {
            "total_frameworks": len(all_frameworks),
            "multiagent_native": len(self.multiagent_frameworks()),
            "frameworks": {
                f.id: {
                    "name": f.name,
                    "multiagent": f.multiagent_support,
                    "channels": f.supported_channels,
                    "version": f.version_tested,
                }
                for f in all_frameworks
            },
        }


# Global registry instance
FRAMEWORKS = FrameworkRegistry()
