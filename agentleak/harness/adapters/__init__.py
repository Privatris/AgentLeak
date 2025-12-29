"""
agentleak Framework Adapters - Integrations with popular agent frameworks.

This module provides adapters for various agent frameworks, allowing agentleak
to capture execution traces from different agent architectures.

Supported frameworks:
- LangChain: Popular framework for LLM application development
- CrewAI: Multi-agent collaborative framework
- AutoGPT: Autonomous goal-driven agent
- MetaGPT: Multi-role software company simulation
- AgentGPT: Web-based autonomous agent platform

Each adapter implements the BaseAdapter interface:
- _setup_agent(): Configure the agent for a scenario
- _run_agent(): Execute and capture trace events

Usage:
    from .harness.adapters import LangChainAdapter, LangChainConfig

    config = LangChainConfig(model_name="gpt-4")
    adapter = LangChainAdapter(config)
    result = adapter.run(scenario)

    # Analyze leakage
    from .detection import DetectionPipeline
    pipeline = DetectionPipeline()
    report = pipeline.analyze(scenario, result)
    print(f"ELR: {report.elr:.2%}")
"""

from .langchain_adapter import LangChainAdapter, LangChainConfig
from .crewai_adapter import CrewAIAdapter, CrewAIConfig
from .autogpt_adapter import AutoGPTAdapter, AutoGPTConfig
from .metagpt_adapter import MetaGPTAdapter, MetaGPTConfig
from .agentgpt_adapter import AgentGPTAdapter, AgentGPTConfig

# Registry of all adapters
ADAPTERS = {
    "langchain": LangChainAdapter,
    "crewai": CrewAIAdapter,
    "autogpt": AutoGPTAdapter,
    "metagpt": MetaGPTAdapter,
    "agentgpt": AgentGPTAdapter,
}

# Registry of all configs
ADAPTER_CONFIGS = {
    "langchain": LangChainConfig,
    "crewai": CrewAIConfig,
    "autogpt": AutoGPTConfig,
    "metagpt": MetaGPTConfig,
    "agentgpt": AgentGPTConfig,
}


def get_adapter(framework: str):
    """Get adapter class for framework."""
    return ADAPTERS.get(framework)


def list_adapters():
    """List supported frameworks."""
    return list(ADAPTERS.keys())


def get_adapter(name: str, config=None):
    """
    Get an adapter by name.

    Args:
        name: Adapter name (langchain, crewai, autogpt, metagpt, agentgpt)
        config: Optional adapter configuration

    Returns:
        Configured adapter instance

    Example:
        adapter = get_adapter("langchain", LangChainConfig(model_name="gpt-4"))
    """
    if name.lower() not in ADAPTERS:
        available = ", ".join(ADAPTERS.keys())
        raise ValueError(f"Unknown adapter: {name}. Available: {available}")

    adapter_cls = ADAPTERS[name.lower()]
    return adapter_cls(config)


def list_adapters():
    """Return list of available adapter names."""
    return list(ADAPTERS.keys())


__all__ = [
    # Adapters
    "LangChainAdapter",
    "LangChainConfig",
    "CrewAIAdapter",
    "CrewAIConfig",
    "AutoGPTAdapter",
    "AutoGPTConfig",
    "MetaGPTAdapter",
    "MetaGPTConfig",
    "AgentGPTAdapter",
    "AgentGPTConfig",
    # Registries
    "ADAPTERS",
    "ADAPTER_CONFIGS",
    # Functions
    "get_adapter",
    "list_adapters",
]
