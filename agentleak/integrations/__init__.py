"""
AgentLeak SDK - Framework Integrations

Provides zero-code integration with major agentic AI frameworks.

Paper Reference: Section 5.1 validates that all major frameworks 
(CrewAI, LangChain, AutoGPT, MetaGPT) exhibit 28-35% internal leakage.

Supported Frameworks:
- CrewAI: Multi-agent role-based orchestration
- LangChain: Agent and chain orchestration  
- AutoGPT: Autonomous agent framework
- MetaGPT: Multi-agent software development

Quick Start:
    from agentleak.integrations import add_agentleak, IntegrationConfig
    
    # Define sensitive data to protect
    vault = {
        "ssn": "123-45-6789",
        "api_key": "sk-...",
        "password": "secret123"
    }
    
    # Add monitoring to any framework
    crew = add_agentleak(crew, vault)  # Auto-detects framework
"""

# Base classes and types
from .base import (
    BaseIntegration,
    IntegrationConfig,
    DetectionMode,
    LeakIncident,
    IntegrationStats,
    SecurityError,
)

# Framework integrations
from .crewai import CrewAIIntegration, add_agentleak_to_crew
from .langchain import LangChainIntegration, add_agentleak_to_langchain
from .autogpt import AutoGPTIntegration, add_agentleak_to_autogpt
from .metagpt import MetaGPTIntegration, add_agentleak_to_metagpt

# Framework detection mapping
_FRAMEWORK_DETECTORS = {
    'Crew': ('crewai', CrewAIIntegration),
    'AgentExecutor': ('langchain', LangChainIntegration),
    'RunnableSequence': ('langchain', LangChainIntegration),
    'Agent': ('autogpt', AutoGPTIntegration),
    'Team': ('metagpt', MetaGPTIntegration),
}


def add_agentleak(framework_object, vault: dict, **kwargs):
    """
    Universal function to add AgentLeak monitoring.
    Auto-detects the framework and applies the appropriate integration.
    
    Args:
        framework_object: Any supported framework object
        vault: Dict of sensitive data to protect
        **kwargs: Additional IntegrationConfig options
        
    Returns:
        The framework object with monitoring attached
    """
    class_name = framework_object.__class__.__name__
    
    if class_name in _FRAMEWORK_DETECTORS:
        _, integration_class = _FRAMEWORK_DETECTORS[class_name]
        config = IntegrationConfig(vault=vault, **kwargs)
        return integration_class(config).attach(framework_object)
    
    module = framework_object.__class__.__module__
    if 'crewai' in module:
        return add_agentleak_to_crew(framework_object, vault, **kwargs)
    if 'langchain' in module:
        return add_agentleak_to_langchain(framework_object, vault, **kwargs)
    if 'autogpt' in module or 'forge' in module:
        return add_agentleak_to_autogpt(framework_object, vault, **kwargs)
    if 'metagpt' in module:
        return add_agentleak_to_metagpt(framework_object, vault, **kwargs)
    
    raise ValueError(f"Cannot detect framework for {class_name}")


def detect_framework(obj) -> str:
    """Detect which framework an object belongs to."""
    class_name = obj.__class__.__name__
    if class_name in _FRAMEWORK_DETECTORS:
        return _FRAMEWORK_DETECTORS[class_name][0]
    module = obj.__class__.__module__
    for fw in ['crewai', 'langchain', 'autogpt', 'forge', 'metagpt']:
        if fw in module:
            return 'autogpt' if fw == 'forge' else fw
    return 'unknown'


__all__ = [
    # Base
    'BaseIntegration',
    'IntegrationConfig', 
    'DetectionMode',
    'LeakIncident',
    'IntegrationStats',
    'SecurityError',
    # Integrations
    'CrewAIIntegration',
    'LangChainIntegration', 
    'AutoGPTIntegration',
    'MetaGPTIntegration',
    # Convenience functions
    'add_agentleak',
    'add_agentleak_to_crew',
    'add_agentleak_to_langchain',
    'add_agentleak_to_autogpt',
    'add_agentleak_to_metagpt',
    'detect_framework',
]