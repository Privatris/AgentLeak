"""
AgentLeak SDK - CrewAI Integration

Zero-code integration with CrewAI multi-agent framework.
Uses CrewAI's native callback system for automatic monitoring.

Paper Reference: Section 5.1 - CrewAI exhibits 33% internal leak rate
on inter-agent messages (C2 channel).

Usage:
    from agentleak.integrations import CrewAIIntegration, IntegrationConfig
    
    config = IntegrationConfig(vault={"ssn": "123-45-6789"})
    integration = CrewAIIntegration(config)
    crew = integration.attach(crew)
    result = crew.kickoff()
"""

from typing import Any, Optional
import logging

from .base import BaseIntegration, IntegrationConfig

logger = logging.getLogger(__name__)

# Check for CrewAI availability
try:
    from crewai.utilities.events.base_event_listener import BaseEventListener
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    class BaseEventListener:
        """Stub when CrewAI not installed."""
        pass


class CrewAIIntegration(BaseIntegration):
    """
    AgentLeak integration for CrewAI framework.
    
    Monitors:
    - C1: Final crew output
    - C2: Inter-agent messages (task delegation)
    - C5: Shared memory/context
    
    Integration method: Native CrewAI event listeners
    """
    
    FRAMEWORK_NAME = "crewai"
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._check_framework()
    
    def _check_framework(self):
        """Check if CrewAI is available."""
        if HAS_CREWAI:
            try:
                import crewai
                self.FRAMEWORK_VERSION = crewai.__version__
            except:
                self.FRAMEWORK_VERSION = "unknown"
        else:
            self.FRAMEWORK_VERSION = "not_installed"
            logger.warning("CrewAI not installed. Install with: pip install crewai")
    
    def attach(self, crew: Any) -> Any:
        """
        Attach AgentLeak monitoring to a CrewAI Crew.
        
        Args:
            crew: CrewAI Crew instance
            
        Returns:
            The same Crew with monitoring attached
        """
        if not HAS_CREWAI:
            logger.warning("CrewAI not available, using mock monitoring")
            return self._attach_mock(crew)
        
        # Create callback and attach
        callback = self._create_callback()
        
        if hasattr(crew, 'callbacks'):
            if crew.callbacks is None:
                crew.callbacks = []
            crew.callbacks.append(callback)
        else:
            # Fallback: wrap kickoff
            original_kickoff = crew.kickoff
            def monitored_kickoff(*args, **kwargs):
                result = original_kickoff(*args, **kwargs)
                self._check_for_leaks(str(result), "C1", {"method": "kickoff"})
                return result
            crew.kickoff = monitored_kickoff
        
        crew._agentleak = self
        logger.info(f"✓ AgentLeak attached to CrewAI ({self.FRAMEWORK_VERSION})")
        return crew
    
    def _create_callback(self):
        """Create a CrewAI-compatible callback."""
        integration = self
        
        class AgentLeakCallback(BaseEventListener):
            def on_task_output(self, task_output):
                """Called when a task completes."""
                content = str(task_output.raw if hasattr(task_output, 'raw') else task_output)
                integration._check_for_leaks(content, "C2", {"event": "task_output"})
            
            def on_agent_action(self, agent, action):
                """Called when agent takes an action."""
                content = str(action)
                integration._check_for_leaks(content, "C2", {"event": "agent_action", "agent": str(agent)})
        
        return AgentLeakCallback()
    
    def _attach_mock(self, crew: Any) -> Any:
        """Mock attachment when CrewAI not installed."""
        crew._agentleak = self
        return crew


def add_agentleak_to_crew(crew: Any, vault: dict, **kwargs) -> Any:
    """
    Convenience function to add AgentLeak to a CrewAI Crew.
    
    Args:
        crew: CrewAI Crew instance
        vault: Dict of sensitive data to protect
        **kwargs: Additional IntegrationConfig options
        
    Returns:
        The Crew with monitoring attached
    """
    config = IntegrationConfig(vault=vault, **kwargs)
    integration = CrewAIIntegration(config)
    return integration.attach(crew)
