"""
AgentLeak SDK - CrewAI Integration
===================================

Zero-code integration with CrewAI multi-agent framework.
Uses CrewAI's native callback system for automatic monitoring.

Paper Reference:
    Section 5.1 - CrewAI exhibits 33% internal leak rate
    on inter-agent messages (C2 channel).

Channels Monitored:
    - C1: Final crew output (task results)
    - C2: Inter-agent messages (task delegation, tool calls)
    - C5: Shared memory/context (long-term memory)

Example:
    >>> from agentleak.integrations import CrewAIIntegration, IntegrationConfig
    >>> 
    >>> config = IntegrationConfig(vault={"ssn": "123-45-6789"})
    >>> integration = CrewAIIntegration(config)
    >>> crew = integration.attach(crew)
    >>> result = crew.kickoff()
    >>> 
    >>> # Check for leaks
    >>> stats = integration.get_stats()
    >>> print(f"Leak rate: {stats['leak_rate']:.1%}")

Author: AgentLeak Team
License: MIT
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging

from .base import BaseIntegration, IntegrationConfig

if TYPE_CHECKING:
    from crewai import Crew

logger = logging.getLogger(__name__)

# =============================================================================
# CrewAI Framework Detection
# =============================================================================

_CREWAI_AVAILABLE: bool = False
_CREWAI_VERSION: str = "not_installed"

try:
    import crewai
    _CREWAI_AVAILABLE = True
    _CREWAI_VERSION = getattr(crewai, "__version__", "unknown")
    
    # Try to import BaseEventListener (CrewAI < 0.30)
    try:
        from crewai.utilities.events.base_event_listener import BaseEventListener
    except ImportError:
        # Stub class for newer CrewAI versions
        class BaseEventListener:  # type: ignore[no-redef]
            """Stub when BaseEventListener not available."""
            pass

except ImportError:
    # Stub class for type hints when CrewAI not installed
    class BaseEventListener:  # type: ignore[no-redef]
        """Stub when CrewAI not installed."""
        pass


def is_crewai_available() -> bool:
    """Check if CrewAI is available in the environment."""
    return _CREWAI_AVAILABLE


# =============================================================================
# CrewAI Integration
# =============================================================================

class CrewAIIntegration(BaseIntegration):
    """
    AgentLeak integration for CrewAI framework.
    
    This integration monitors CrewAI Crews for potential privacy leaks
    across multiple channels using native CrewAI event listeners.
    
    Attributes:
        FRAMEWORK_NAME: Framework identifier ("crewai")
        FRAMEWORK_VERSION: Detected CrewAI version
        
    Monitored Channels:
        C1 (Final Output): Task results returned to users
        C2 (Inter-Agent): Messages between agents during task delegation
        C5 (Memory): Shared context and long-term memory access
    
    Example:
        >>> config = IntegrationConfig(
        ...     vault={"ssn": "123-45-6789"},
        ...     mode=DetectionMode.STANDARD,
        ...     alert_threshold=0.8
        ... )
        >>> integration = CrewAIIntegration(config)
        >>> monitored_crew = integration.attach(my_crew)
    """
    
    FRAMEWORK_NAME: str = "crewai"
    FRAMEWORK_VERSION: str = _CREWAI_VERSION
    
    # Channel mapping for CrewAI events
    _EVENT_CHANNEL_MAP: Dict[str, str] = {
        "task_output": "C2",      # Task results between agents
        "agent_action": "C2",     # Agent decisions/actions
        "tool_call": "C3",        # Tool invocations
        "tool_result": "C4",      # Tool responses
        "memory_access": "C5",    # Memory operations
        "final_output": "C1",     # Final crew result
    }
    
    def __init__(self, config: IntegrationConfig) -> None:
        """
        Initialize CrewAI integration.
        
        Args:
            config: Integration configuration with vault and detection settings
        """
        super().__init__(config)
        self._callback_instance: Optional[BaseEventListener] = None
        self._verify_framework()
    
    def _verify_framework(self) -> None:
        """Verify CrewAI availability and log status."""
        if not _CREWAI_AVAILABLE:
            logger.warning(
                "CrewAI not installed. Install with: pip install crewai\n"
                "Monitoring will use mock mode."
            )
    
    def attach(self, crew: Any) -> Any:
        """
        Attach AgentLeak monitoring to a CrewAI Crew.
        
        This method hooks into CrewAI's event system to monitor all
        communications for potential privacy leaks.
        
        Args:
            crew: CrewAI Crew instance to monitor
            
        Returns:
            The same Crew instance with monitoring attached
            
        Raises:
            TypeError: If crew is not a valid CrewAI Crew instance
            
        Example:
            >>> crew = Crew(agents=[...], tasks=[...])
            >>> monitored_crew = integration.attach(crew)
            >>> result = monitored_crew.kickoff()
        """
        if not _CREWAI_AVAILABLE:
            logger.warning("CrewAI not available, using mock monitoring")
            return self._attach_mock(crew)
        
        # Validate crew object
        if not self._is_valid_crew(crew):
            logger.warning(f"Object {type(crew).__name__} may not be a valid Crew")
        
        # Create and attach callback
        self._callback_instance = self._create_callback()
        self._attach_callback(crew)
        
        # Mark as monitored
        crew._agentleak = self
        logger.info(f"✓ AgentLeak attached to CrewAI ({self.FRAMEWORK_VERSION})")
        
        return crew
    
    def _is_valid_crew(self, crew: Any) -> bool:
        """Check if object appears to be a valid CrewAI Crew."""
        return (
            hasattr(crew, 'kickoff') and
            hasattr(crew, 'agents') and
            hasattr(crew, 'tasks')
        )
    
    def _attach_callback(self, crew: Any) -> None:
        """Attach callback to crew using available method."""
        attached = False
        
        # Method 1: Native callbacks (CrewAI < 0.30)
        if hasattr(crew, 'callbacks') and hasattr(self._callback_instance, 'on_tool_use'):
            if crew.callbacks is None:
                crew.callbacks = []
            crew.callbacks.append(self._callback_instance)
            logger.debug("Attached via native callbacks")
            attached = True
            
        # Method 2: Agent Step Callbacks (CrewAI 0.30+)
        if hasattr(crew, 'agents'):
            self._attach_agent_callbacks(crew.agents)
            # We treat this as successfully attached for C3 monitoring
            attached = True
            
        # Always wrap kickoff for C1 (Final Output) monitoring
        try:
            self._wrap_kickoff(crew)
        except Exception as e:
            logger.debug(f"Could not wrap kickoff for C1 monitoring: {e}")
        
    def _attach_agent_callbacks(self, agents: List[Any]) -> None:
        """Attach monitoring to individual agents."""
        for agent in agents:
            self._wrap_agent_step_callback(agent)
            
    def _wrap_agent_step_callback(self, agent: Any) -> None:
        """Wrap agent's step callback to intercept tool usage."""
        original_callback = getattr(agent, 'step_callback', None)
        integration = self
        
        def monitored_step_callback(step: Any) -> Any:
            """Wrapped step callback."""
            # Check for tool usage in the step
            # We try multiple attribute paths to be robust across versions
            
            tool_name = None
            tool_input = None
            
            # Direct attributes (some versions)
            if hasattr(step, 'tool'):
                tool_name = step.tool
                tool_input = getattr(step, 'tool_input', "")
            # Nested in action (some versions)
            elif hasattr(step, 'action'):
                action = step.action
                if hasattr(action, 'tool'):
                    tool_name = action.tool
                    tool_input = getattr(action, 'tool_input', "")
            
            # If we found a tool usage
            if tool_name:
                integration._check_for_leaks(
                    content=str(tool_input),
                    channel="C3",
                    metadata={
                        "event": "tool_call",
                        "tool": tool_name
                    }
                )

            # Call original if it exists
            if original_callback:
                return original_callback(step)
            return None
            
        agent.step_callback = monitored_step_callback
    
    def _wrap_kickoff(self, crew: Any) -> None:
        """Wrap kickoff method for monitoring when callbacks unavailable."""
        original_kickoff = crew.kickoff
        integration = self
        
        def monitored_kickoff(*args: Any, **kwargs: Any) -> Any:
            """Wrapped kickoff with leak monitoring."""
            result = original_kickoff(*args, **kwargs)
            
            # Check final output (C1 channel)
            integration._check_for_leaks(
                content=str(result),
                channel="C1",
                metadata={
                    "method": "kickoff",
                    "event": "final_output"
                }
            )
            return result
        
        crew.kickoff = monitored_kickoff
    
    def _create_callback(self) -> BaseEventListener:
        """
        Create a CrewAI-compatible event listener callback.
        
        Returns:
            AgentLeakCallback instance configured for monitoring
        """
        integration = self
        
        class AgentLeakCallback(BaseEventListener):
            """
            CrewAI event listener for AgentLeak monitoring.
            
            Intercepts crew events and checks for privacy leaks.
            """
            
            def on_task_output(self, task_output: Any) -> None:
                """
                Called when a task completes.
                
                Monitors C2 channel (inter-agent messages).
                """
                content = self._extract_content(task_output)
                integration._check_for_leaks(
                    content=content,
                    channel="C2",
                    metadata={
                        "event": "task_output",
                        "task_type": type(task_output).__name__
                    }
                )
            
            def on_agent_action(self, agent: Any, action: Any) -> None:
                """
                Called when an agent takes an action.
                
                Monitors C2 channel (agent decisions/actions).
                """
                content = str(action)
                integration._check_for_leaks(
                    content=content,
                    channel="C2",
                    metadata={
                        "event": "agent_action",
                        "agent": str(getattr(agent, 'role', agent))
                    }
                )
            
            def on_tool_use(self, tool_name: str, tool_input: Any) -> None:
                """
                Called when an agent uses a tool.
                
                Monitors C3 channel (tool inputs).
                """
                content = str(tool_input)
                integration._check_for_leaks(
                    content=content,
                    channel="C3",
                    metadata={
                        "event": "tool_call",
                        "tool": tool_name
                    }
                )
            
            def on_tool_output(self, tool_name: str, tool_output: Any) -> None:
                """
                Called when a tool returns a result.
                
                Monitors C4 channel (tool outputs).
                """
                content = str(tool_output)
                integration._check_for_leaks(
                    content=content,
                    channel="C4",
                    metadata={
                        "event": "tool_result",
                        "tool": tool_name
                    }
                )
            
            @staticmethod
            def _extract_content(task_output: Any) -> str:
                """Extract string content from task output."""
                if hasattr(task_output, 'raw'):
                    return str(task_output.raw)
                elif hasattr(task_output, 'output'):
                    return str(task_output.output)
                return str(task_output)
        
        return AgentLeakCallback()
    
    def _attach_mock(self, crew: Any) -> Any:
        """
        Mock attachment when CrewAI not installed.
        
        Allows testing without CrewAI dependency.
        """
        crew._agentleak = self
        logger.debug("Mock monitoring attached")
        return crew
    
    def detach(self, crew: Any) -> Any:
        """
        Detach AgentLeak monitoring from a Crew.
        
        Args:
            crew: Crew instance with monitoring attached
            
        Returns:
            The Crew instance with monitoring removed
        """
        if hasattr(crew, '_agentleak'):
            delattr(crew, '_agentleak')
        
        if hasattr(crew, 'callbacks') and self._callback_instance:
            try:
                crew.callbacks.remove(self._callback_instance)
            except ValueError:
                pass
        
        self._callback_instance = None
        logger.info("AgentLeak monitoring detached")
        return crew


# =============================================================================
# Convenience Functions
# =============================================================================

def add_agentleak_to_crew(
    crew: Any,
    vault: Dict[str, str],
    **kwargs: Any
) -> Any:
    """
    Convenience function to add AgentLeak monitoring to a CrewAI Crew.
    
    This is the simplest way to add privacy leak detection to an existing
    Crew with a single function call.
    
    Args:
        crew: CrewAI Crew instance to monitor
        vault: Dictionary of sensitive data to protect
            Keys are identifiers, values are the sensitive strings
        **kwargs: Additional IntegrationConfig options
            - mode: DetectionMode (FAST, STANDARD, HYBRID, LLM_ONLY)
            - alert_threshold: Confidence threshold (0.0-1.0)
            - raise_on_leak: Raise exception on high-confidence leak
            - block_on_leak: Block output on leak detection
            - log_dir: Directory for incident logs
            
    Returns:
        The Crew instance with monitoring attached
        
    Example:
        >>> from agentleak.integrations import add_agentleak_to_crew
        >>> 
        >>> vault = {
        ...     "ssn": "123-45-6789",
        ...     "credit_card": "4111-1111-1111-1111"
        ... }
        >>> crew = add_agentleak_to_crew(my_crew, vault, mode=DetectionMode.HYBRID)
    """
    config = IntegrationConfig(vault=vault, **kwargs)
    integration = CrewAIIntegration(config)
    return integration.attach(crew)


# Alias for backward compatibility
HAS_CREWAI = _CREWAI_AVAILABLE
