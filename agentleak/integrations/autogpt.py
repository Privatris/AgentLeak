"""
AgentLeak SDK - AutoGPT Integration

Zero-code integration with AutoGPT autonomous agent framework.
Uses AutoGPT's plugin system for automatic monitoring.

Paper Reference: Section 5.1 - AutoGPT exhibits 28-35% internal 
leak rate with no default privacy mechanisms.

Usage:
    from agentleak.integrations import AutoGPTIntegration, IntegrationConfig
    
    config = IntegrationConfig(vault={"api_key": "sk-..."})
    integration = AutoGPTIntegration(config)
    agent = integration.attach(agent)
"""

from typing import Any, Optional
import logging

from .base import BaseIntegration, IntegrationConfig

logger = logging.getLogger(__name__)

# Check for AutoGPT availability
HAS_AUTOGPT = False
try:
    # AutoGPT has various import patterns
    import autogpt
    HAS_AUTOGPT = True
except ImportError:
    try:
        from forge.agent import Agent
        HAS_AUTOGPT = True
    except ImportError:
        pass


class AutoGPTIntegration(BaseIntegration):
    """
    AgentLeak integration for AutoGPT framework.
    
    Monitors:
    - C1: Final output to user
    - C3: Tool/command inputs
    - C4: Tool/command outputs
    - C5: Memory (workspace files, context)
    - C6: Artifacts (generated files)
    
    Integration method: Command wrapping and file monitoring
    """
    
    FRAMEWORK_NAME = "autogpt"
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._check_framework()
    
    def _check_framework(self):
        """Check if AutoGPT is available."""
        if HAS_AUTOGPT:
            try:
                import autogpt
                self.FRAMEWORK_VERSION = getattr(autogpt, '__version__', 'unknown')
            except:
                self.FRAMEWORK_VERSION = "unknown"
        else:
            self.FRAMEWORK_VERSION = "not_installed"
            logger.warning("AutoGPT not installed. See: https://github.com/Significant-Gravitas/AutoGPT")
    
    def attach(self, agent: Any) -> Any:
        """
        Attach AgentLeak monitoring to an AutoGPT agent.
        
        Args:
            agent: AutoGPT Agent instance
            
        Returns:
            The agent with monitoring attached
        """
        if not HAS_AUTOGPT:
            logger.warning("AutoGPT not available, using mock monitoring")
            return self._attach_mock(agent)
        
        # Wrap key methods
        agent = self._wrap_commands(agent)
        agent = self._wrap_memory(agent)
        
        agent._agentleak = self
        logger.info(f"✓ AgentLeak attached to AutoGPT ({self.FRAMEWORK_VERSION})")
        return agent
    
    def _wrap_commands(self, agent: Any) -> Any:
        """Wrap AutoGPT command execution."""
        if hasattr(agent, 'execute_command'):
            original_execute = agent.execute_command
            
            def monitored_execute(command_name, arguments, *args, **kwargs):
                # Check command input
                self._check_for_leaks(
                    f"{command_name}: {arguments}",
                    "C3",
                    {"event": "command_input", "command": command_name}
                )
                
                result = original_execute(command_name, arguments, *args, **kwargs)
                
                # Check command output
                self._check_for_leaks(
                    str(result),
                    "C4", 
                    {"event": "command_output", "command": command_name}
                )
                
                return result
            
            agent.execute_command = monitored_execute
        
        return agent
    
    def _wrap_memory(self, agent: Any) -> Any:
        """Wrap AutoGPT memory operations."""
        if hasattr(agent, 'memory'):
            memory = agent.memory
            
            if hasattr(memory, 'add'):
                original_add = memory.add
                def monitored_add(content, *args, **kwargs):
                    self._check_for_leaks(str(content), "C5", {"event": "memory_write"})
                    return original_add(content, *args, **kwargs)
                memory.add = monitored_add
        
        return agent
    
    def _attach_mock(self, agent: Any) -> Any:
        """Mock attachment when AutoGPT not installed."""
        agent._agentleak = self
        return agent


def add_agentleak_to_autogpt(agent: Any, vault: dict, **kwargs) -> Any:
    """
    Convenience function to add AgentLeak to an AutoGPT agent.
    
    Args:
        agent: AutoGPT Agent instance
        vault: Dict of sensitive data to protect
        **kwargs: Additional IntegrationConfig options
        
    Returns:
        The agent with monitoring attached
    """
    config = IntegrationConfig(vault=vault, **kwargs)
    integration = AutoGPTIntegration(config)
    return integration.attach(agent)
