"""
AgentLeak SDK - LangChain Integration

Zero-code integration with LangChain framework.
Uses LangChain's callback system for automatic monitoring.

Paper Reference: Section 5.1 - LangChain evaluated with 28-35% 
internal leak rate on multi-agent configurations.

Usage:
    from agentleak.integrations import LangChainIntegration, IntegrationConfig
    
    config = IntegrationConfig(vault={"api_key": "sk-..."})
    integration = LangChainIntegration(config)
    agent = integration.attach(agent)
"""

from typing import Any, Optional
import logging

from .base import BaseIntegration, IntegrationConfig

logger = logging.getLogger(__name__)

# Check for LangChain availability
try:
    from langchain.callbacks.base import BaseCallbackHandler
    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False
        class BaseCallbackHandler:
            """Stub when LangChain not installed."""
            pass


class LangChainIntegration(BaseIntegration):
    """
    AgentLeak integration for LangChain framework.
    
    Monitors:
    - C1: Final agent output
    - C2: Inter-agent messages (in multi-agent setups)
    - C3: Tool inputs
    - C4: Tool outputs
    - C5: Memory operations
    
    Integration method: LangChain callback handlers
    """
    
    FRAMEWORK_NAME = "langchain"
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._check_framework()
    
    def _check_framework(self):
        """Check if LangChain is available."""
        if HAS_LANGCHAIN:
            try:
                import langchain
                self.FRAMEWORK_VERSION = langchain.__version__
            except:
                self.FRAMEWORK_VERSION = "unknown"
        else:
            self.FRAMEWORK_VERSION = "not_installed"
            logger.warning("LangChain not installed. Install with: pip install langchain")
    
    def attach(self, agent: Any) -> Any:
        """
        Attach AgentLeak monitoring to a LangChain agent/chain.
        
        Args:
            agent: LangChain agent, chain, or runnable
            
        Returns:
            The agent with monitoring attached
        """
        if not HAS_LANGCHAIN:
            logger.warning("LangChain not available, using mock monitoring")
            return self._attach_mock(agent)
        
        callback = self._create_callback()
        
        # Try different attachment methods
        if hasattr(agent, 'callbacks'):
            if agent.callbacks is None:
                agent.callbacks = []
            agent.callbacks.append(callback)
        elif hasattr(agent, 'with_config'):
            # LangChain Expression Language (LCEL)
            agent = agent.with_config(callbacks=[callback])
        else:
            # Wrap invoke/run methods
            agent = self._wrap_agent(agent, callback)
        
        agent._agentleak = self
        logger.info(f"✓ AgentLeak attached to LangChain ({self.FRAMEWORK_VERSION})")
        return agent
    
    def _create_callback(self):
        """Create a LangChain-compatible callback handler."""
        integration = self
        
        class AgentLeakHandler(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                """Called when LLM returns."""
                content = str(response.generations[0][0].text if response.generations else response)
                integration._check_for_leaks(content, "C1", {"event": "llm_end"})
            
            def on_tool_start(self, serialized, input_str, **kwargs):
                """Called before tool execution."""
                integration._check_for_leaks(str(input_str), "C3", {"event": "tool_input", "tool": serialized.get("name", "unknown")})
            
            def on_tool_end(self, output, **kwargs):
                """Called after tool execution."""
                integration._check_for_leaks(str(output), "C4", {"event": "tool_output"})
            
            def on_chain_end(self, outputs, **kwargs):
                """Called when chain completes."""
                integration._check_for_leaks(str(outputs), "C1", {"event": "chain_end"})
            
            def on_agent_action(self, action, **kwargs):
                """Called when agent takes action."""
                content = f"{action.tool}: {action.tool_input}"
                integration._check_for_leaks(content, "C2", {"event": "agent_action"})
        
        return AgentLeakHandler()
    
    def _wrap_agent(self, agent: Any, callback) -> Any:
        """Wrap agent methods when callbacks not supported."""
        if hasattr(agent, 'invoke'):
            original_invoke = agent.invoke
            def monitored_invoke(input, *args, **kwargs):
                kwargs.setdefault('callbacks', []).append(callback)
                return original_invoke(input, *args, **kwargs)
            agent.invoke = monitored_invoke
        
        if hasattr(agent, 'run'):
            original_run = agent.run
            def monitored_run(input, *args, **kwargs):
                kwargs.setdefault('callbacks', []).append(callback)
                return original_run(input, *args, **kwargs)
            agent.run = monitored_run
        
        return agent
    
    def _attach_mock(self, agent: Any) -> Any:
        """Mock attachment when LangChain not installed."""
        agent._agentleak = self
        return agent


def add_agentleak_to_langchain(agent: Any, vault: dict, **kwargs) -> Any:
    """
    Convenience function to add AgentLeak to a LangChain agent.
    
    Args:
        agent: LangChain agent, chain, or runnable
        vault: Dict of sensitive data to protect
        **kwargs: Additional IntegrationConfig options
        
    Returns:
        The agent with monitoring attached
    """
    config = IntegrationConfig(vault=vault, **kwargs)
    integration = LangChainIntegration(config)
    return integration.attach(agent)
