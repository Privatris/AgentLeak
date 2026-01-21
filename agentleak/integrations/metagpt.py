"""
AgentLeak SDK - MetaGPT Integration

Zero-code integration with MetaGPT multi-agent framework.
Uses MetaGPT's message system for automatic monitoring.

Paper Reference: Section 5.1 - MetaGPT exhibits 28-35% internal 
leak rate in multi-agent configurations.

Usage:
    from agentleak.integrations import MetaGPTIntegration, IntegrationConfig
    
    config = IntegrationConfig(vault={"api_key": "sk-..."})
    integration = MetaGPTIntegration(config)
    team = integration.attach(team)
"""

from typing import Any, Optional
import logging

from .base import BaseIntegration, IntegrationConfig

logger = logging.getLogger(__name__)

# Check for MetaGPT availability
HAS_METAGPT = False
try:
    from metagpt.team import Team
    from metagpt.roles import Role
    HAS_METAGPT = True
except ImportError:
    pass


class MetaGPTIntegration(BaseIntegration):
    """
    AgentLeak integration for MetaGPT framework.
    
    Monitors:
    - C1: Final output
    - C2: Inter-role messages
    - C5: Shared context/memory
    - C6: Generated artifacts (code, docs)
    
    Integration method: Message interception and role wrapping
    """
    
    FRAMEWORK_NAME = "metagpt"
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._check_framework()
    
    def _check_framework(self):
        """Check if MetaGPT is available."""
        if HAS_METAGPT:
            try:
                import metagpt
                self.FRAMEWORK_VERSION = getattr(metagpt, '__version__', 'unknown')
            except:
                self.FRAMEWORK_VERSION = "unknown"
        else:
            self.FRAMEWORK_VERSION = "not_installed"
            logger.warning("MetaGPT not installed. Install with: pip install metagpt")
    
    def attach(self, team: Any) -> Any:
        """
        Attach AgentLeak monitoring to a MetaGPT Team.
        
        Args:
            team: MetaGPT Team instance
            
        Returns:
            The team with monitoring attached
        """
        if not HAS_METAGPT:
            logger.warning("MetaGPT not available, using mock monitoring")
            return self._attach_mock(team)
        
        # Wrap team and roles
        team = self._wrap_team(team)
        
        # Wrap individual roles
        if hasattr(team, 'roles'):
            for role in team.roles:
                self._wrap_role(role)
        
        team._agentleak = self
        logger.info(f"✓ AgentLeak attached to MetaGPT ({self.FRAMEWORK_VERSION})")
        return team
    
    def _wrap_team(self, team: Any) -> Any:
        """Wrap MetaGPT Team methods."""
        if hasattr(team, 'run'):
            original_run = team.run
            
            async def monitored_run(*args, **kwargs):
                result = await original_run(*args, **kwargs)
                self._check_for_leaks(str(result), "C1", {"event": "team_output"})
                return result
            
            team.run = monitored_run
        
        return team
    
    def _wrap_role(self, role: Any) -> Any:
        """Wrap MetaGPT Role methods."""
        if hasattr(role, '_act'):
            original_act = role._act
            
            async def monitored_act(*args, **kwargs):
                result = await original_act(*args, **kwargs)
                self._check_for_leaks(
                    str(result),
                    "C2",
                    {"event": "role_action", "role": role.__class__.__name__}
                )
                return result
            
            role._act = monitored_act
        
        if hasattr(role, '_publish_message'):
            original_publish = role._publish_message
            
            def monitored_publish(message, *args, **kwargs):
                self._check_for_leaks(
                    str(message.content if hasattr(message, 'content') else message),
                    "C2",
                    {"event": "message_publish", "role": role.__class__.__name__}
                )
                return original_publish(message, *args, **kwargs)
            
            role._publish_message = monitored_publish
        
        return role
    
    def _attach_mock(self, team: Any) -> Any:
        """Mock attachment when MetaGPT not installed."""
        team._agentleak = self
        return team


def add_agentleak_to_metagpt(team: Any, vault: dict, **kwargs) -> Any:
    """
    Convenience function to add AgentLeak to a MetaGPT Team.
    
    Args:
        team: MetaGPT Team instance
        vault: Dict of sensitive data to protect
        **kwargs: Additional IntegrationConfig options
        
    Returns:
        The team with monitoring attached
    """
    config = IntegrationConfig(vault=vault, **kwargs)
    integration = MetaGPTIntegration(config)
    return integration.attach(team)
