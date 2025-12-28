"""
AgentLeak Channels - The 7 distinct leakage channels (C1-C7).

From the paper:
    Agent execution creates seven distinct observable channels:
    - C1: Final outputs (user-visible responses)
    - C2: Inter-agent messages (coordination between agents)
    - C3: Tool inputs (arguments passed to external APIs)
    - C4: Tool outputs (data returned from APIs)
    - C5: Memory writes (persistent scratchpads, vector stores)
    - C6: Logs and telemetry (framework-level execution traces)
    - C7: Persisted artifacts (files, tickets, emails)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class Channel(str, Enum):
    """The 7 leakage channels defined in AgentLeak."""
    
    C1_FINAL_OUTPUT = "C1_final_output"
    C2_INTER_AGENT = "C2_inter_agent"
    C3_TOOL_INPUT = "C3_tool_input"
    C4_TOOL_OUTPUT = "C4_tool_output"
    C5_MEMORY_WRITE = "C5_memory_write"
    C6_LOG = "C6_log"
    C7_ARTIFACT = "C7_artifact"
    
    @property
    def short_name(self) -> str:
        """Short name for display (e.g., 'C1')."""
        return self.value.split("_")[0]
    
    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            "C1_final_output": "Final Output",
            "C2_inter_agent": "Inter-Agent Messages",
            "C3_tool_input": "Tool Inputs",
            "C4_tool_output": "Tool Outputs",
            "C5_memory_write": "Memory Writes",
            "C6_log": "Logs & Telemetry",
            "C7_artifact": "Persisted Artifacts",
        }
        return names.get(self.value, self.value)
    
    @property
    def weight(self) -> float:
        """Default sensitivity weight for this channel."""
        weights = {
            "C1_final_output": 1.0,    # Highest visibility
            "C2_inter_agent": 0.9,     # Often logged
            "C3_tool_input": 0.8,      # Logged by APIs
            "C4_tool_output": 0.6,     # Less controllable
            "C5_memory_write": 0.7,    # Persistent
            "C6_log": 0.5,             # Often overlooked
            "C7_artifact": 1.0,        # Difficult to retract
        }
        return weights.get(self.value, 1.0)
    
    @property
    def description(self) -> str:
        """Detailed description of the channel."""
        descriptions = {
            "C1_final_output": "User-visible responses returned by the agent",
            "C2_inter_agent": "Communication between coordinated agents, often persisted in logs",
            "C3_tool_input": "Arguments passed to external API calls, typically logged by service providers",
            "C4_tool_output": "Data returned from APIs, potentially containing sensitive payloads",
            "C5_memory_write": "Persistent agent scratchpads, vector stores, and intermediate notes",
            "C6_log": "Framework-level execution traces, sometimes transmitted to external services",
            "C7_artifact": "Generated files, tickets, emails, and other long-lived outputs",
        }
        return descriptions.get(self.value, "")
    
    @classmethod
    def all_channels(cls) -> list["Channel"]:
        """Return all channels in order."""
        return list(cls)


@dataclass
class ChannelEvent:
    """An event captured on a specific channel."""
    
    channel: Channel
    content: str
    timestamp: float
    agent_id: str = "primary"
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ChannelConfig:
    """Configuration for a specific channel's monitoring."""
    
    channel: Channel
    enabled: bool = True
    weight: float = None  # None = use default
    custom_detector: Optional[Callable] = None
    
    def __post_init__(self):
        if self.weight is None:
            self.weight = self.channel.weight


class ChannelManager:
    """
    Manages channel configuration and event collection.
    
    Provides a centralized way to:
    - Enable/disable specific channels
    - Configure channel weights
    - Collect events across all channels
    - Generate channel-specific reports
    
    Example:
        manager = ChannelManager()
        manager.configure_channel(Channel.C6_LOG, enabled=False)  # Ignore logs
        
        manager.record(Channel.C1_FINAL_OUTPUT, "Hello, John Doe")
        manager.record(Channel.C3_TOOL_INPUT, '{"patient_id": "12345"}')
        
        events = manager.get_events_by_channel()
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self._channel_configs: dict[Channel, ChannelConfig] = {}
        self._events: list[ChannelEvent] = []
        
        # Initialize default configs
        for channel in Channel:
            self._channel_configs[channel] = ChannelConfig(channel=channel)
    
    def configure_channel(
        self,
        channel: Channel,
        enabled: bool = None,
        weight: float = None,
        custom_detector: Callable = None,
    ) -> None:
        """Configure a specific channel."""
        cfg = self._channel_configs[channel]
        
        if enabled is not None:
            cfg.enabled = enabled
        if weight is not None:
            cfg.weight = weight
        if custom_detector is not None:
            cfg.custom_detector = custom_detector
    
    def is_enabled(self, channel: Channel) -> bool:
        """Check if a channel is enabled."""
        return self._channel_configs[channel].enabled
    
    def get_weight(self, channel: Channel) -> float:
        """Get the weight for a channel."""
        return self._channel_configs[channel].weight
    
    def record(
        self,
        channel: Channel,
        content: str,
        agent_id: str = "primary",
        metadata: dict = None,
        timestamp: float = None,
    ) -> Optional[ChannelEvent]:
        """Record an event on a channel."""
        if not self.is_enabled(channel):
            logger.debug(f"Channel {channel.short_name} is disabled, skipping event")
            return None
        
        import time
        event = ChannelEvent(
            channel=channel,
            content=content,
            timestamp=timestamp or time.time(),
            agent_id=agent_id,
            metadata=metadata,
        )
        self._events.append(event)
        return event
    
    def get_events(self, channel: Channel = None) -> list[ChannelEvent]:
        """Get events, optionally filtered by channel."""
        if channel is None:
            return list(self._events)
        return [e for e in self._events if e.channel == channel]
    
    def get_events_by_channel(self) -> dict[Channel, list[ChannelEvent]]:
        """Group events by channel."""
        result = {ch: [] for ch in Channel}
        for event in self._events:
            result[event.channel].append(event)
        return result
    
    def get_channel_summary(self) -> dict[str, Any]:
        """Get summary statistics per channel."""
        by_channel = self.get_events_by_channel()
        return {
            ch.short_name: {
                "name": ch.display_name,
                "enabled": self.is_enabled(ch),
                "weight": self.get_weight(ch),
                "event_count": len(events),
            }
            for ch, events in by_channel.items()
        }
    
    def clear(self) -> None:
        """Clear all recorded events."""
        self._events = []
    
    def reset(self) -> None:
        """Reset to default configuration and clear events."""
        self._events = []
        for channel in Channel:
            self._channel_configs[channel] = ChannelConfig(channel=channel)
