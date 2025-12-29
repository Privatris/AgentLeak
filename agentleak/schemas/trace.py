"""
AgentLeak Trace Schema - Data structures for execution traces.

The trace format captures every event during scenario execution,
enabling leakage detection across all 7 channels (C1-C7).
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

from .scenario import Channel, AttackClass


class EventType(str, Enum):
    """Types of events captured in traces."""

    MESSAGE_IN = "message_in"  # Input to agent
    MESSAGE_OUT = "message_out"  # Output from agent
    TOOL_CALL = "tool_call"  # Agent calls a tool
    TOOL_RESULT = "tool_result"  # Tool returns result
    MEMORY_WRITE = "memory_write"  # Agent writes to memory
    MEMORY_READ = "memory_read"  # Agent reads from memory
    LOG_EVENT = "log_event"  # Framework-level log
    ARTIFACT_WRITE = "artifact_write"  # Agent creates file/doc/email
    AGENT_MESSAGE = "agent_message"  # Inter-agent communication
    SYSTEM_EVENT = "system_event"  # Harness-level events


class TraceEvent(BaseModel):
    """
    A single event in an execution trace.

    Each event captures:
    - What happened (event_type)
    - Who did it (agent_id)
    - What was the content
    - Which channel it affects
    - Timing and context
    """

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sequence_num: int = Field(..., description="Event sequence number in trace")

    # Context
    scenario_id: str
    agent_id: str = Field(..., description="Agent that triggered event")
    role: Optional[str] = Field(None, description="Agent role")

    # Event details
    event_type: EventType
    channel: Channel = Field(..., description="Which leakage channel this affects")
    content: str = Field(..., description="The actual content/text")

    # Tool-specific (optional)
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_output: Optional[str] = None

    # Memory-specific (optional)
    memory_key: Optional[str] = None
    memory_value: Optional[Any] = None

    # Artifact-specific (optional)
    artifact_type: Optional[str] = None  # file, email, ticket, etc.
    artifact_path: Optional[str] = None

    # Policy context
    policy_context: Optional[PolicyContext] = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class PolicyContext(BaseModel):
    """Policy and attack context for an event."""

    allowed_set_hash: Optional[str] = None
    attack_class: Optional[AttackClass] = None
    attack_active: bool = False
    defense_enabled: bool = False
    defense_name: Optional[str] = None

    model_config = {"extra": "forbid"}


class TraceMetadata(BaseModel):
    """Metadata for a complete execution trace."""

    trace_id: str
    scenario_id: str
    framework: str = Field(..., description="Agent framework used (e.g., 'langchain')")
    model: str = Field(..., description="LLM model used (e.g., 'gpt-4')")
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    total_events: int = 0
    total_tokens: int = 0
    success: Optional[bool] = None
    error: Optional[str] = None

    model_config = {"extra": "forbid"}


class ExecutionTrace(BaseModel):
    """
    Complete execution trace for a scenario.

    This is what the harness produces and what the detection
    pipeline consumes.
    """

    metadata: TraceMetadata
    events: list[TraceEvent] = Field(default_factory=list)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def get_events_by_channel(self, channel: Channel) -> list[TraceEvent]:
        """Filter events by leakage channel."""
        return [e for e in self.events if e.channel == channel]

    def get_events_by_type(self, event_type: EventType) -> list[TraceEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_all_content(self) -> list[str]:
        """Get all content strings for leakage scanning."""
        contents = []
        for event in self.events:
            contents.append(event.content)
            if event.tool_args:
                contents.append(str(event.tool_args))
            if event.tool_output:
                contents.append(event.tool_output)
            if event.memory_value:
                contents.append(str(event.memory_value))
        return contents

    def get_channel_content(self, channel: Channel) -> list[str]:
        """Get all content for a specific channel."""
        return [e.content for e in self.events if e.channel == channel]

    def to_jsonl(self) -> str:
        """Export trace as JSONL format."""
        lines = [self.metadata.model_dump_json()]
        for event in self.events:
            lines.append(event.model_dump_json())
        return "\n".join(lines)

    model_config = {"extra": "forbid"}


if __name__ == "__main__":
    # Quick test
    from datetime import datetime

    trace = ExecutionTrace(
        metadata=TraceMetadata(
            trace_id="trace_001",
            scenario_id="agentleak_fin_00001",
            framework="langchain",
            model="gpt-4",
            started_at=datetime.utcnow(),
        ),
        events=[
            TraceEvent(
                sequence_num=1,
                scenario_id="agentleak_fin_00001",
                agent_id="A1",
                event_type=EventType.TOOL_CALL,
                channel=Channel.C3_TOOL_INPUT,
                content="Querying case database",
                tool_name="case_db",
                tool_args={"case_id": "T-1932"},
            ),
            TraceEvent(
                sequence_num=2,
                scenario_id="agentleak_fin_00001",
                agent_id="A1",
                event_type=EventType.MESSAGE_OUT,
                channel=Channel.C1_FINAL_OUTPUT,
                content="The transaction dispute has been resolved.",
            ),
        ],
    )

    print(f"✓ Created trace with {trace.event_count} events")
    print(f"  Framework: {trace.metadata.framework}")
    print(f"  C1 events: {len(trace.get_events_by_channel(Channel.C1_FINAL_OUTPUT))}")
    print(f"  C3 events: {len(trace.get_events_by_channel(Channel.C3_TOOL_INPUT))}")
