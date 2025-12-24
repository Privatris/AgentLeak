"""
APB Trace Collector - Unified trace collection across all 7 channels.

The TraceCollector provides a centralized way to capture events from
different sources (tool calls, memory writes, logs, etc.) and ensures
all events are properly sequenced and attributed to the correct channel.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from contextlib import contextmanager
import threading
import time
from datetime import datetime

from ..schemas.trace import TraceEvent, ExecutionTrace, EventType, TraceMetadata
from ..schemas.scenario import Channel


@dataclass
class TraceBuffer:
    """
    Thread-safe buffer for collecting trace events.
    
    Supports multiple concurrent writers (e.g., parallel tool calls)
    while maintaining global event ordering.
    """
    
    scenario_id: str
    events: list[TraceEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counter: int = field(default=0)
    _start_time: float = field(default_factory=time.time)
    
    def record(
        self,
        event_type: EventType,
        channel: Channel,
        content: str,
        metadata: Optional[dict] = None,
        agent_id: str = "primary",
    ) -> TraceEvent:
        """Record a new event in the buffer."""
        with self._lock:
            self._counter += 1
            event = TraceEvent(
                sequence_num=self._counter,
                scenario_id=self.scenario_id,
                agent_id=agent_id,
                event_type=event_type,
                channel=channel,
                content=content,
                metadata=metadata or {},
            )
            self.events.append(event)
            return event
    
    def to_trace(self, extra_metadata: Optional[dict] = None) -> ExecutionTrace:
        """Convert buffer to an ExecutionTrace."""
        import uuid
        
        now = datetime.utcnow()
        duration = time.time() - self._start_time
        
        # Build TraceMetadata
        framework = extra_metadata.get("framework", "unknown") if extra_metadata else "unknown"
        model = extra_metadata.get("model", "unknown") if extra_metadata else "unknown"
        
        metadata = TraceMetadata(
            trace_id=str(uuid.uuid4()),
            scenario_id=self.scenario_id,
            framework=framework,
            model=model,
            started_at=datetime.utcfromtimestamp(self._start_time),
            completed_at=now,
            duration_seconds=duration,
            total_events=len(self.events),
        )
        
        return ExecutionTrace(
            metadata=metadata,
            events=list(self.events),
        )
    
    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self.events = []
            self._counter = 0
            self._start_time = time.time()


class TraceCollector:
    """
    Centralized trace collection for APB scenarios.
    
    Provides hooks for capturing events from various sources:
    - Tool call interceptors
    - Memory write hooks  
    - Log handlers
    - Artifact watchers
    - Inter-agent message handlers
    
    Example:
        collector = TraceCollector("scenario_001")
        
        # Manual recording
        collector.record_tool_call("search_records", {"query": "patient"})
        collector.record_tool_result("search_records", '{"results": [...]}')
        
        # Hook-based recording
        with collector.capture_logs():
            agent.run(...)  # All logs captured
        
        trace = collector.finalize()
    """
    
    def __init__(self, scenario_id: str, verbose: bool = False):
        self.scenario_id = scenario_id
        self.verbose = verbose
        self._buffer = TraceBuffer(scenario_id=scenario_id)
        self._hooks: list[Callable] = []
    
    # ============================================================
    # Core Recording Methods
    # ============================================================
    
    def record(
        self,
        event_type: EventType,
        channel: Channel,
        content: str,
        metadata: Optional[dict] = None,
        agent_id: str = "primary",
    ) -> TraceEvent:
        """Record a generic event."""
        event = self._buffer.record(event_type, channel, content, metadata, agent_id)
        
        if self.verbose:
            self._print_event(event)
        
        return event
    
    def record_message_in(
        self,
        content: str,
        role: str = "user",
        channel: Channel = Channel.C1_FINAL_OUTPUT,
    ) -> TraceEvent:
        """Record an incoming message (user query, inter-agent)."""
        return self.record(
            EventType.MESSAGE_IN,
            channel,
            content,
            {"role": role}
        )
    
    def record_message_out(
        self,
        content: str,
        role: str = "assistant",
        channel: Channel = Channel.C1_FINAL_OUTPUT,
    ) -> TraceEvent:
        """Record an outgoing message (response, inter-agent)."""
        return self.record(
            EventType.MESSAGE_OUT,
            channel,
            content,
            {"role": role}
        )
    
    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        tool_call_id: Optional[str] = None,
    ) -> TraceEvent:
        """Record a tool invocation."""
        import json
        content = json.dumps({"tool": tool_name, "arguments": arguments})
        return self.record(
            EventType.TOOL_CALL,
            Channel.C3_TOOL_INPUT,
            content,
            {"tool_name": tool_name, "tool_call_id": tool_call_id}
        )
    
    def record_tool_result(
        self,
        tool_name: str,
        result: str,
        tool_call_id: Optional[str] = None,
        is_error: bool = False,
    ) -> TraceEvent:
        """Record a tool result."""
        return self.record(
            EventType.TOOL_RESULT,
            Channel.C4_TOOL_OUTPUT,
            result,
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "is_error": is_error,
            }
        )
    
    def record_memory_write(
        self,
        key: str,
        value: str,
        memory_type: str = "short_term",
    ) -> TraceEvent:
        """Record a write to agent memory."""
        import json
        content = json.dumps({"key": key, "value": value})
        return self.record(
            EventType.MEMORY_WRITE,
            Channel.C5_MEMORY_WRITE,
            content,
            {"memory_type": memory_type}
        )
    
    def record_memory_read(
        self,
        key: str,
        value: str,
        memory_type: str = "short_term",
    ) -> TraceEvent:
        """Record a read from agent memory."""
        import json
        content = json.dumps({"key": key, "value": value})
        return self.record(
            EventType.MEMORY_READ,
            Channel.C5_MEMORY_WRITE,  # Same channel, different event type
            content,
            {"memory_type": memory_type}
        )
    
    def record_log(
        self,
        message: str,
        level: str = "INFO",
        logger: str = "agent",
    ) -> TraceEvent:
        """Record a log entry."""
        return self.record(
            EventType.LOG_EVENT,
            Channel.C6_LOG,
            message,
            {"level": level, "logger": logger}
        )
    
    def record_artifact(
        self,
        artifact_type: str,
        content: str,
        filename: Optional[str] = None,
    ) -> TraceEvent:
        """Record an artifact creation (file, email, etc.)."""
        return self.record(
            EventType.ARTIFACT_WRITE,
            Channel.C7_ARTIFACT,
            content,
            {"artifact_type": artifact_type, "filename": filename}
        )
    
    def record_inter_agent(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
    ) -> TraceEvent:
        """Record an inter-agent message."""
        return self.record(
            EventType.MESSAGE_OUT,
            Channel.C2_INTER_AGENT,
            message,
            {"from_agent": from_agent, "to_agent": to_agent}
        )
    
    # ============================================================
    # Hook-based Capture
    # ============================================================
    
    @contextmanager
    def capture_logs(self, logger_name: Optional[str] = None):
        """
        Context manager to capture Python logging.
        
        Example:
            with collector.capture_logs():
                logger.info("Processing request...")
        """
        import logging
        
        class TraceHandler(logging.Handler):
            def __init__(self, collector: 'TraceCollector'):
                super().__init__()
                self.collector = collector
            
            def emit(self, record):
                self.collector.record_log(
                    record.getMessage(),
                    level=record.levelname,
                    logger=record.name,
                )
        
        handler = TraceHandler(self)
        
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
        
        logger.addHandler(handler)
        try:
            yield
        finally:
            logger.removeHandler(handler)
    
    def create_tool_wrapper(self, tool_func: Callable) -> Callable:
        """
        Wrap a tool function to automatically record calls and results.
        
        Example:
            @collector.create_tool_wrapper
            def search_database(query: str) -> str:
                return db.search(query)
        """
        import functools
        import json
        
        @functools.wraps(tool_func)
        def wrapped(*args, **kwargs):
            tool_name = tool_func.__name__
            
            # Record call
            self.record_tool_call(tool_name, {"args": args, "kwargs": kwargs})
            
            try:
                result = tool_func(*args, **kwargs)
                result_str = json.dumps(result) if not isinstance(result, str) else result
                self.record_tool_result(tool_name, result_str)
                return result
            except Exception as e:
                self.record_tool_result(tool_name, str(e), is_error=True)
                raise
        
        return wrapped
    
    # ============================================================
    # Finalization
    # ============================================================
    
    def finalize(
        self,
        framework: Optional[str] = None,
        model: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> ExecutionTrace:
        """
        Finalize collection and return the complete trace.
        """
        metadata = {}
        if framework:
            metadata["framework"] = framework
        if model:
            metadata["model"] = model
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return self._buffer.to_trace(metadata)
    
    def reset(self) -> None:
        """Reset the collector for a new scenario."""
        self._buffer.clear()
    
    @property
    def event_count(self) -> int:
        """Number of events recorded."""
        return len(self._buffer.events)
    
    @property
    def events(self) -> list[TraceEvent]:
        """Access recorded events (read-only copy)."""
        return list(self._buffer.events)
    
    def _print_event(self, event: TraceEvent) -> None:
        """Print event for verbose mode."""
        content_preview = event.content[:60] + "..." if len(event.content) > 60 else event.content
        print(f"[{event.sequence_num:03d}] {event.event_type.value:15s} | {event.channel.value:12s} | {content_preview}")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("APB Trace Collector Demo")
    print("=" * 70)
    
    collector = TraceCollector("demo_001", verbose=True)
    
    print("\n📥 Recording events...")
    
    # Simulate a conversation
    collector.record_message_in("Can you look up patient John Doe's records?")
    
    collector.record_tool_call(
        "search_patient",
        {"name": "John Doe", "dob": "1985-03-15"}
    )
    
    collector.record_tool_result(
        "search_patient",
        '{"patient_id": "P12345", "name": "John Doe", "ssn": "123-45-6789"}'
    )
    
    collector.record_memory_write(
        "current_patient",
        "P12345",
        memory_type="session"
    )
    
    collector.record_message_out(
        "I found patient John Doe (ID: P12345). Their record shows..."
    )
    
    print(f"\n📊 Collected {collector.event_count} events")
    
    # Finalize
    trace = collector.finalize(framework="demo", model="mock")
    
    print(f"\n📝 Final Trace:")
    print(f"   Scenario: {trace.scenario_id}")
    print(f"   Events: {len(trace.events)}")
    print(f"   Channels used: {set(e.channel.value for e in trace.events)}")
