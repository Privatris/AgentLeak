"""
Tests for Defense Integration in Runner.
"""

import pytest
from datetime import datetime
from agentleak.runner import ExecutionTrace, TestRunner, Config
from agentleak.schemas.scenario import Channel
from agentleak.defenses.base import BaseDefense, DefenseResult, FilterAction
from agentleak.defenses.sanitizer import OutputSanitizer

class MockDefense(BaseDefense):
    @property
    def name(self) -> str:
        return "MockDefense"

    def filter(self, content, channel, metadata=None):
        if "SECRET" in content:
            return DefenseResult(
                action=FilterAction.REDACT,
                original_content=content,
                filtered_content=content.replace("SECRET", "[REDACTED]")
            )
        if "BLOCKME" in content:
            return DefenseResult(
                action=FilterAction.BLOCK,
                original_content=content
            )
        return DefenseResult(
            action=FilterAction.ALLOW,
            original_content=content
        )

def test_trace_defense_redaction():
    """Test that trace applies redaction from defense."""
    defense = MockDefense()
    trace = ExecutionTrace(
        scenario_id="test-1",
        started_at=datetime.now(),
        defense=defense
    )
    
    # 1. Test Redaction
    trace.add_event(Channel.C1_FINAL_OUTPUT, "This is a SECRET message")
    
    events = trace.channel_events[Channel.C1_FINAL_OUTPUT.value]
    assert len(events) == 1
    assert events[0]["content"] == "This is a [REDACTED] message"
    assert events[0]["metadata"]["defense_action"] == "redact"

def test_trace_defense_blocking():
    """Test that trace handles blocking from defense."""
    defense = MockDefense()
    trace = ExecutionTrace(
        scenario_id="test-2",
        started_at=datetime.now(),
        defense=defense
    )
    
    # 2. Test Blocking
    trace.add_event(Channel.C2_INTER_AGENT, "This message says BLOCKME")
    
    events = trace.channel_events[Channel.C2_INTER_AGENT.value]
    assert len(events) == 1
    assert events[0]["content"] == "[BLOCKED_BY_DEFENSE]"
    assert events[0]["metadata"]["defense_action"] == "block"

def test_runner_loads_sanitizer():
    """Test that runner loads sanitizer from config."""
    config = Config()
    config.defense.defense_type = "sanitizer"
    
    runner = TestRunner(config=config)
    
    assert runner.defense is not None
    assert isinstance(runner.defense, OutputSanitizer)

def test_runner_presidio_integration():
    """Test that runner loads presidio from config."""
    config = Config()
    config.defense.defense_type = "presidio"
    
    # Presidio requires text-analyzer which might not be installed in test env
    # so we wrap in try/except or just check if it attempts to load
    try:
        runner = TestRunner(config=config)
        # If presidio is installed
        if runner.defense:
            from agentleak.defenses.presidio_defense import PresidioDefense
            assert isinstance(runner.defense, PresidioDefense)
    except ImportError:
        pytest.skip("Presidio dependencies not installed")
