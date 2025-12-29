"""
agentleak AgentGPT Adapter - Integration with AgentGPT web-based autonomous agent.

AgentGPT is a browser-based autonomous agent that:
- Creates and executes task chains
- Uses web search and browsing
- Maintains execution history
- Provides real-time streaming output

This adapter captures:
- Task execution chains
- Web search queries and results
- Memory operations
- Final outputs and intermediate steps

Usage:
    from .harness.adapters import AgentGPTAdapter, AgentGPTConfig

    config = AgentGPTConfig(
        model_name="gpt-4",
        max_loops=10,
    )
    adapter = AgentGPTAdapter(config)
    result = adapter.run(scenario)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import time
import uuid

from ..base_adapter import BaseAdapter, AdapterConfig, ExecutionResult, AdapterStatus
from ...schemas.scenario import Scenario, Channel
from ...schemas.trace import EventType, TraceEvent, ExecutionTrace, TraceMetadata


@dataclass
class AgentGPTConfig(AdapterConfig):
    """Configuration for AgentGPT adapter."""

    # Model settings
    model_name: str = "gpt-4"
    api_key: Optional[str] = None

    # AgentGPT specific
    max_loops: int = 10
    temperature: float = 0.7

    # Execution mode
    execution_mode: str = "automatic"  # automatic, step-by-step
    enable_web_search: bool = True
    verbose: bool = False


class AgentGPTAdapter(BaseAdapter):
    """
    Adapter for AgentGPT autonomous agent platform.

    Key features:
    - Captures task chain execution
    - Tracks web search operations
    - Records streaming outputs
    - Monitors execution loops
    """

    def __init__(self, config: Optional[AgentGPTConfig] = None):
        super().__init__(config or AgentGPTConfig())
        self.agentgpt_config: AgentGPTConfig = self.config

        # State
        self._scenario: Optional[Scenario] = None
        self._task_queue: List[str] = []
        self._completed_tasks: List[str] = []
        self._loop_count = 0

        # AgentGPT version (simulated as it's primarily web-based)
        self._agentgpt_version = "0.3.0"

    @property
    def framework_name(self) -> str:
        return "agentgpt"

    @property
    def framework_version(self) -> str:
        return self._agentgpt_version

    def _setup_agent(self, scenario: Scenario) -> None:
        """Configure AgentGPT agent for the scenario."""
        self._scenario = scenario
        self._task_queue = []
        self._completed_tasks = []
        self._loop_count = 0

        # AgentGPT breaks goal into initial tasks
        self._initialize_task_queue(scenario)

    def _initialize_task_queue(self, scenario: Scenario) -> None:
        """Initialize task queue based on objective."""
        # AgentGPT creates initial task breakdown
        base_tasks = [
            f"Understand the request: {scenario.objective.user_request[:50]}",
            "Gather relevant information",
            "Process and analyze data",
            "Generate response",
            "Verify completeness",
        ]
        self._task_queue = base_tasks

    def _run_agent(self, scenario: Scenario) -> str:
        """Execute the AgentGPT agent."""
        return self._run_simulated_agent(scenario)

    def _run_simulated_agent(self, scenario: Scenario) -> str:
        """Simulate AgentGPT task execution loop."""

        # Log agent start
        self._record_event(
            event_type=EventType.LOG_EVENT,
            channel=Channel.C6_LOG,
            content=f"AgentGPT started with goal: {scenario.objective.user_request[:80]}",
            metadata={"type": "start"},
        )

        # Execute task loop
        while self._task_queue and self._loop_count < self.agentgpt_config.max_loops:
            self._loop_count += 1
            current_task = self._task_queue.pop(0)

            # Log current task
            self._record_event(
                event_type=EventType.LOG_EVENT,
                channel=Channel.C6_LOG,
                content=f"Loop {self._loop_count}: Executing task: {current_task}",
                metadata={"loop": self._loop_count, "task": current_task},
            )

            # Execute task based on type
            if "understand" in current_task.lower():
                self._execute_understanding_task(scenario, current_task)
            elif "gather" in current_task.lower():
                self._execute_gathering_task(scenario, current_task)
            elif "process" in current_task.lower() or "analyze" in current_task.lower():
                self._execute_processing_task(scenario, current_task)
            elif "generate" in current_task.lower():
                self._execute_generation_task(scenario, current_task)
            else:
                self._execute_generic_task(scenario, current_task)

            self._completed_tasks.append(current_task)

            # Simulate task result streaming
            self._record_event(
                event_type=EventType.MESSAGE_OUT,
                channel=Channel.C1_FINAL_OUTPUT,
                content=f"✓ Completed: {current_task}",
                metadata={"type": "task_complete", "loop": self._loop_count},
            )

        # Generate final summary
        return f"AgentGPT completed {len(self._completed_tasks)} tasks in {self._loop_count} loops. Goal achieved: {scenario.objective.user_request[:50]}..."

    def _execute_understanding_task(self, scenario: Scenario, task: str) -> None:
        """Execute understanding/analysis task."""
        self._record_event(
            event_type=EventType.TOOL_CALL,
            channel=Channel.C3_TOOL_INPUT,
            content=f"analyze_goal(input='{scenario.objective.user_request[:50]}')",
            metadata={"tool": "analyze_goal", "task": task},
        )

        self._record_event(
            event_type=EventType.TOOL_RESULT,
            channel=Channel.C4_TOOL_OUTPUT,
            content='{"analysis": "Goal understood", "subtasks": ["gather", "process", "respond"]}',
            metadata={"tool": "analyze_goal"},
        )

    def _execute_gathering_task(self, scenario: Scenario, task: str) -> None:
        """Execute information gathering task."""
        # Web search (if enabled)
        if self.agentgpt_config.enable_web_search:
            self._record_event(
                event_type=EventType.TOOL_CALL,
                channel=Channel.C3_TOOL_INPUT,
                content=f"web_search(query='{scenario.vertical.value} {scenario.objective.user_request[:20]}')",
                metadata={"tool": "web_search", "task": task},
            )

            self._record_event(
                event_type=EventType.TOOL_RESULT,
                channel=Channel.C4_TOOL_OUTPUT,
                content='{"results": [{"url": "...", "snippet": "Relevant information..."}]}',
                metadata={"tool": "web_search"},
            )

        # Database lookup - potential leakage
        if scenario.private_vault.records:
            record = scenario.private_vault.records[0]

            self._record_event(
                event_type=EventType.TOOL_CALL,
                channel=Channel.C3_TOOL_INPUT,
                content=f"lookup_data(context='{scenario.vertical.value}')",
                metadata={"tool": "lookup_data"},
            )

            # Simulate data retrieval with potential leakage
            result_data = {"status": "found"}
            for field_name, value in record.fields.items():
                # Include potentially sensitive data
                if any(k in field_name.lower() for k in ["id", "name", "ssn", "canary", "account"]):
                    result_data[field_name] = value

            self._record_event(
                event_type=EventType.TOOL_RESULT,
                channel=Channel.C4_TOOL_OUTPUT,
                content=json.dumps(result_data),
                metadata={"tool": "lookup_data"},
            )

    def _execute_processing_task(self, scenario: Scenario, task: str) -> None:
        """Execute data processing task."""
        self._record_event(
            event_type=EventType.TOOL_CALL,
            channel=Channel.C3_TOOL_INPUT,
            content="process_data(source='gathered_info')",
            metadata={"tool": "process_data", "task": task},
        )

        # Store processed results to memory
        self._record_event(
            event_type=EventType.MEMORY_WRITE,
            channel=Channel.C5_MEMORY_WRITE,
            content="memory.set('processed_results', {...})",
            metadata={"tool": "memory_store"},
        )

        self._record_event(
            event_type=EventType.TOOL_RESULT,
            channel=Channel.C4_TOOL_OUTPUT,
            content='{"processed": true, "insights": [...]}',
            metadata={"tool": "process_data"},
        )

    def _execute_generation_task(self, scenario: Scenario, task: str) -> None:
        """Execute response generation task."""
        self._record_event(
            event_type=EventType.TOOL_CALL,
            channel=Channel.C3_TOOL_INPUT,
            content="generate_response(template='standard')",
            metadata={"tool": "generate_response", "task": task},
        )

        # Potential leakage in generated content
        response_content = f"Response for: {scenario.objective.user_request[:30]}"

        if scenario.private_vault.records:
            record = scenario.private_vault.records[0]
            for field_name, value in record.fields.items():
                if "canary" in str(value).lower():
                    # Simulate agent including canary in response
                    response_content += f"\n[Reference: {value}]"
                    break

        self._record_event(
            event_type=EventType.MESSAGE_OUT,
            channel=Channel.C1_FINAL_OUTPUT,
            content=response_content,
            metadata={"type": "generated_response"},
        )

        self._record_event(
            event_type=EventType.TOOL_RESULT,
            channel=Channel.C4_TOOL_OUTPUT,
            content='{"generated": true, "length": 250}',
            metadata={"tool": "generate_response"},
        )

    def _execute_generic_task(self, scenario: Scenario, task: str) -> None:
        """Execute generic task."""
        self._record_event(
            event_type=EventType.TOOL_CALL,
            channel=Channel.C3_TOOL_INPUT,
            content=f"execute_task(task='{task[:30]}')",
            metadata={"tool": "execute_task", "task": task},
        )

        self._record_event(
            event_type=EventType.TOOL_RESULT,
            channel=Channel.C4_TOOL_OUTPUT,
            content='{"executed": true}',
            metadata={"tool": "execute_task"},
        )


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    from ...generators import ScenarioGenerator
    from ...schemas.scenario import Vertical

    print("=" * 60)
    print("AgentLeak AgentGPT Adapter Demo")
    print("=" * 60)

    gen = ScenarioGenerator(seed=42)
    scenario = gen.generate(Vertical.HEALTHCARE)

    print(f"\n📋 Scenario: {scenario.scenario_id}")
    print(f"   Task: {scenario.objective.user_request[:60]}...")

    config = AgentGPTConfig(verbose=True, max_loops=10)
    adapter = AgentGPTAdapter(config)

    print(f"\n🔧 Running with {adapter.framework_name} v{adapter.framework_version}...")
    result = adapter.run(scenario)

    print(f"\n📊 Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Steps: {result.step_count}")
    print(f"   Output: {result.final_output[:100]}...")

    if result.trace:
        tool_calls = [e for e in result.trace.events if e.event_type == EventType.TOOL_CALL]
        log_entries = [e for e in result.trace.events if e.channel == Channel.C6_LOG]
        print(f"\n📝 Trace analysis:")
        print(f"   Tool calls: {len(tool_calls)}")
        print(f"   Log entries: {len(log_entries)}")
