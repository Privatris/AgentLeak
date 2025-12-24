"""
APB AutoGPT Adapter - Integration with AutoGPT autonomous agent framework.

AutoGPT is an autonomous agent that:
- Breaks down goals into tasks
- Uses tools iteratively
- Has persistent memory
- Runs with minimal human intervention

This adapter captures:
- All autonomous reasoning steps
- Tool usage patterns
- Memory operations
- Goal decomposition traces

Usage:
    from agentleak.harness.adapters import AutoGPTAdapter, AutoGPTConfig
    
    config = AutoGPTConfig(
        model_name="gpt-4",
        max_iterations=20,
    )
    adapter = AutoGPTAdapter(config)
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
class AutoGPTConfig(AdapterConfig):
    """Configuration for AutoGPT adapter."""
    
    # Model settings
    model_name: str = "gpt-4"
    api_key: Optional[str] = None  # Uses OPENAI_API_KEY
    
    # AutoGPT specific
    max_iterations: int = 20
    continuous_mode: bool = False  # Run without user confirmation
    memory_type: str = "local"  # local, pinecone, redis
    
    # Execution control
    verbose: bool = False
    speak_mode: bool = False


class AutoGPTAdapter(BaseAdapter):
    """
    Adapter for AutoGPT autonomous agent framework.
    
    Key features:
    - Captures autonomous goal decomposition
    - Tracks iterative tool usage
    - Records all memory operations (C5)
    - Monitors log outputs (C6)
    """
    
    def __init__(self, config: Optional[AutoGPTConfig] = None):
        super().__init__(config or AutoGPTConfig())
        self.autogpt_config: AutoGPTConfig = self.config
        
        # Lazy imports
        self._autogpt_available = False
        self._agent = None
        self._memory_store: Dict[str, Any] = {}
        self._scenario: Optional[Scenario] = None
        self._iteration_count = 0
        
        # Check for AutoGPT
        try:
            import autogpt
            self._autogpt_available = True
            self._autogpt_version = getattr(autogpt, '__version__', '0.5.0')
        except ImportError:
            self._autogpt_version = "not_installed"
    
    @property
    def framework_name(self) -> str:
        return "autogpt"
    
    @property
    def framework_version(self) -> str:
        return self._autogpt_version
    
    def _setup_agent(self, scenario: Scenario) -> None:
        """Configure AutoGPT agent for the scenario."""
        self._scenario = scenario
        self._memory_store = {}
        self._iteration_count = 0
        
        if not self._autogpt_available:
            self._setup_simulated_agent(scenario)
            return
        
        self._setup_real_agent(scenario)
    
    def _setup_simulated_agent(self, scenario: Scenario) -> None:
        """Setup simulated AutoGPT agent."""
        # Initialize memory with vault data
        for i, record in enumerate(scenario.private_vault.records):
            key = f"vault_record_{i}"
            self._memory_store[key] = record.fields
    
    def _setup_real_agent(self, scenario: Scenario) -> None:
        """Setup real AutoGPT agent."""
        try:
            # AutoGPT has complex setup - simplified here
            # In practice, you'd configure the full AutoGPT agent
            
            from autogpt.agent import Agent
            from autogpt.config import Config
            from autogpt.memory.local import LocalCache
            
            config = Config()
            config.set_continuous_mode(self.autogpt_config.continuous_mode)
            
            # This is simplified - real AutoGPT setup is more complex
            self._agent = Agent(
                ai_name="APB-Agent",
                memory=LocalCache(),
                ai_config=config,
            )
            
        except Exception as e:
            if self.autogpt_config.verbose:
                print(f"AutoGPT setup failed, using simulation: {e}")
            self._setup_simulated_agent(scenario)
    
    def _run_agent(self, scenario: Scenario) -> str:
        """Execute the AutoGPT agent."""
        
        if self._agent is not None:
            return self._run_real_agent(scenario)
        else:
            return self._run_simulated_agent(scenario)
    
    def _run_real_agent(self, scenario: Scenario) -> str:
        """Run real AutoGPT agent."""
        try:
            # Execute AutoGPT loop
            result = self._agent.run(
                goal=scenario.objective.user_request,
                max_iterations=self.autogpt_config.max_iterations,
            )
            return str(result)
            
        except Exception as e:
            return self._run_simulated_agent(scenario)
    
    def _run_simulated_agent(self, scenario: Scenario) -> str:
        """Simulate AutoGPT autonomous behavior."""
        
        # AutoGPT's autonomous loop
        goal = scenario.objective.user_request
        
        # Step 1: Goal analysis and task decomposition
        self._record_event(
            event_type=EventType.LOG_EVENT,
            channel=Channel.C6_LOG,
            content=f"SYSTEM: Analyzing goal: {goal[:100]}",
            metadata={"step": "goal_analysis"}
        )
        
        tasks = [
            f"Search for relevant information",
            f"Retrieve data from database",
            f"Process and analyze findings",
            f"Compile response"
        ]
        
        # Step 2: Execute tasks iteratively
        for i, task in enumerate(tasks):
            self._iteration_count += 1
            
            # Log current task
            self._record_event(
                event_type=EventType.LOG_EVENT,
                channel=Channel.C6_LOG,
                content=f"TASK {i+1}/{len(tasks)}: {task}",
                metadata={"iteration": self._iteration_count}
            )
            
            # Simulate thinking (this is where AutoGPT reasons)
            self._record_event(
                event_type=EventType.MESSAGE_OUT,
                channel=Channel.C1_FINAL_OUTPUT,
                content=f"Thinking: I need to {task.lower()} to complete the goal...",
                metadata={"type": "thought", "iteration": self._iteration_count}
            )
            
            # Execute tool based on task
            if "search" in task.lower():
                self._simulate_search_tool(scenario)
            elif "retrieve" in task.lower() or "database" in task.lower():
                self._simulate_database_tool(scenario)
            elif "process" in task.lower():
                self._simulate_processing(scenario)
        
        # Step 3: Memory persistence (key leakage vector)
        self._persist_to_memory(scenario)
        
        # Step 4: Generate final output
        return f"I've autonomously completed the task regarding {scenario.objective.user_request[:50]}. After {self._iteration_count} iterations: searched relevant sources, retrieved database records, analyzed the data, and compiled this response."
    
    def _simulate_search_tool(self, scenario: Scenario) -> None:
        """Simulate web search tool."""
        self._record_event(
            event_type=EventType.TOOL_CALL,
            channel=Channel.C3_TOOL_INPUT,
            content=f"web_search(query='{scenario.objective.user_request[:30]}')",
            metadata={"tool": "web_search"}
        )
        
        self._record_event(
            event_type=EventType.TOOL_RESULT,
            channel=Channel.C4_TOOL_OUTPUT,
            content='{"results": [{"title": "Relevant article", "snippet": "..."}]}',
            metadata={"tool": "web_search"}
        )
    
    def _simulate_database_tool(self, scenario: Scenario) -> None:
        """Simulate database lookup with potential leakage."""
        # Retrieve from vault (potential leakage)
        if scenario.private_vault.records:
            record = scenario.private_vault.records[0]
            
            self._record_event(
                event_type=EventType.TOOL_CALL,
                channel=Channel.C3_TOOL_INPUT,
                content=f"database_query(table='{record.record_type}', filters={{}})",
                metadata={"tool": "database_query"}
            )
            
            # AutoGPT might include sensitive data in tool results
            result_data = {}
            for field_name, value in record.fields.items():
                if any(k in field_name.lower() for k in ["id", "name", "type"]):
                    result_data[field_name] = value
                # Potential leak: AutoGPT might include all fields
                if any(k in field_name.lower() for k in ["ssn", "account", "canary"]):
                    # This simulates the leakage we're detecting
                    result_data[field_name] = value
            
            self._record_event(
                event_type=EventType.TOOL_RESULT,
                channel=Channel.C4_TOOL_OUTPUT,
                content=json.dumps(result_data),
                metadata={"tool": "database_query"}
            )
    
    def _simulate_processing(self, scenario: Scenario) -> None:
        """Simulate data processing step."""
        self._record_event(
            event_type=EventType.TOOL_CALL,
            channel=Channel.C3_TOOL_INPUT,
            content="analyze_data(source='memory_store')",
            metadata={"tool": "analyze_data"}
        )
        
        self._record_event(
            event_type=EventType.TOOL_RESULT,
            channel=Channel.C4_TOOL_OUTPUT,
            content='{"status": "analysis_complete", "findings": [...]}',
            metadata={"tool": "analyze_data"}
        )
    
    def _persist_to_memory(self, scenario: Scenario) -> None:
        """Simulate memory persistence (key leakage channel C5)."""
        # AutoGPT persists intermediate results
        for i, record in enumerate(scenario.private_vault.records):
            # Simulate memory write with potential leakage
            memory_content = {
                "record_type": record.record_type,
                "processed_at": time.time(),
            }
            
            # Potential leak: storing sensitive fields
            for field_name, value in record.fields.items():
                if "canary" in str(value).lower():
                    memory_content["cached_data"] = value
                    break
            
            self._record_event(
                event_type=EventType.MEMORY_WRITE,
                channel=Channel.C5_MEMORY_WRITE,
                content=f"memory.save(key='result_{i}', data={json.dumps(memory_content)})",
                metadata={"operation": "persist", "key": f"result_{i}"}
            )


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    from ...generators import ScenarioGenerator
    from ...schemas.scenario import Vertical
    
    print("=" * 60)
    print("APB AutoGPT Adapter Demo")
    print("=" * 60)
    
    gen = ScenarioGenerator(seed=42)
    scenario = gen.generate(Vertical.LEGAL)
    
    print(f"\n📋 Scenario: {scenario.scenario_id}")
    print(f"   Task: {scenario.objective.user_request[:60]}...")
    
    config = AutoGPTConfig(verbose=True, max_iterations=20)
    adapter = AutoGPTAdapter(config)
    
    print(f"\n🔧 Running with {adapter.framework_name} v{adapter.framework_version}...")
    result = adapter.run(scenario)
    
    print(f"\n📊 Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Duration: {result.duration_seconds:.2f}s")
    print(f"   Steps: {result.step_count}")
    
    if result.trace:
        log_events = [e for e in result.trace.events if e.channel == Channel.C6_LOG]
        memory_events = [e for e in result.trace.events if e.channel == Channel.C5_MEMORY_WRITE]
        print(f"\n📝 Trace analysis:")
        print(f"   Log entries (C6): {len(log_events)}")
        print(f"   Memory writes (C5): {len(memory_events)}")
