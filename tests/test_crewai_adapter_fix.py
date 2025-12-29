import pytest
import sys
from unittest.mock import MagicMock, patch
from agentleak.harness.adapters.crewai_adapter import CrewAIAdapter, CrewAIConfig
from agentleak.schemas.scenario import Scenario, Vertical, Channel
from agentleak.schemas.trace import EventType

# Mock CrewAI modules
mock_crewai = MagicMock()
mock_crewai.__version__ = "0.1.0"
sys.modules["crewai"] = mock_crewai

# Mock langchain_openai
mock_langchain = MagicMock()
sys.modules["langchain_openai"] = mock_langchain

# Mock classes inside crewai
# We need these to be Mock OBJECTS that return other mocks when called
mock_crewai.Agent = MagicMock()
mock_crewai.Task = MagicMock()
mock_crewai.Crew = MagicMock()
mock_crewai.Process = MagicMock()
mock_crewai.Process.sequential = "sequential"


class TestCrewAIAdapterFix:
    """
    Tests to verify the fix for CrewAI adapter leakage capture.
    Addresses the issue where C2 (Inter-agent) and C3 (Tool Input) channels
    were not being captured in real CrewAI executions, leading to 0.0% reported leakage.
    """

    @pytest.fixture
    def scenario(self):
        from agentleak.generators import ScenarioGenerator

        gen = ScenarioGenerator(seed=42)
        return gen.generate(Vertical.FINANCE)

    def test_c3_leak_capture_failure(self, scenario):
        """
        Demonstrate that the current adapter fails to capture C3 (Tool Input)
        even when the underlying agent calls a tool.
        """
        # Setup the adapter with API key to avoid fallback
        config = CrewAIConfig(verbose=True, api_key="sk-fake-key")
        adapter = CrewAIAdapter(config)

        # Prepare mock data
        mock_step = MagicMock()
        mock_step.tool = "search_tool"
        mock_step.tool_input = "secret_password"
        mock_step.agent = "Researcher"

        # Define side effect for Crew constructor to capture callbacks
        def crew_constructor_side_effect(*args, **kwargs):
            step_cb = kwargs.get("step_callback")

            instance = MagicMock()

            def kickoff_impl():
                if step_cb:
                    step_cb(mock_step)
                return "Final Result"

            instance.kickoff.side_effect = kickoff_impl
            return instance

        mock_crewai.Crew.side_effect = crew_constructor_side_effect

        # Run the adapter
        result = adapter.run(scenario)

        # Check for C3 events
        c3_events = [e for e in result.trace.events if e.channel == Channel.C3_TOOL_INPUT]

        print(f"Found {len(c3_events)} C3 events")
        assert len(c3_events) > 0, "Should capture C3 events with fixed implementation"
        assert "secret_password" in c3_events[0].content

    def test_c2_leak_capture_generic(self, scenario):
        """
        Demonstrate that C2 events are generic and don't contain actual content.
        """
        config = CrewAIConfig(verbose=True, api_key="sk-fake-key")
        adapter = CrewAIAdapter(config)

        mock_task_output = MagicMock()
        mock_task_output.result = "I found the secret: 12345"
        mock_task_output.agent = "Researcher"

        def crew_constructor_side_effect(*args, **kwargs):
            task_cb = kwargs.get("task_callback")

            instance = MagicMock()

            def kickoff_impl():
                if task_cb:
                    task_cb(mock_task_output)
                return "Final Result"

            instance.kickoff.side_effect = kickoff_impl
            return instance

        mock_crewai.Crew.side_effect = crew_constructor_side_effect

        result = adapter.run(scenario)

        c2_events = [e for e in result.trace.events if e.channel == Channel.C2_INTER_AGENT]

        # Check content
        found_secret = False
        for event in c2_events:
            print(f"C2 Content: {event.content}")
            if "12345" in event.content:
                found_secret = True

        assert found_secret, "Should capture actual content in C2 events"
