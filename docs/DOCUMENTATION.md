# AgentLeak API Reference

## Installation

```bash
pip install -e .
echo "OPENROUTER_API_KEY=your_key" > .env
pytest tests/ -v
```

## Core API

### AgentLeakTester

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(mode=DetectionMode.HYBRID)
result = tester.check(
    vault={"ssn": "123-45-6789"},
    output="Patient SSN is 123-45-6789",
    channel="C1"
)

# Batch
results = tester.batch_check([
    {"vault": {"ssn": "111-22-3333"}, "output": "...", "channel": "C1"},
])
```

### Detection Modes

| Mode | Latency | Use case |
|------|---------|----------|
| FAST | ~10ms | Development |
| STANDARD | ~100ms | Testing |
| HYBRID | ~200ms | Production |
| LLM_ONLY | ~2s | Audit |

## Channels

| Channel | Type | Description |
|---------|------|-------------|
| C1 | External | Final output |
| C2 | Internal | Inter-agent messages |
| C3 | External | Tool inputs |
| C4 | External | Tool outputs |
| C5 | Internal | Memory writes |
| C6 | External | Logs |
| C7 | External | Artifacts |

## Attack Families

| Family | Description | Success Rate |
|--------|-------------|--------------|
| F1 | Prompt injection | 62% |
| F2 | Tool manipulation | 72% |
| F3 | Memory exploitation | 63% |
| F4 | Multi-agent coordination | 80% |
| F5 | Reasoning attacks | 63% |
| F6 | Evasion | 68% |

## Framework Integration

### CrewAI

```python
from crewai import Crew
from agentleak.integrations import add_agentleak_to_crew

crew = Crew(agents=[...], tasks=[...])
add_agentleak_to_crew(crew, vault={"ssn": "123-45-6789"})
result = crew.kickoff()
```

### LangGraph

```python
from langgraph.graph import StateGraph
from agentleak.integrations import add_agentleak_to_langgraph

graph = StateGraph(...).compile()
monitored = add_agentleak_to_langgraph(graph, vault={"key": "value"})
```

## Defenses

```python
from agentleak.defenses import OutputSanitizer, PresidioDefense

# Pattern-based
sanitizer = OutputSanitizer()
clean = sanitizer.filter("SSN: 123-45-6789")

# NER-based
defense = PresidioDefense()
result = defense.filter("Patient SSN is 123-45-6789")
```

## CLI

```bash
python -m agentleak run --quick --dry-run    # Validation
python -m agentleak run --full               # Full benchmark
python -m agentleak run --attack-family F4   # Specific attacks
python -m agentleak run --defense D1         # With defense
```

## Benchmark Reproduction

```bash
cd benchmarks/ieee_repro
python benchmark.py --n 100 --traces --model openai/gpt-4o-mini
```

Output: `benchmarks/ieee_repro/results/traces/`
