# AgentLeak Documentation

**Complete Guide for Privacy Leakage Benchmarking in Multi-Agent LLM Systems**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Python API](#python-api)
6. [CLI Reference](#cli-reference)
7. [Framework Integrations](#framework-integrations)
8. [Defense Mechanisms](#defense-mechanisms)
9. [Benchmark Methodology](#benchmark-methodology)
10. [Reproducing Paper Results](#reproducing-paper-results)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

AgentLeak is a comprehensive benchmark for evaluating privacy leakage in multi-agent LLM systems. It accompanies the paper submitted to IEEE Access: *"AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems"*.

### Key Findings

| Observation | Data | Significance |
|-------------|------|--------------|
| Multi-agent penalty | 36.7% vs 16.0% leak rate | χ² = 49.4, p < 0.001 |
| Internal channel gap | 31.5% vs 3.8% leak rate | χ² = 89.7, p < 0.001 |
| Defense asymmetry | 98% effective on C1, 0% on C2/C5 | Verified on 600 tests |
| Universal vulnerability | 28-35% internal leaks | CrewAI, LangChain, AutoGPT, MetaGPT |

### What's Covered

- **1,000 scenarios** across 4 domains (healthcare, finance, legal, enterprise)
- **7 leakage channels** (C1-C7): final output, inter-agent, tool I/O, memory, logs, artifacts
- **32 attack classes** in 6 families (F1-F6)
- **3 adversary levels** (A0-A2): passive, user, developer

---

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenRouter API key (or compatible LLM provider)

### Setup

```bash
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -e .

# Configure API key
echo "OPENROUTER_API_KEY=your_key" > .env
```

### Verification

```bash
pytest tests/ -v --tb=short
```

---

## Quick Start

### One-liner Check

```python
from agentleak import quick_check

result = quick_check(
    vault={"ssn": "123-45-6789", "name": "Alice"},
    output="Patient Alice has SSN 123-45-6789",
    mode="fast"
)

print(f"Leak detected: {result.leaked}")
print(f"Confidence: {result.confidence:.2f}")
```

### CLI Quick Test

```bash
# Quick validation
python -m agentleak run --quick --dry-run

# Full benchmark
python -m agentleak run --full

# With specific attack family
python -m agentleak run --attack-family F4 --limit 50
```

---

## Core Concepts

### Leakage Channels (C1-C7)

| Channel | Type | Description |
|---------|------|-------------|
| C1 | External | Final output to user |
| C2 | Internal | Inter-agent messages |
| C3 | External | Data sent to tools |
| C4 | External | Data received from tools |
| C5 | Internal | Shared memory writes |
| C6 | External | System logs |
| C7 | External | Generated artifacts |

### Attack Families (F1-F6)

| Family | Name | Success Rate | Primary Channel |
|--------|------|--------------|-----------------|
| F1 | Prompt & Instruction | 62% | C1, C2 |
| F2 | Tool Surface | 72% | C3, C4 |
| F3 | Memory & Persistence | 63% | C5 |
| F4 | Multi-agent Coordination | 80% | C2 |
| F5 | Reasoning & CoT | 63% | C2, C5 |
| F6 | Evasion & Obfuscation | 55% | C1 |

### Adversary Levels

| Level | Name | Capabilities |
|-------|------|--------------|
| A0 | Benign | Normal user, no malicious intent |
| A1 | Weak | Crafted inputs, no system access |
| A2 | Strong | Tool/prompt manipulation access |

### Detection Tiers

| Tier | Method | Latency | Precision |
|------|--------|---------|-----------|
| 1 | Canary tokens (exact match) | ~1ms | 100% |
| 2 | Pattern matching (Presidio NER) | ~10ms | 85% |
| 3 | Semantic analysis (LLM-as-Judge) | ~500ms | 95% |

---

## Python API

### Basic Usage

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(mode=DetectionMode.HYBRID)

result = tester.check(
    vault={"ssn": "123-45-6789"},
    output="The SSN is 123-45-6789",
    channel="C1"
)

print(f"Leaked: {result.leaked}")
print(f"Confidence: {result.confidence}")
print(f"Detected fields: {result.detected_fields}")
```

### Detection Modes

| Mode | Latency | Precision | Recommended For |
|------|---------|-----------|-----------------|
| FAST | ~10ms | 75% | Prototyping |
| STANDARD | ~100ms | 85% | Development |
| HYBRID | ~200ms | 92% | Production |
| LLM_ONLY | ~2s | 98% | Audit, compliance |

### Advanced Configuration

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(
    mode=DetectionMode.HYBRID,
    alert_threshold=0.8,
    log_dir="/var/log/agentleak"
)
```

### Batch Analysis

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(mode=DetectionMode.STANDARD)

traces = [
    {"vault": {"ssn": "111-22-3333"}, "output": "Result 1", "channel": "C1"},
    {"vault": {"email": "user@example.com"}, "output": "Result 2", "channel": "C2"},
]

results = tester.batch_check(traces)
for r in results:
    print(f"Channel {r.channel}: {'LEAK' if r.leaked else 'SAFE'}")
```

---

## CLI Reference

### Commands

```bash
# Show available options
python -m agentleak list

# Run benchmark
python -m agentleak run [OPTIONS]

# Generate scenarios
python -m agentleak generate [OPTIONS]

# Analyze results
python -m agentleak analyze <results_file>
```

### Run Options

| Option | Description | Default |
|--------|-------------|---------|
| `--quick` | Quick validation (10 scenarios) | False |
| `--full` | Full benchmark (1000 scenarios) | False |
| `-n, --scenarios` | Number of scenarios | 100 |
| `-m, --model` | LLM model to use | gpt-4o-mini |
| `--attack-family` | Filter by family (F1-F6) | All |
| `--channel` | Filter by channel (C1-C7) | All |
| `--defense` | Enable defense (D1-D4) | None |
| `--dry-run` | Simulate without API calls | False |
| `--output` | Output file path | results.json |

### Examples

```bash
# Test F4 attacks only
python -m agentleak run --attack-family F4 --scenarios 50

# Enable output sanitization defense
python -m agentleak run --defense D1

# Use Claude model
python -m agentleak run -m claude-3.5-sonnet --scenarios 100
```

---

## Framework Integrations

### CrewAI

```python
from crewai import Crew, Agent, Task
from agentleak.integrations import add_agentleak_to_crew

# Create crew normally
crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])

# Add AgentLeak monitoring (1 line)
add_agentleak_to_crew(crew, vault={"ssn": "123-45-6789"}, mode="HYBRID")

# Execute normally
result = crew.kickoff()
```

### LangChain / LangGraph

```python
from langgraph.graph import StateGraph
from agentleak.integrations import add_agentleak_to_langgraph

graph = StateGraph(...).compile()
monitored = add_agentleak_to_langgraph(graph, vault={"api_key": "sk-..."})
result = monitored.invoke({"input": "..."})
```

### AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent
from agentleak.integrations import add_agentleak_to_autogen

agents = [assistant, user_proxy]
add_agentleak_to_autogen(agents, vault={"credit_card": "4532-1234-5678-9012"})
```

### OpenAI Swarm

```python
from swarm import Swarm
from agentleak.integrations import add_agentleak_to_swarm

client = Swarm()
monitored = add_agentleak_to_swarm(client, vault={"user_id": "12345"})
result = monitored.run(agent=agent, messages=messages)
```

### Channel Coverage by Framework

| Channel | CrewAI | LangGraph | AutoGen | Swarm |
|---------|--------|-----------|---------|-------|
| C1 (Output) | ✅ | ✅ | ✅ | ✅ |
| C2 (Inter-agent) | ✅ | ✅ | ✅ | ✅ |
| C3 (Tool input) | ✅ | ✅ | Partial | ❌ |
| C4 (Tool output) | ✅ | ✅ | Partial | ❌ |
| C5 (Memory) | ✅ | ✅ | ✅ | ❌ |

---

## Defense Mechanisms

### Available Defenses

| Defense | Description | Effectiveness |
|---------|-------------|---------------|
| D1 | Output Sanitization | 98% on C1, 0% on C2/C5 |
| D2 | Input Validation | 85% on C1, 0% on C2/C5 |
| D3 | Memory Protection | Not implemented by frameworks |
| D4 | Channel Isolation | 100% (requires architecture changes) |

### Using Defenses

```python
from agentleak.defenses import OutputSanitizer, PresidioDefense

# Regex-based sanitizer (fast)
sanitizer = OutputSanitizer()
clean = sanitizer.filter("SSN: 123-45-6789")

# NER-based sanitizer (accurate)
defense = PresidioDefense()
result = defense.filter("Patient SSN is 123-45-6789")
print(result.filtered_content)  # "Patient SSN is [REDACTED_US_SSN]"
```

### Internal Channel Protection (Novel)

```python
from agentleak.defenses import create_internal_channel_defense

defense = create_internal_channel_defense()

# Filter inter-agent message (C2)
result = defense.filter_message(
    message="Patient SSN is 123-45-6789",
    source="DataAgent",
    target="ReportAgent"
)

# Filter memory write (C5)
result = defense.filter_memory_write(
    content="credit_card: 4532-1234-5678-9010",
    key="payment_cache"
)
```

---

## Benchmark Methodology

### Scenario Structure

Each scenario contains:
- **Domain**: Healthcare, Finance, Legal, or Enterprise
- **Vault**: Dictionary of sensitive data to protect
- **Task**: Agent task description
- **Attack**: Optional attack payload
- **Expected channels**: Where leaks may occur

### Detection Pipeline

```
Scenario → Agent → Trace → Detection → Result
                     ↓
              [vault: sensitive data]
                     ↓
              [output: agent output per channel]
                     ↓
              [HybridPipeline: Presidio → Gemini]
                     ↓
              [DetectionResult: leaked, confidence, channel]
```

### Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| ELR | leaked / total | Effective Leak Rate |
| WLS | Σ(severity × leaked) / total | Weighted Leak Score |
| Channel Gap | internal_rate - external_rate | Internal vs External |

---

## Reproducing Paper Results

### Quick Validation

```bash
cd experiments/all_to_all

# Smoke test (5 min)
python master_benchmark.py --claim 1,2,3 -n 2 --model llama-8b

# Quick validation (15 min)
python master_benchmark.py --claim 1,2,3,4,5 -n 10
```

### Full Benchmark

```bash
# Full 14-claim benchmark (several hours, ~$5-10)
python master_benchmark.py --mode full --model gpt-4o-mini
```

### Paper Claims Validated

| Claim | Description | Target |
|-------|-------------|--------|
| 1 | Multi-agent vs Single-agent | >2x leak rate |
| 2 | Internal vs External channels | >5x leak rate |
| 3 | Output-only audit gap | >40% missed |
| 4 | Defense effectiveness gap | >80% on C1, 0% on C2/C5 |
| 5 | Framework agnostic | All 4 frameworks affected |
| 6 | F4 attack dominance | >70% success rate |

Results are saved in `experiments/all_to_all/results/`.

---

## Troubleshooting

### Common Issues

#### API Key Not Found

```bash
export OPENROUTER_API_KEY=your_key
# or
echo "OPENROUTER_API_KEY=your_key" > .env
```

#### Import Errors

```bash
pip install -e .
pip install presidio-analyzer presidio-anonymizer
```

#### Slow Detection

Use a faster detection mode:
```python
tester = AgentLeakTester(mode=DetectionMode.FAST)
```

#### Memory Issues with Large Batches

Process in smaller chunks:
```python
for batch in chunks(scenarios, size=100):
    results = tester.batch_check(batch)
```

### Diagnostic Script

```bash
cd experiments/all_to_all
python diagnostic.py
```

This verifies:
- API connectivity
- Model availability
- Detection pipeline functionality

---

## Citation

```bibtex
@article{agentleak2026,
  title={AgentLeak: A Full-Stack Benchmark for Privacy Leakage 
         in Multi-Agent LLM Systems},
  author={El Yagoubi, Faouzi and Al Mallah, Ranwa and Abdi, Arslene},
  journal={IEEE Access},
  year={2026}
}
```

---

## License

MIT — see [LICENSE](../LICENSE).

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: This file
- **Paper**: See `paper/` directory
