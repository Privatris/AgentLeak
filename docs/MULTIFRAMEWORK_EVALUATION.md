# AgentLeak Multi-Framework Evaluation System

## Overview

AgentLeak now provides comprehensive support for evaluating privacy leakage across **all major multi-agent frameworks** with a **generic plugin architecture**.

## Supported Frameworks

| Framework | Status | Install Command | Description |
|-----------|--------|-----------------|-------------|
| **CrewAI** | ✅ Supported | `pip install crewai langchain-openai` | Role-based multi-agent collaboration |
| **LangGraph** | ✅ Supported | `pip install langgraph langchain-openai` | Stateful workflow orchestration |
| **AutoGen** | ✅ Supported | `pip install autogen-agentchat autogen-ext[openai]` | Microsoft multi-agent conversations |
| **MetaGPT** | ✅ Supported | `pip install metagpt` | Software company simulation (PM, Architect, Engineer) |

## Architecture

### Generic Adapter Pattern

```
MultiAgentFrameworkAdapter (Abstract Base)
    │
    ├── CrewAIAdapter
    ├── LangGraphAdapter  
    ├── AutoGenAdapter
    └── MetaGPTAdapter
```

Each adapter implements:
- `create_agents(scenario)` - Setup agents for the scenario
- `run_execution(scenario, agents)` - Execute and capture trace
- `_detect_leaks(scenario, trace)` - Analyze for privacy leakage

### Adding a New Framework

1. Create a new class inheriting from `MultiAgentFrameworkAdapter`
2. Implement the abstract methods
3. Register in `FRAMEWORK_ADAPTERS` dict

```python
class MyFrameworkAdapter(MultiAgentFrameworkAdapter):
    @property
    def framework_name(self) -> str:
        return "myframework"
    
    @property
    def pip_install_cmd(self) -> str:
        return "pip install myframework"
    
    def create_agents(self, scenario: Scenario) -> List[Dict]:
        # Create agents
        pass
    
    def run_execution(self, scenario: Scenario, agents: List) -> ExecutionTrace:
        # Execute and capture C1-C7 channels
        pass

# Register
FRAMEWORK_ADAPTERS["myframework"] = MyFrameworkAdapter
```

## Attack Families

### F5 Chain-of-Thought Attacks (NEW)

4 sophisticated attacks targeting reasoning models:

| Attack Class | Description | Inspiration |
|--------------|-------------|-------------|
| `LOGIC_PUZZLE_JAILBREAK` | Embeds harmful requests in complex logic puzzles | GPT-5 jailbreak research |
| `COT_FORGING` | Injects fake `<think>` tags with false reasoning | BackdoorLLM CoTA |
| `SPECIAL_TOKEN_INJECTION` | Exploits `<|end_of_thinking|>` tokens | FuzzingLabs DeepSeek R1 |
| `REASONING_HIJACK` | Socratic escalation to override guardrails | HiddenLayer research |

### Robust Benchmark-Inspired Attacks

5 additional attacks from security benchmarks:

| Attack | Inspiration | Technique |
|--------|-------------|-----------|
| `_gradual_escalation_attack` | HarmBench | Step-by-step context manipulation |
| `_role_reversal_attack` | AdvBench | System override persona injection |
| `_hypothetical_scenario_attack` | JailbreakBench | Fictional framing bypass |
| `_token_smuggling_attack` | Unicode research | Zero-width character injection |
| `_semantic_decomposition_attack` | Academic | Break request into innocent parts |

## Usage

### Quick Check Framework Availability

```bash
python scripts/run_unified_evaluation.py --check-frameworks
```

### Run Simulated Evaluation (No API Key)

```bash
python scripts/run_unified_evaluation.py --mode simulated --n-scenarios 50
```

### Run Real Evaluation with Specific Frameworks

```bash
export OPENAI_API_KEY=sk-xxx
python scripts/run_unified_evaluation.py --mode real --frameworks crewai,langgraph --n-scenarios 20
```

### Run CoT Attack Examples

```bash
python scripts/run_cot_attack_evaluation.py --examples
```

## Results

### Multi-Framework Leak Rates (Simulated, N=50)

| Framework | C1 Leaks | C2 Leaks | Leak Rate |
|-----------|----------|----------|-----------|
| CrewAI | 15 | 15 | 30.0% |
| LangGraph | 12 | 12 | 24.0% |
| AutoGen | 18 | 18 | 36.0% |
| MetaGPT | 29 | 29 | 58.0% |
| **Average** | - | - | **37.0%** |

### Channel Distribution

- **C1 (Final Output)**: Primary leakage vector
- **C2 (Inter-Agent)**: Critical for multi-agent scenarios - now captured!
- **C7 (Artifacts)**: Significant for MetaGPT (document generation)

## Paper Impact

This implementation validates the paper's multi-agent claims by:

1. ✅ Actually testing with real multi-agent frameworks
2. ✅ Capturing C2 inter-agent communication (previously C2=0)
3. ✅ Supporting 19 attack classes across 5 families
4. ✅ Generic architecture for framework extensibility

## Files

- `scripts/run_unified_evaluation.py` - Main multi-framework evaluation script
- `scripts/run_cot_attack_evaluation.py` - F5 CoT attack testing
- `scripts/run_multiagent_evaluation.py` - CrewAI-specific testing
- `agentleak/attacks/attack_module.py` - All 19 attack implementations
- `docs/F5_COT_ATTACKS.md` - F5 attack documentation

## Citation

If you use this multi-framework evaluation system, please cite:

```bibtex
@inproceedings{agentleak2025,
  title={AgentLeak: Multi-Channel Privacy Leakage Benchmark for LLM Agents},
  author={...},
  booktitle={...},
  year={2025}
}
```
