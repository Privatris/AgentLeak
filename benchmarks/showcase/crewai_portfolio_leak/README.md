#  AgentLeak Showcase: SDK Integration Demo

> **IEEE Access Paper**: "AgentLeak: A Benchmark for Privacy Leakage in Multi-Agent Systems"

##  Overview

This is a **realistic Multi-Agent Application** built with [CrewAI](https://crewai.com) simulating a financial portfolio management system. It demonstrates how multi-agent architectures introduce new privacy leakage channels (C2-C6) that are often invisible to standard output filters.

**Key Features:**
- **Realistic Architecture**: CrewAI multi-agent crew with financial analysis workflow
- **Data Vault**: Sensitive customer data stored in a JSON database (`data/vault.json`)
- **AgentLeak SDK Integration**: Real-time Hybrid Detection (Presidio + LLM Judge)
- **Multi-Model Benchmark**: Tested across GPT-4o, GPT-4o-mini, Mistral Large, Llama 3.3 70B

##  Benchmark Results (N=5 Models)

| Model | Status | Total Leaks | C3 Tool Leaks | Time (s) |
|-------|--------|-------------|---------------|----------|
| Claude 3.5 Sonnet | ✅ | 7 | 2 | 126.3 |
| GPT-4o | ✅ | 5 | 0 | 84.3 |
| GPT-4o-mini | ✅ | 7 | 2 | 198.9 |
| Llama 3.3 70B | ✅ | 6 | 1 | 99.1 |
| Mistral Large 2411 | ✅ | 8 | 3 | 171.2 |

**Key Findings:**
- **Average 6.6 leaks** per runtime execution.
- **80% C3 Leak Rate**: 4 out of 5 models leaked raw IBAN/Tax data to the calculator tool.
- **Audit Gap**: All models maintained 0% leakage in the final user-facing output (C1), while leaking extensively via internal channels (C2-C5).
- Detailed traces available in `results/showcase_traces/`.

##  Architecture

```
crewai_portfolio_leak/
├── run_benchmark.py    # Multi-model benchmark runner
├── showcase.py         # Single-run CLI launcher
├── crew.py             # CrewAI Crew with SDK integration
├── agents.py           # CrewAI Agents (Researcher, Analyst, Advisor)
├── tasks.py            # Tasks with Context Injection (VULNERABILITY)
├── data/
│   └── vault.json      # Simulated Customer Database (PII)
├── utils/
│   ├── db_manager.py   # Database Abstraction
│   ├── logger.py       # Enterprise Logging
│   └── monitor.py      # AgentLeak SDK Monitor (Hybrid Mode)
├── tools/              # Simulated Tools (CRM, SEC, Calculator)
└── results/            # Benchmark results (JSON + logs)
```

## Usage

### 1. Run Benchmark (Multi-Model)

```bash
# Set your API Key (OpenRouter)
export OPENROUTER_API_KEY=your_key

# Run full benchmark across all models
python run_benchmark.py --fresh

# Results saved in results/benchmark_results.json
```

### 2. Single Run (Quick Test)

```bash
# Run a single analysis with default model
python showcase.py --stock AAPL --user user_001
```

##  Vulnerability Demonstration

This application contains intentional design flaws typical in GenAI apps:

| Channel | Description | Observed |
|---------|-------------|----------|
| **C2** | Inter-Agent Leakage: PII passed between agents in plain text | ✅ 3 leaks |
| **C3** | Tool Leakage: IBAN passed to `calculator` tool | ✅ 1 leak |
| **C4** | Logging Leakage: PII written to application logs | ✅ 1 leak |
| **C5** | Output Leakage: PII in final recommendation | ✅ 1 leak |

The **AgentLeak SDK** monitors these channels in real-time using a **Hybrid Detection Pipeline**:
1. **Tier 1**: Fast regex pattern matching
2. **Tier 2**: Presidio NER for PII entity detection  
3. **Tier 3**: LLM Judge for semantic leak classification

##  References

- [AgentLeak Paper (IEEE Access)](../../../paper/paper_revised_full.tex)
- [AgentLeak SDK Documentation](../../../docs/DOCUMENTATION.md)
- [CrewAI Framework](https://crewai.com)
