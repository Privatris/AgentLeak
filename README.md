# AgentLeak

Benchmark for privacy leakage in multi-agent LLM systems.

This repository accompanies the IEEE Access paper: *AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems*.

## Key Results (5,694 traces across 5 models)

| Model | C1 (Output) | C2 (Internal) | H1 (Audit Gap) | Total Leak |
|-------|-------------|---------------|----------------|------------|
| **Claude-3.5-Sonnet** | 8.2% | 53.9% | 45.7% | 55.2% |
| GPT-4o | 17.2% | 76.8% | 59.6% | 77.6% |
| GPT-4o-mini | 41.2% | 75.3% | 34.2% | 76.3% |
| Llama-3.3-70B | 26.9% | 67.8% | 41.3% | 89.9% |
| Mistral-Large | 47.5% | 96.2% | 48.7% | 99.3% |
| **Average** | **28.2%** | **74.0%** | **45.9%** | **79.7%** |

### Key Findings

- **Internal channels leak 2.6× more** than external (74.0% vs 28.2%)
- **Output-only audits miss 45.9%** of violations
- **Claude 3.5 Sonnet paradox**: Lowest C1 leakage (8.2%) but 6.6× internal/external ratio—the highest among all models
- Pattern C2 > C1 holds **across all 5 models** tested

## Scope

- 1,000 scenarios (healthcare, finance, legal, corporate)
- 7 channels: C1 output, C2 inter-agent, C3-C4 tools, C5 memory, C6 logs, C7 artifacts
- 32 attack classes, 6 families
- SDK: CrewAI, LangChain, AutoGPT, MetaGPT

## 🚀 Reproduction

To reproduce the results from the IEEE paper, including the C3/C6 secondary channel analysis:

```bash
cd benchmarks/ieee_repro
# Follow instructions in benchmarks/ieee_repro/README.md
```

## Structure

- `agentleak/`: The core framework SDK
- `agentleak_data/`: The dataset of 1000 scenarios
- `benchmarks/ieee_repro/`: Scripts to reproduce the paper's findings, including Finding 7 (Tools & Logs).
- `benchmarks/showcase/`: Real-world CrewAI integration demo showing the SDK in action.
- `paper/`: The LaTeX source of the IEEE Access paper

## Setup

```bash
git clone https://github.com/Privatris/AgentLeak
cd AgentLeak
pip install -e .
pytest tests/ -v
```

## Usage

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(mode=DetectionMode.HYBRID)
result = tester.check(
    vault={"ssn": "123-45-6789"},
    output="The SSN is 123-45-6789",
    channel="C1"
)
print(f"Leak: {result.leaked}, Confidence: {result.confidence}")
```

CLI:
```bash
python -m agentleak run --quick --dry-run
python -m agentleak run --full
```

## Reproduction

```bash
cd benchmarks/ieee_repro
python benchmark.py --n 100 --traces --model openai/gpt-4o-mini
```

Traces are in `benchmarks/ieee_repro/results/traces/`.

## Citation

```bibtex
@article{agentleak2026,
  title={AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems},
  author={El Yagoubi, Faouzi and Al Mallah, Ranwa},
  journal={IEEE Access},
  year={2026}
}
```

## License

MIT
