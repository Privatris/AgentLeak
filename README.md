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
- **Finding 7 (Tool Leakage)**: Tool inputs (C3) and system logs (C6) exhibit extremely high leakage rates (up to **85%** on Claude 3.5), even when the final agent output (C1) is perfectly sanitized.
- Pattern C2 > C1 holds **across all 5 models** tested

## Scope

- 1,000 scenarios (healthcare, finance, legal, corporate)
- 7 channels: C1 output, C2 inter-agent, C3-C4 tools, C5 memory, C6 logs, C7 artifacts
- 32 attack classes, 6 families
- SDK: CrewAI, LangChain, AutoGPT, MetaGPT

## Reproduction

### Main Benchmark (C1, C2, C5)
To reproduce the main results (Output, Internal, Memory):
```bash
cd benchmarks/ieee_repro
python benchmark.py --n 1000 --traces --model openai/gpt-4o
```

### Advanced Tools & Logs Benchmark (C3, C6)
Targets "Secondary Channel" leakage where sensitive data is sent to external tools or dumped in logs.
```bash
cd benchmarks/ieee_repro
# Run for a specific model (e.g., Claude 3.5)
python benchmark_tools.py --n 100 --model anthropic/claude-3.5-sonnet

# Or run the automated multi-model test suite
./run_tools_benchmark.sh
```
Results are saved in `benchmarks/ieee_repro/results/tools/`.

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
@article{el2026agentleak,
  title        = {AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems},
  author       = {El Yagoubi, Faouzi and Badu-Marfo, Godwin and Al Mallah, Ranwa},
  journal      = {arXiv preprint arXiv:2602.11510},
  year         = {2026},
  url          = {https://arxiv.org/abs/2602.11510},
  abstract     = {Multi-agent Large Language Model (LLM) systems create privacy risks that current benchmarks cannot measure. When agents coordinate on tasks, sensitive data passes through inter-agent messages, shared memory, and tool arguments, pathways that output-only audits never inspect. We introduce AgentLeak, the first full-stack benchmark for privacy leakage covering internal channels, spanning 1,000 scenarios across healthcare, finance, legal, and corporate domains, paired with a 32-class attack taxonomy and a three-tier detection pipeline. Testing several models across thousands of traces shows that internal channels in multi-agent configurations are the primary privacy vulnerability and that output-only audits miss a large fraction of violations, underscoring the need for coordinated privacy protections on inter-agent communication.},
  note         = {Submitted to arXiv on 12 Feb 2026.},
}
```

## License

MIT
