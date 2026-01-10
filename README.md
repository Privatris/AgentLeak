# AgentLeak

**A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems**

## Overview

AgentLeak systematically audits **7 leakage channels** and tests against **19 attack classes** across **5 families**. Unlike output-only benchmarks, it captures privacy violations in inter-agent messages, tool calls, memory writes, and artifacts.

## Key Results

| Model | ELR ↓ | WLS ↓ | Leaks |
|-------|-------|-------|-------|
| **Gemini-2.5-Pro** | **20.0%** | 1.22 | 20/100 |
| Llama-3-70B | 27.0% | 2.60 | 27/100 |
| GPT-4o | 31.0% | 2.81 | 31/100 |
| GPT-4o-mini | 41.0% | 3.83 | 41/100 |
| Qwen-2.5-72B | 53.0% | 9.22 | 53/100 |
| Claude-3-Haiku | 53.5% | 4.24 | 53/100 |

- 38% average leakage rate despite explicit privacy instructions
- 82% of leaks are semantic (not verbatim copying)

## Quick Start

```bash
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Quick benchmark
export OPENROUTER_API_KEY="your-key"
python scripts/quick_eval.py -n 10 -v
```

## Project Structure

```
AgentLeak/
├── agentleak/             # Core package
│   ├── core/              # Attack taxonomy, channels
│   ├── detection/         # 3-tier detection pipeline
│   ├── generators/        # Scenario generation
│   ├── harness/           # Framework adapters
│   ├── metrics/           # ELR, WLS, CLR
│   └── schemas/           # Pydantic models
├── agentleak_data/        # Dataset (1000 scenarios)
├── scripts/               # Benchmark scripts
└── tests/                 # Test suite (315 tests)
```

## Architecture

### Leakage Channels (C1–C7)

| Channel | Description |
|---------|-------------|
| C1 | Final output |
| C2 | Inter-agent messages |
| C3 | Tool inputs |
| C4 | Tool outputs |
| C5 | Memory writes |
| C6 | Logs |
| C7 | Artifacts |

### Detection Pipeline

1. **CanaryMatcher** — Exact match for synthetic markers
2. **PatternAuditor** — Regex-based PII detection
3. **SemanticDetector** — Embedding similarity (τ=0.72)

### Attack Taxonomy

5 families, 19 classes:
- F1: Prompt Injection (4)
- F2: Tool-Surface Attacks (4)
- F3: Memory Attacks (4)
- F4: Multi-Agent Attacks (4)
- F5: Reasoning Attacks (3)

## Metrics

| Metric | Description |
|--------|-------------|
| **ELR** | Exact Leakage Rate |
| **WLS** | Weighted Leakage Score |
| **CLR** | Channel Leakage Rate |
| **TSR** | Task Success Rate |

## Citation

```bibtex
@article{elyagoubi2025agentleak,
  title={AgentLeak: A Full-Stack Benchmark for Privacy Leakage 
         in Multi-Agent LLM Systems},
  author={El Yagoubi, Faouzi and Al Mallah, Ranwa and Abdi, Arslene},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
