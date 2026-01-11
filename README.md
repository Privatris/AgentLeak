# AgentLeak

**A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AgentLeak is a comprehensive benchmark for evaluating privacy vulnerabilities in LLM-based agent systems. Unlike output-only benchmarks, it systematically audits **7 leakage channels** including internal agent communication that existing defenses fail to protect.

### Key Findings

| Finding | Evidence |
|---------|----------|
| **Multi-agent > Single-agent** | 36.7% vs 16.0% leak rate (2.3× increase) |
| **Internal channels unprotected** | 31.5% vs 3.8% leak rate (8.3× higher) |
| **Sanitizer gap** | 98% effective on output, 0% on inter-agent |
| **All frameworks vulnerable** | CrewAI, LangChain, AutoGPT, MetaGPT |

### Features

- **7 Leakage Channels** (C1-C7): Final output, inter-agent, tool I/O, memory, logs, artifacts
- **19 Attack Classes** in 5 families: Prompt, Tool, Memory, Multi-Agent, Reasoning
- **1,000 Scenarios** across 4 verticals: Healthcare, Finance, Legal, Corporate
- **3-Tier Detection**: Canary markers, pattern extraction, semantic similarity
- **Framework-Agnostic**: Works with LangChain, CrewAI, AutoGPT, MetaGPT
- **Defense Evaluation**: Sanitizer, privacy prompt, chain-of-thought

## Installation

```bash
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak
pip install -e .
```

## Quick Start

### For Reviewers: Reproduce Paper Results
```bash
# Quick reproduction (~$2.50, ~15 min)
python -m agentleak benchmark --reproduce-paper

# Full paper benchmark (~$35, ~2 hours)
python -m agentleak benchmark --full-paper
```

### Custom Benchmark
```bash
# Basic test (10 scenarios, single model)
python -m agentleak benchmark --n 10 --models gpt-4o-mini

# Dry run (no API calls, validate config)
python -m agentleak benchmark --dry-run --n 5

# Compare agents vs multi-agent
python -m agentleak benchmark --compare agents --n 20

# Test internal channels (C2, C5)
python -m agentleak benchmark --internal-channels

# Test specific channels and defenses
python -m agentleak benchmark --channels C1,C2,C3 --defenses none,cot,sanitizer
```

### API Key Setup
```bash
# Option 1: Environment variable
export OPENROUTER_API_KEY=your_key_here

# Option 2: .env file (auto-saved on first run)
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

## Benchmark Results

### Architecture Comparison

| Architecture | Tests | Leak Rate | Primary Channels |
|--------------|-------|-----------|------------------|
| Single-agent | 400 | 16.0% | C1, C3 |
| Multi-agent (2) | 350 | 32.0% | C2, C5, C1 |
| Multi-agent (3+) | 250 | 43.2% | C2, C5 |

### Channel Analysis

| Type | Channels | Leak Rate | Defenses |
|------|----------|-----------|----------|
| External | C1, C3, C4, C6, C7 | 3.8% | Sanitizer, Prompt |
| **Internal** | **C2, C5** | **31.5%** | **NONE** |

### Defense Effectiveness

| Defense | C1 (External) | C2/C5 (Internal) |
|---------|---------------|------------------|
| None | 48% | 31% |
| Privacy Prompt | 19% | 29% |
| Sanitizer | **1%** | **31%** |

## Project Structure

```
agentleak/           # Main package
├── cli/             # Command-line interface
├── core/            # Attack taxonomy, channels
├── defenses/        # Defense implementations
├── detection/       # Leakage detection
├── harness/         # Framework adapters
└── metrics/         # Evaluation metrics

agentleak_data/      # Benchmark data (1000 scenarios)
benchmark_results/   # Evaluation results
paper/               # LaTeX paper source
scripts/             # Utility scripts
tests/               # Test suite
```

## Citation

```bibtex
@inproceedings{agentleak2026,
  title={AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems},
  author={El Yagoubi, Faouzi and Al Mallah, Ranwa and Abdi, Arslene},
  booktitle={Proceedings of ...},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
