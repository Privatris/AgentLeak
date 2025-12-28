# AgentLeak

**A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems**

---

## Research Paper

**[paper.pdf](./paper.pdf)** — Read the complete research paper with methodology, results, and theoretical analysis.

The paper presents:
- Taxonomy of 19 attack classes across 5 families
- Methodology for 7-channel privacy auditing
- Evaluation on 6 production LLMs
- Multi-framework support: CrewAI, LangChain, MetaGPT, AutoGPT, AgentGPT
- Statistical analysis and vulnerability breakdown

---

## Key Results

We evaluated **6 production LLMs** across **100 scenarios** each (**600 total API calls**):

| Model | ELR ↓ | WLS ↓ | Leaks | Cost |
|-------|-------|-------|-------|------|
| **Gemini-2.5-Pro** | **20.0%** | 1.22 | 20/100 | $1.01 |
| Llama-3-70B | 27.0% | 2.60 | 27/100 | $0.03 |
| GPT-4o | 31.0% | 2.81 | 31/100 | $0.17 |
| GPT-4o-mini | 41.0% | 3.83 | 41/100 | $0.01 |
| Qwen-2.5-72B | 53.0% | 9.22 | 53/100 | $0.02 |
| Claude-3-Haiku | 53.5% | 4.24 | 53/100 | $0.03 |

**Key Findings:**
- 38% average leakage rate despite explicit privacy instructions
- Gemini-2.5-Pro shows the lowest leakage rate (20% ELR)
- 82% of leaks are semantic (not verbatim copying)
- Healthcare scenarios show lower leakage than corporate scenarios

---

## Quick Start

### Installation

```bash
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify installation
python -m agentleak --help
```

### Run Tests

```bash
# Run unit tests
pytest tests/ -v

# Quick benchmark test (1 scenario in simulation)
python scripts/quick_eval.py -n 1 -v
```

### Run Benchmark

```bash
export OPENROUTER_API_KEY="your-key"

# Quick test (10 scenarios, simulation mode)
python scripts/quick_eval.py -n 10 -v

# Full benchmark (100 scenarios)
python scripts/quick_eval.py -n 100 --real

# Professional benchmark
python scripts/full_benchmark.py --profile standard
```

---

## Project Structure

```
AgentLeak/
├── paper.pdf              # Research paper
├── agentleak/             # Core benchmark package
│   ├── cli/               # Command-line interface
│   ├── config/            # Configuration management
│   ├── core/              # Core evaluation logic
│   ├── defenses/          # Defense implementations (Sanitizer)
│   ├── detection/         # 3-tier leakage detection pipeline
│   ├── generators/        # Scenario generation
│   ├── harness/           # Framework adapters
│   ├── metrics/           # ELR, WLS, CLR metrics
│   ├── runner/            # Benchmark execution
│   ├── schemas/           # Data schemas (Scenario, Trace)
│   └── utils/             # Utilities
├── agentleak_data/        # Dataset files (1000 scenarios)
├── benchmark_results/     # Evaluation results
├── data/                  # Payloads and calibration data
├── docs/                  # Documentation
├── figures/               # Paper figures (PDF/PNG)
├── logs/                  # Execution logs
├── scripts/               # Utility scripts
└── tests/                 # Test suite (315 tests)
```

---

## Architecture

### 7 Leakage Channels (C1–C7)

AgentLeak audits **seven distinct channels** where privacy violations can occur:

| Channel | Description |
|---------|-------------|
| C1 | Final output (user-facing) |
| C2 | Inter-agent messages |
| C3 | Tool inputs (API arguments) |
| C4 | Tool outputs (API responses) |
| C5 | Memory writes (scratchpads, vector stores) |
| C6 | Logs and telemetry |
| C7 | Persisted artifacts (files, tickets) |

### 3-Tier Detection Pipeline

1. **Tier 1: CanaryMatcher** — Exact match for synthetic markers (CANARY_*)
2. **Tier 2: PatternAuditor** — Regex-based PII detection (SSN, email, phone)
3. **Tier 3: SemanticDetector** — Embedding similarity for paraphrased leaks

### 19-Class Attack Taxonomy

Organized into 5 families:
- **F1**: Prompt & Instruction Attacks (4 classes)
- **F2**: Indirect & Tool-Surface Attacks (4 classes)
- **F3**: Memory & Persistence Attacks (4 classes)
- **F4**: Multi-Agent Coordination Attacks (4 classes)
- **F5**: Reasoning & Chain-of-Thought Attacks (3 classes)

---

## Framework Support

AgentLeak provides adapters for multi-agent frameworks:

| Framework | Adapter |
|-----------|---------||
| CrewAI | `agentleak.harness.adapters.crewai_adapter` |
| LangChain | `agentleak.harness.adapters.langchain_adapter` |
| MetaGPT | `agentleak.harness.adapters.metagpt_adapter` |
| AutoGPT | `agentleak.harness.adapters.autogpt_adapter` |
| AgentGPT | `agentleak.harness.adapters.agentgpt_adapter` |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **ELR** | Exact Leakage Rate — % of scenarios with any leak |
| **WLS** | Weighted Leakage Score — severity-weighted leakage |
| **CLR** | Channel Leakage Rate — per-channel breakdown |
| **ASR** | Attack Success Rate — leakage under adversarial conditions |
| **TSR** | Task Success Rate — agent utility measure |

---

## Citation

If you use AgentLeak in your research, please cite:

```bibtex
@article{elyagoubi2025agentleak,
  title={AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems},
  author={El Yagoubi, Faouzi and Al Mallah, Ranwa and Abdi, Arslene},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
