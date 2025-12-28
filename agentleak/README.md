# 🔒 AgentLeak

**A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📋 Overview

AgentLeak is a comprehensive benchmark for evaluating privacy leakage in multi-agent LLM systems. Unlike traditional benchmarks that only inspect final outputs, AgentLeak systematically audits **7 distinct leakage channels** and tests against **19 attack classes** organized in **5 families**.

> *"LLM agents have evolved beyond simple chatbots. They call external tools, maintain persistent memory, and coordinate with other agents—expanding the attack surface significantly."*

### The Problem

Privacy breaches occur through intermediate channels invisible to traditional output-only audits. A scheduling agent might:
- Return a sanitized confirmation to the user ✓
- But simultaneously copy patient medical history to a calendar API ✗
- Store sensitive data in shared memory ✗
- Log SSN in telemetry traces ✗

**Existing benchmarks miss 6 out of 7 leakage channels.**

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **7 Leakage Channels** | Final output, inter-agent, tool I/O, memory, logs, artifacts |
| **19 Attack Classes** | From prompt injection to multi-agent collusion |
| **1,000 Scenarios** | Across healthcare, finance, legal, and corporate domains |
| **3-Tier Detection** | Canary matching, pattern extraction, semantic similarity |
| **Framework-Agnostic** | Works with LangChain, CrewAI, AutoGPT, MetaGPT |
| **Pareto Analysis** | Privacy-utility tradeoff measurement |

## 📡 The 7 Leakage Channels (C1-C7)

```
C1: Final Output      │ User-visible responses (traditional benchmark focus)
C2: Inter-Agent       │ Messages between coordinated agents  
C3: Tool Inputs       │ Arguments passed to external APIs
C4: Tool Outputs      │ Data returned from APIs
C5: Memory Writes     │ Persistent scratchpads, vector stores
C6: Logs & Telemetry  │ Framework-level execution traces
C7: Artifacts         │ Generated files, tickets, emails
```

## ⚔️ The 19-Class Attack Taxonomy

| Family | Attacks | Target Channels |
|--------|---------|-----------------|
| **F1: Prompt & Instruction** | DPI, Role Confusion, Context Override, Format Coercion | C1, C2, C3 |
| **F2: Tool Surface** | IPI, Tool Poisoning, RAG Bait, Link Exfiltration | C1, C3, C5 |
| **F3: Memory & Persistence** | Memory Exfil, Vector Leak, Log Leak, Artifact Leak | C5, C6, C7 |
| **F4: Multi-Agent** | Cross-Agent Collusion, Role Boundary, Delegation Exploit, Shared Memory | C2, C5 |
| **F5: Reasoning/CoT** | Logic Puzzle Jailbreak, CoT Forging, Special Token Injection | C1 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/agentleak/agentleak.git
cd agentleak

# Install dependencies
pip install -e .

# Or just install core packages
pip install rich pyyaml openai pydantic
```

### Set API Key

```bash
export OPENROUTER_API_KEY="your-api-key"
```

### Run Interactive CLI

```bash
# Interactive mode
python -m agentleak

# Quick test (10 scenarios)
python agentleak/quick_test.py --model gpt-4o-mini --scenarios 10

# Model comparison
python agentleak/compare_models.py --models gpt-4o-mini,claude-3-haiku --scenarios 50
```

## 📊 Metrics

### Utility Metrics
- **TSR (Task Success Rate)**: Fraction of scenarios where the objective is met

### Leakage Metrics
- **ELR (Exact Leak Rate)**: Fraction of scenarios with any leakage
- **WLS (Weighted Leak Score)**: Severity-weighted leakage (SSN=5.0, postal code=1.0)
- **CLR (Channel Leak Rate)**: Per-channel leakage breakdown
- **ASR (Attack Success Rate)**: Leakage under adversarial conditions

### Detection Pipeline

```
Stage 1: Exact Canary Matching  │ Regex for unique markers
Stage 2: Structured Field Audit │ Template-based extraction (SSN, CC, etc.)
Stage 3: Semantic Similarity    │ Embedding-based paraphrase detection
```

## 🏥 Verticals (250 scenarios each)

| Vertical | Representative Tasks |
|----------|---------------------|
| **Healthcare** | Patient scheduling, claim verification, referral coordination |
| **Finance** | KYC/AML workflows, portfolio reporting, transaction disputes |
| **Legal** | Contract review, discovery triage, privileged email drafting |
| **Corporate** | Incident response, HR case handling, trade secret routing |

## 🛡️ Baselines & Defenses

### No-Defense Baselines
- **Baseline**: Framework defaults, no privacy instructions
- **Policy Prompts**: System-level privacy guidance
- **Role Separation**: Multi-agent clearance levels

### Implemented Defenses
- **Output Filtering**: PII scrubbers (regex + NER) on final outputs
- **Input Sanitization**: Filter sensitive data before agent execution
- **System Prompt Hardening**: Privacy-aware prompts with CoT reasoning
- **Memory Minimization**: Disable persistent memory (config option)

### External Guardrails (for benchmarking)
- PromptGuard (Meta)
- NeMo Guardrails (NVIDIA)
- LlamaGuard 3 (Meta)
- Lakera Guard (commercial)
- Rebuff (open-source)

## 📁 Project Structure

```
agentleak/
├── core/           # Channels (C1-C7), Attacks (19), Scenarios
├── schemas/        # Pydantic models (Scenario, Trace, Results)
├── generators/     # Data and scenario generation
├── detection/      # Multi-tier leakage detection
├── defenses/       # Privacy defenses and mitigations
├── metrics/        # ELR, WLS, TSR computation
├── harness/        # Framework adapters (LangChain, CrewAI, etc.)
├── cli/            # Interactive CLI with Rich
├── config/         # YAML configuration
└── runner/         # Test execution
```

## 🔬 Reproducibility

We release three dataset variants:

| Variant | Scenarios | Use Case | Correlation |
|---------|-----------|----------|-------------|
| **Lite** | 100 | CI/CD validation, rapid iteration | r=0.94 |
| **Medium** | 250 | Academic research | r=0.97 |
| **Full** | 1,000 | Publication-ready evaluation | - |

## 📈 Example Results

| Model | ELR | TSR | Dominant Channel |
|-------|-----|-----|------------------|
| GPT-4o | 14% | 92% | C2 (Inter-Agent) |
| Claude-3-Haiku | 28% | 88% | C1 (Final Output) |
| Gemini-2.0-Flash | 31% | 85% | C2 (Inter-Agent) |
| Qwen-2.5-72B | 43% | 78% | C1 (Final Output) |

*Results from hardened subset evaluation. See paper for full methodology.*

## 📖 Citation

```bibtex
@article{elyagoubi2025agentleak,
  title={AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems},
  author={El Yagoubi, Faouzi and Al Mallah, Ranwa and Abdi, Arslene},
  journal={arXiv preprint},
  year={2025}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 👥 Authors

- **Faouzi EL YAGOUBI** - Polytechnique Montréal
- **Ranwa AL MALLAH** - Polytechnique Montréal  
- **Arslene ABDI** - Publicis Ressources

## 🔗 Links

- [Paper (arXiv)](https://arxiv.org/abs/xxxx.xxxxx)
- [Documentation](docs/)
- [Issue Tracker](https://github.com/agentleak/agentleak/issues)
