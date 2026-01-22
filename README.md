# AgentLeak

**Benchmark pour l'analyse des fuites de données dans les systèmes multi-agents LLM**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Contexte

Les systèmes multi-agents basés sur les LLM présentent des vulnérabilités de confidentialité qui passent souvent inaperçues. AgentLeak propose une méthodologie systématique pour auditer ces systèmes en analysant **7 canaux de fuite** — y compris les communications internes entre agents que les mécanismes de défense actuels ne protègent pas.

Ce travail accompagne notre article soumis à IEEE Access : *AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems*.

---

## Principaux résultats

| Observation | Données | Significativité |
|-------------|---------|-----------------|
| Pénalité multi-agent | Taux de fuite 36.7% vs 16.0% | χ² = 49.4, p < 0.001 |
| Écart canaux internes | Taux de fuite 31.5% vs 3.8% | χ² = 89.7, p < 0.001 |
| Asymétrie des défenses | 98% efficace sur C1, 0% sur C2/C5 | Vérifié sur 600 tests |
| Vulnérabilité universelle | 28-35% de fuites internes | CrewAI, LangChain, AutoGPT, MetaGPT |

---

## Fonctionnalités

### Périmètre du benchmark
- **1 000 scénarios** couvrant 4 secteurs (santé, finance, juridique, entreprise)
- **7 canaux de fuite** (C1-C7) : sortie finale, inter-agent, entrées/sorties d'outils, mémoire, logs, artefacts
- **32 classes d'attaques** organisées en 6 familles (F1-F6)
- **3 niveaux d'adversaire** (A0-A2) : passif, utilisateur, développeur


# AgentLeak: Benchmark for Data Leakage in Multi-Agent LLM Systems

**AgentLeak** is an open-source benchmark for evaluating data leakage in multi-agent systems based on large language models (LLMs). It measures the ability of defenses to limit the disclosure of sensitive information across different channels and attack scenarios.

![Benchmark](paper/figures/benchmark_overview.png)

## Main Features

- **32 attack families** covering major leakage surfaces
- **7 leakage channels** (output, memory, logs, artifacts, etc.)
- **Hybrid detection** (PII, LLM-as-Judge, rules)
- **Integrated and extensible defense mechanisms**
- **SDK integrations** (LangChain, CrewAI, AutoGPT, MetaGPT)
- **Comprehensive test suite** (pytest)

## Installation

### Prerequisites

- Python 3.10+
- `pip install -e .`

### Quick Installation

```bash
git clone https://github.com/agentleak/agentleak.git
cd agentleak
pip install -e .
```

### Test the Installation

```bash
pytest tests/ -v --tb=short
```

---

## Usage

### Run the Benchmark

```bash
# Quick test (validation)
python -m agentleak run --quick --dry-run

# Full benchmark
python -m agentleak run --full

# Filter by attack family
python -m agentleak run --attack-family F4 --limit 50

# With defenses enabled
python -m agentleak run --defense D1
```

### Reproduce Results

The `experiments/all_to_all/` folder contains scripts to validate the claims of the paper:

```bash
cd experiments/all_to_all

# Quick test (smoke test)
python smoke_test.py --claims "1,2,3" --scenarios 10

# Full benchmark
python master_benchmark.py --mode full
```

### Programmatic Usage

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(mode=DetectionMode.HYBRID)

result = tester.check(
    vault={"ssn": "123-45-6789", "email": "patient@hospital.com"},
    output="The patient whose SSN is 123-45-6789 has been treated.",
    channel="C1"
)

print(f"Leak detected: {result.leaked}")
print(f"Confidence: {result.confidence}")
```

---

## Project Structure

```
AgentLeak/
├── agentleak/                  # Main package
│   ├── catalog/                # Canonical definitions
│   │   ├── attacks.py          # 32 attack classes
│   │   ├── defenses.py         # Defense implementations
│   │   └── channels.py         # 7 leakage channels
│   ├── detection/              # Detection pipeline
│   │   ├── basic_detectors.py  # Tier 1-2 detectors
│   │   ├── llm_judge.py        # LLM-as-Judge
│   │   └── presidio_detector.py# PII detection
│   ├── defenses/               # Defense mechanisms
│   ├── integrations/           # Framework integrations
│   └── metrics/                # Evaluation metrics
│
├── agentleak_data/             # Benchmark data
│   ├── datasets/               # Scenarios
│   └── prompts/                # System prompts
│
├── experiments/                # Validation scripts
│   └── all_to_all/             # Full benchmark
│
├── tests/                      # Test suite
├── docs/                       # Detailed documentation
└── paper/                      # IEEE paper sources
```

---

## Benchmark Results

### Architectural Comparison

| Architecture      | Tests | Leakage Rate | 95% CI         |
|-------------------|-------|--------------|----------------|
| Single agent      | 400   | 16.0%        | [12.9%, 19.7%] |
| Multi-agent (2)   | 350   | 32.0%        | [27.3%, 37.1%] |
| Multi-agent (3+)  | 250   | 43.2%        | [37.1%, 49.5%] |

### Channel Analysis

| Channel           | Type    | Leakage Rate | Defense Effectiveness |
|-------------------|---------|--------------|----------------------|
| C1 (Final output) | External| 4.8%         | 98%                  |
| C2 (Inter-agent)  | Internal| 31.0%        | 0%                   |
| C3 (Tool input)   | External| 3.7%         | 85%                  |
| C4 (Tool output)  | External| 3.2%         | 80%                  |
| C5 (Memory)       | Internal| 32.0%        | 0%                   |
| C6 (Logs)         | External| 3.2%         | 90%                  |
| C7 (Artifacts)    | External| 4.2%         | 75%                  |

### Attack Family Performance

| Family | Name                  | Success Rate | Preferred Channel |
|--------|-----------------------|--------------|------------------|
| F1     | Prompt & Instruction  | 62.2%        | C1, C2           |
| F2     | Tool Surface          | 71.7%        | C3, C4           |
| F3     | Memory & Persistence  | 62.7%        | C5               |
| F4     | Multi-agent Coordination| 80.0%      | C2               |
| F5     | Reasoning & CoT       | 62.7%        | C2, C5           |
| F6     | Evasion & Obfuscation | 55.0%        | C1               |

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

MIT — see [LICENSE](LICENSE).

---

## Contribution

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
pip install -e ".[dev]"
pytest tests/ -v
black agentleak/ tests/
```

---

## Contribution

Les contributions sont bienvenues. Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

```bash
pip install -e ".[dev]"
pytest tests/ -v
black agentleak/ tests/
```
