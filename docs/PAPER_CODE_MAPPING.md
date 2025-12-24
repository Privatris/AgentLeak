# 📊 Paper ↔ Code Mapping

This document provides a detailed mapping between the claims, tables, and figures in our NeurIPS 2025 paper and the corresponding code implementations.

---

## Contributions Mapping

### C1: Benchmark at Scale (1000 Scenarios)

| Paper Section | Code Files |
|---------------|------------|
| §4.1 Scenario Template | `apb/schemas/scenario.py` |
| §4.2 Verticals and Task Families | `apb/generators/scenario_generator.py` |
| §4.3 Controlled Realism | `apb/generators/canary_generator.py` |
| §4.4 Scenario Distribution | `apb/generators/distribution.py` |

**Key Classes:**
- `Scenario` - Main scenario data structure
- `PrivateVault` - Container for sensitive records
- `AllowedSet` - Ground-truth permitted disclosures
- `ScenarioGenerator` - Generates scenarios per vertical

```python
# Generate 100 healthcare scenarios
from apb.generators import ScenarioGenerator
from apb.schemas.scenario import Vertical

gen = ScenarioGenerator(seed=42)
scenarios = gen.generate_batch(100, vertical=Vertical.HEALTHCARE)
```

---

### C2: 15-Class Attack Taxonomy

| Paper Section | Code Files |
|---------------|------------|
| §5 Attack Taxonomy | `apb/attacks/attack_module.py` |
| Table 2: Attack Classes | `apb/attacks/attack_module.py::ATTACK_REGISTRY` |
| §5.1 Attack Implementation | `apb/attacks/payloads/` |

**Attack Families (Table 2):**

| Family | Code Enum | Attack Classes |
|--------|-----------|----------------|
| F1: Prompt | `AttackFamily.PROMPT` | `DPI`, `ROLE_CONFUSION`, `CONTEXT_OVERRIDE`, `FORMAT_COERCION` |
| F2: Tool-Surface | `AttackFamily.TOOL` | `IPI`, `TOOL_POISONING`, `RAG_BAIT`, `LINK_EXFIL` |
| F3: Memory | `AttackFamily.MEMORY` | `MEMORY_EXFIL`, `VECTOR_LEAK`, `LOG_LEAK`, `ARTIFACT_LEAK` |
| F4: Multi-Agent | `AttackFamily.MULTI_AGENT` | `CROSS_AGENT`, `ROLE_BOUNDARY`, `DELEGATION_EXPLOIT` |

```python
# Get all attacks for a family
from apb.attacks import AttackModule, AttackFamily

attacks = AttackModule.get_attacks_by_family(AttackFamily.TOOL)
```

---

### C3: Framework-Agnostic Harness

| Paper Section | Code Files |
|---------------|------------|
| §6 Evaluation Harness | `apb/harness/base_adapter.py` |
| §6.1 Unified Trace Format | `apb/schemas/trace.py` |
| §6.2 Framework Adapters | `apb/harness/adapters/` |
| Table 3: Adapters | `apb/harness/adapters/__init__.py` |

**Supported Frameworks:**

| Framework | Adapter Class | Config Class |
|-----------|---------------|--------------|
| LangChain | `LangChainAdapter` | `LangChainConfig` |
| CrewAI | `CrewAIAdapter` | `CrewAIConfig` |
| AutoGPT | `AutoGPTAdapter` | `AutoGPTConfig` |
| MetaGPT | `MetaGPTAdapter` | `MetaGPTConfig` |
| AgentGPT | `AgentGPTAdapter` | `AgentGPTConfig` |

```python
# Use LangChain adapter
from apb.harness.adapters import LangChainAdapter, LangChainConfig

config = LangChainConfig(model_name="gpt-4")
adapter = LangChainAdapter(config)
result = adapter.execute(scenario)
```

---

### C4: Metrics + Detection Pipeline

| Paper Section | Code Files |
|---------------|------------|
| §7.1 Utility Metrics | `apb/metrics/utility_metrics.py` |
| §7.2 Leakage Metrics | `apb/metrics/leakage_metrics.py` |
| §7.3 Detection Pipeline | `apb/detection/pipeline.py` |
| §7.4 Detection Calibration | `apb/detection/calibration.py` |
| §7.5 Pareto Frontier | `apb/metrics/pareto.py` |

**Metrics Implementation (Equations 1-3):**

| Equation | Function | File |
|----------|----------|------|
| Eq. 1: ELR | `compute_elr()` | `apb/metrics/leakage_metrics.py` |
| Eq. 2: WLS | `compute_wls()` | `apb/metrics/leakage_metrics.py` |
| Eq. 3: CLR | `compute_clr()` | `apb/metrics/leakage_metrics.py` |

```python
# Compute metrics
from apb.metrics import compute_elr, compute_wls, compute_clr

elr = compute_elr(scenarios, traces)
wls = compute_wls(scenarios, traces, severity_weights)
clr = compute_clr(scenarios, traces, channel=Channel.TOOL_INPUT)
```

---

### C5: LCF Defense

| Paper Section | Code Files |
|---------------|------------|
| §8.3 LCF Description | `apb/defenses/lcf_defense.py` |
| LEACE Projection | `apb/defenses/leace.py` |
| Variance Budget | `apb/defenses/budget_tracker.py` |

```python
# Apply LCF defense
from apb.defenses import LCFDefense

lcf = LCFDefense(lambda_strength=0.5)
protected_embedding = lcf.apply(embedding, sensitive_attributes)
```

---

## Tables Reproduction

### Table 4: Per-Vertical Statistics

```bash
python scripts/run_benchmark.py --stats-only --output-format latex
```

**Code:** `scripts/run_benchmark.py::compute_vertical_stats()`

### Table 5: Benign Results (A0)

```bash
python scripts/run_benchmark.py --setting A0 --output results/table5.csv
```

**Code:** `scripts/run_benchmark.py::run_benign_evaluation()`

### Table 6: Adversarial Results (A2)

```bash
python scripts/run_benchmark.py --setting A2 --output results/table6.csv
```

**Code:** `scripts/run_benchmark.py::run_adversarial_evaluation()`

### Table 7: Channel Leakage Rates

```bash
python scripts/run_benchmark.py --channel-breakdown --output results/table7.csv
```

**Code:** `scripts/run_benchmark.py::compute_channel_breakdown()`

### Table 8: Defense Comparison

```bash
python scripts/run_benchmark.py --defenses all --output results/table8.csv
```

**Code:** `scripts/run_benchmark.py::compare_defenses()`

---

## Figures Reproduction

### Figure 1: System Overview

**Manual diagram** - see `figures/system_overview.pdf`

### Figure 2: Pareto Frontier

```bash
python scripts/generate_figures.py --figure pareto --output figures/pareto.pdf
```

**Code:** `scripts/generate_figures.py::plot_pareto_frontier()`

### Figure 3: Attack × Channel Heatmap

```bash
python scripts/generate_figures.py --figure heatmap --output figures/heatmap.pdf
```

**Code:** `scripts/generate_figures.py::plot_attack_channel_heatmap()`

---

## 3-Tier Canary System

| Tier | Name | Detection | Code |
|------|------|-----------|------|
| T1 | Obvious | Exact regex | `detection/canary_detector.py::detect_obvious()` |
| T2 | Realistic | Pattern match | `detection/canary_detector.py::detect_realistic()` |
| T3 | Semantic | Embedding similarity | `detection/canary_detector.py::detect_semantic()` |

```python
# Generate tiered canaries
from apb.generators import CanaryGenerator, CanaryTier

gen = CanaryGenerator()
obvious = gen.generate(CanaryTier.OBVIOUS, field_type="ssn")
realistic = gen.generate(CanaryTier.REALISTIC, field_type="ssn")
semantic = gen.generate(CanaryTier.SEMANTIC, field_type="diagnosis")
```

---

## PrivacyLens-Inspired Components

We integrated best practices from PrivacyLens (NeurIPS 2024):

| PrivacyLens Concept | APB Implementation |
|---------------------|-------------------|
| Contextual Integrity Seeds | `apb/generators/contextual_integrity.py` |
| Seed→Vignette Generation | `apb/generators/vignette_generator.py` |
| SurgeryKit Refinement | `apb/generators/vignette_generator.py::SurgeryKitModule` |
| Multi-Level Probing | `apb/detection/probing_evaluation.py` |
| Two-Stage Leakage Detection | `apb/detection/leakage_detector.py` |
| API Usage Tracking | `apb/utils/api_tracker.py` |

---

## Running Full Reproduction

```bash
# 1. Install dependencies
pip install -e .

# 2. Run tests to verify setup
pytest tests/ -v

# 3. Generate APB-Lite dataset
python -m apb.cli generate --preset lite

# 4. Run benchmark (simulation mode)
python scripts/run_benchmark.py --quick

# 5. Generate all tables (requires API key for real mode)
export OPENAI_API_KEY=sk-...
python scripts/run_benchmark.py --full --output-format latex
```

---

## Test Coverage

| Component | Test File | Tests |
|-----------|-----------|-------|
| Schemas | `tests/test_schemas.py` | 48 |
| Generators | `tests/test_generators.py` | 35 |
| Detection | `tests/test_detection.py` | 28 |
| Attacks | `tests/test_attacks.py` | 22 |
| Harness | `tests/test_harness.py` | 18 |
| PrivacyLens | `tests/test_privacylens_integration.py` | 43 |
| **Total** | | **255** |

```bash
# Run all tests
pytest tests/ -v --tb=short
```
