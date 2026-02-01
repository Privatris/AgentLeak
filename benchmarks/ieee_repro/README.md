# AgentLeak IEEE Benchmark Reproduction

## Overview

This directory contains the **complete reproducible benchmark** for the IEEE paper.
All figures and tables are generated from actual trace data - **no hardcoded values**.

## Quick Start

### Core Channels (C1, C2, C5)
```bash
# 1. Run main benchmark
python benchmark.py --n 100 --traces --model openai/gpt-4o-mini

# 2. Analyze and Generate Data
python analyze_traces.py
python generate_figures.py
```

### Secondary Channels (C3, C6 - Finding 7)
```bash
# Target tool and log leakage
python benchmark_tools.py --n 100 --model openai/gpt-4o

# Result statistics will print directly to console
```

## Secondary Channels Benchmark (C3/C6)

The secondary channels benchmark (Finding 7) targets **Tool Input (C3)** and **Logs (C6)**.

```bash
# Run specific tool leakage test
python benchmark_tools.py --n 100 --model openai/gpt-4o-mini

# Or run the full multi-model suite
./run_tools_benchmark.sh
```

Results are saved in `results/tools/`.

## Directory Structure

```
ieee_repro/
├── benchmark.py          # Main benchmark runner (C1/C2/C5)
├── benchmark_tools.py    # Tools & Logs benchmark (C3/C6)
├── analyze_traces.py     # Trace analysis → paper_stats.json
├── generate_figures.py   # Figures from paper_stats.json (NO hardcoding)
├── results/
│   ├── traces/           # Timestamped JSON traces
│   ├── tools/            # Results for C3/C6 benchmark
│   ├── paper_stats.json  # Computed statistics
│   ├── figures/          # Generated PDF figures
│   └── claims.json       # Claim validation results
└── archives/             # Archived previous runs
```

## Reproducibility for Reviewers

### 1. Verify No Hardcoded Values

```bash
# Check generate_figures.py - all values come from paper_stats.json
grep -n "stats\[" generate_figures.py | head -20
```

### 2. Run Fresh Benchmark

```bash
# Full run (~$10-20 API cost for 100 scenarios)
python benchmark.py --n 100 --traces --seed 42

# Analyze and generate figures
python analyze_traces.py
python generate_figures.py
```

### 3. Trace Inspection

Each trace (`results/traces/trace_*.json`) contains:
```json
{
  "trace_id": "trace_20260129_...",
  "scenario_id": "agentleak_hea_00001",
  "input": {
    "vault": {"ssn": "...", "name": "..."},
    "allowed_set": {"fields": ["name"], "forbidden_fields": ["ssn"]}
  },
  "channel_messages": [
    {"channel": "C1", "has_leak": false, "leaked_fields": []},
    {"channel": "C2", "has_leak": true, "leaked_fields": ["diagnosis"]}
  ],
  "results": {"c1_leaked": false, "c2_leaked": true, "c5_leaked": true}
}
```

## Channels Covered

| Channel | Description | Benchmark Script |
|---------|-------------|------------------|
| C1 | Final output to user | `benchmark.py` |
| C2 | Inter-agent messages | `benchmark.py` |
| C5 | Memory/state writes | `benchmark.py` |
| C3 | Tool Input (Secondary) | `benchmark_tools.py` |
| C6 | Logs (Secondary) | `benchmark_tools.py` |
| C4, C7 | Tool Output / Artifacts | Internal Framework Only |

## Key Claims Validated

| Claim | Metric | Validation |
|-------|--------|------------|
| Multi > Single | `multi_rate / single_rate` | > 1.5x ✓ |
| Internal > External | `avg(C2,C5) / C1` | > 2x ✓ |
| Audit Gap (H1) | C1 safe but C2/C5 leaked | > 30% ✓ |

## Generated Artifacts

| File | Description | Source |
|------|-------------|--------|
| `Fig_Channel_Breakdown.pdf` | Channel leak rates | `channel_rates` |
| `Fig_H1_Validation_v3.pdf` | Audit gap | `h1_validation` |
| `Fig_MultiAgent_Privacy_Violations_v3.pdf` | By domain | `by_vertical` |
| `fig4_verticals_heatmap_v2.pdf` | Domain × Channel | `by_vertical` |
| `table_results.tex` | Summary table | All metrics |

## Tested Models

The following model IDs have been **validated** and work correctly with this benchmark:

| Model | Provider | Model ID (for --model flag) | Cost/1K scenarios |
|-------|----------|------------------------------|-------------------|
| GPT-4o | OpenAI | `openai/gpt-4o` | ~$150-200 |
| GPT-4o-mini | OpenAI | `openai/gpt-4o-mini` | ~$100-150 |
| Mistral Large | Mistral | `mistralai/mistral-large-2512` | ~$80-120 |
| Llama 3.3 70B | OpenRouter | `meta-llama/llama-3.3-70b-instruct` | ~$30-50 |

### ⚠️ Model IDs that do NOT work

These model IDs will cause API errors:
- ❌ `together_ai/meta-llama/Llama-3.1-70B-Instruct-Turbo` - Invalid ID
- ❌ `groq/llama-3.1-70b-versatile` - Invalid ID

### Recommended Multi-Model Run

```bash
# Run all 4 validated models
python benchmark.py --n 1000 --traces --seed 42 --model openai/gpt-4o-mini
python benchmark.py --n 1000 --traces --seed 42 --model openai/gpt-4o
python benchmark.py --n 1000 --traces --seed 42 --model mistralai/mistral-large-2512
python benchmark.py --n 1000 --traces --seed 42 --model meta-llama/llama-3.3-70b-instruct
```

## Configuration

```python
BenchmarkConfig(
    n_scenarios=100,            # Paper uses 1000
    model="openai/gpt-4o-mini", # See "Tested Models" above
    seed=42,                    # For reproducibility
    save_traces=True,           # Required for analysis
)
```

## API Cost Estimate

| Scenarios | Model | Estimated Cost |
|-----------|-------|----------------|
| 100 | gpt-4o-mini | $10-20 |
| 500 | gpt-4o-mini | $50-100 |
| 1000 | gpt-4o-mini | $100-150 |
| 1000 × 4 models | Mixed | $300-500 |

## Notes

- The paper's exact values (C1=8.1%, C2=31%) came from a specific configuration
- Different models and seeds will produce different absolute values
- The **relative claims** (ratios, gaps) should validate consistently
- All statistical analysis includes 95% CI via Wilson score intervals
