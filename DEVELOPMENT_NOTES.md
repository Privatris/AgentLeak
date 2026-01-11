# AgentLeak Development Notes

**Last Updated:** 2026-01-11

## Project Overview

AgentLeak is a privacy leakage benchmark for multi-agent LLM systems.

## Key Claims (Paper)

1. **Multi-agent > Single-agent**: 36.7% vs 16.0% leak rate (2.3× increase)
2. **Internal channels unprotected**: C2/C5 have 31.5% leak rate vs 3.8% external (8.3×)
3. **Defenses fail on internal**: Sanitizer 98% on C1, 0% on C2/C5
4. **All frameworks vulnerable**: CrewAI, LangChain, AutoGPT, MetaGPT (28-35%)

## Repository Structure

```
agentleak/           # Main package
├── cli/             # Command-line interface
├── core/            # Attack taxonomy, channels, scenarios
├── defenses/        # Defense implementations
├── detection/       # Leakage detection pipeline
├── generators/      # Scenario generators
├── harness/         # Framework adapters
├── metrics/         # Evaluation metrics
├── runner/          # Benchmark runner
├── schemas/         # Data schemas
└── utils/           # Utilities

agentleak_data/      # Benchmark data
├── agentleak_1000.jsonl      # 1000 scenarios
├── agentleak_1000_summary.json
├── llm_judge_prompt.txt      # LLM-as-judge prompt
├── scenario_example.jsonl    # Example scenario
└── trace_sample.jsonl        # Example trace

benchmark_results/   # Results
├── FINAL_RESULTS.json
├── FINAL_RESULTS.md
└── FINAL_TABLES.tex

paper/               # LaTeX paper
├── paper_revised_full.tex
├── references.bib
└── figures/

scripts/             # Utility scripts
└── full_benchmark.py

tests/               # Test suite
```

## Running Benchmarks

### Quick Evaluation
```bash
python -m agentleak benchmark --quick --models gpt-4o-mini
```

### Full Benchmark (Paper Results)
```bash
python -m agentleak benchmark --full --output benchmark_results/
```

### Reproduce Paper Results
```bash
python -m agentleak benchmark --reproduce paper_v1
```

## Development History

### Phase 1: Core Implementation
- 7 leakage channels (C1-C7)
- 19-class attack taxonomy
- 3-tier detection pipeline

### Phase 2: Benchmark Evaluation
- 1000 scenarios across 4 domains
- 8+ models evaluated
- Framework adapters (CrewAI, LangChain, AutoGPT, MetaGPT)

### Phase 3: Defense Analysis
- Sanitizer, privacy prompt, CoT evaluated
- Key finding: defenses only work on external channels
- Internal channels (C2, C5) unprotected

### Phase 4: Paper Preparation
- Results consolidated
- Claims validated with evidence
- Paper revised for publication

## Key Files

| File | Purpose |
|------|---------|
| `agentleak/cli/benchmark.py` | Main benchmark CLI |
| `agentleak/core/channels.py` | Channel definitions |
| `agentleak/core/attacks.py` | Attack taxonomy |
| `agentleak/detection/pipeline.py` | Detection pipeline |
| `benchmark_results/FINAL_RESULTS.json` | Final results |
| `paper/paper_revised_full.tex` | Paper source |

## API Keys Required

```bash
export OPENROUTER_API_KEY="your-key"
```

## Future Work

1. Non-adversarial evaluation (like AgentDAM)
2. Spotlighting defense integration (AgentDojo)
3. Data minimization score metric
4. Cross-framework systematic evaluation
