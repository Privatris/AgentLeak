# AgentLeak Evaluation Summary

Benchmark evaluation of privacy leakage across 1,000 scenarios with 8 LLM models and 4 multi-agent frameworks.

## Architecture Comparison

| Architecture | Tests | Leak Rate | Primary Channels |
|--------------|-------|-----------|------------------|
| Single-agent | 400 | 16.0% | C1, C3 |
| Multi-agent (2 agents) | 350 | 32.0% | C2, C5, C1 |
| Multi-agent (3+ agents) | 250 | 43.2% | C2, C5 |

Multi-agent systems exhibit 2.3× higher leak rates than single-agent configurations (χ² = 47.3, p < 0.001).

## Channel Analysis

| Type | Channels | Leak Rate | Available Defenses |
|------|----------|-----------|-------------------|
| External | C1, C3, C4, C6, C7 | 3.8% | Sanitizer, Privacy Prompt |
| Internal | C2, C5 | 31.5% | None |

Internal channels show 8.3× higher leak rates due to absence of inter-agent privacy controls (χ² = 89.7, p < 0.001).

## Defense Effectiveness

| Defense | C1 (Output) | C2/C5 (Internal) |
|---------|-------------|------------------|
| Baseline | 48% | 31% |
| Privacy Prompt | 19% | 29% |
| Chain-of-Thought | 22% | 31% |
| Sanitizer | 1% | 31% |

Sanitizer achieves 98% effectiveness on external outputs but 0% on internal channels.

## Framework Evaluation

| Framework | C2 Leak Rate | Inter-agent Privacy |
|-----------|--------------|---------------------|
| CrewAI | 33% | Not implemented |
| AutoGPT | 35% | Not implemented |
| LangChain | 29% | Not implemented |
| MetaGPT | 28% | Not implemented |

All evaluated frameworks lack privacy controls for inter-agent communication.

## Validation

- Simulation-based evaluation: N=1,000 scenarios
- Real framework validation: N=655 scenarios across 4 frameworks
- C2 leakage rates: CrewAI 33%, AutoGPT 35%, LangChain 29%, MetaGPT 28%
- Statistical tests: Chi-square with 95% confidence intervals
