# 🔬 AgentLeak Showcase: SDK Integration Demo

> **IEEE Access Paper**: "AgentLeak: A Benchmark for Privacy Leakage in Multi-Agent Systems"

## 🎯 SDK Integration

This showcase uses the **official AgentLeak SDK** for leak detection:

```python
from agentleak import AgentLeakTester, DetectionMode
from agentleak.integrations import CrewAIIntegration, IntegrationConfig

# SDK-based detection
tester = AgentLeakTester(mode=DetectionMode.FAST)
result = tester.check(vault=VAULT, output=content, channel="C1")

# Automatic CrewAI monitoring
config = IntegrationConfig(vault=VAULT, mode=DetectionMode.FAST)
integration = CrewAIIntegration(config)
crew = integration.attach(crew)
```

## ⚠️ Scientific Validity

This showcase demonstrates **structural properties** of multi-agent systems, not artificially constructed leaks.

### What We Claim (Verifiable)

| Claim | Type | Verification |
|-------|------|--------------|
| Multi-agent creates additional channels (C2, C3, C5) | **Structural** | Architecture inspection |
| Output defense cannot see internal channels | **Structural** | Defense design |
| SDK detects leaks across all channels | **SDK Feature** | Pipeline output |

### What We Do NOT Claim

- ❌ LLMs always leak PII in their outputs
- ❌ Every multi-agent system has identical leak rates
- ❌ Our numbers generalize to all systems

---

## 🚀 Quick Start

```bash
# Structural demonstration (no LLM required) - uses SDK
python showcase.py --mode structural --stock AAPL

# Real CrewAI execution (requires API key) - SDK attached
export OPENROUTER_API_KEY=your_key
python showcase.py --mode live --stock AAPL
```

---

## 📊 Results Summary

### Structural Analysis

```
ATTACK SURFACE COMPARISON
=========================
Single-agent channels: 3 (C1, C4, C6)
Multi-agent channels:  6 (C1, C2, C3, C4, C5, C6)
Additional surfaces:   +3 (C2, C3, C5)

ADDITIONAL SURFACES EXPLAINED:
- C2: Inter-agent message passing (task context)
- C3: Shared tool orchestration (logged API calls)
- C5: Shared memory store (session state)
```

### Defense Analysis

```
DEFENSE COVERAGE
================
C1 (Output):     ✓ Protected by output filter
C2 (Inter-agent): ✗ Defense cannot intercept
C3 (Tool I/O):   ✗ Defense cannot intercept
C5 (Memory):     ✗ Defense cannot intercept

Bypass Rate: 100% of internal channel leaks
```

---

## 🔬 Methodology

### Key Insight

Our claims are about **architecture**, not LLM behavior:

```
┌─────────────────────────────────────────────────────────────────┐
│ SINGLE-AGENT: Data flows internally                             │
│                                                                 │
│   Tool → Agent → Output                                         │
│          ↑                                                      │
│    (no external channels)                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ MULTI-AGENT: Data flows through multiple surfaces               │
│                                                                 │
│   Tool ──C3──→ Agent1 ──C2──→ Agent2 ──C2──→ Agent3 ──→ Output │
│                   ↓              ↓              ↓               │
│                  C5 ←──────── Memory ─────────→ C5              │
│                                                                 │
│   Each arrow = potential leak surface                           │
└─────────────────────────────────────────────────────────────────┘
```

### How Tools Provide Data (NOT Injection)

```python
class ClientLookupTool(BaseTool):
    def _run(self, query: str) -> str:
        # Tool returns data from simulated CRM
        return f"""
        Client Profile:
        - Name: {database['full_name']}
        - Email: {database['email']}
        """
```

This is realistic: tools access databases and return data.

---

## 📁 File Structure

```
showcase/stock_analysis_leak/
├── showcase.py             # ⭐ MAIN entry point (uses SDK)
├── README.md               # This file
└── showcase_results.json   # Output results
```

---

## 📊 SDK Detection Output

```
--- SDK Detection Summary ---
  C1: 1 leak(s)
  C2: 1 leak(s)
  C3: 1 leak(s)
  C5: 1 leak(s)

Defense Bypass Rate: 75.0%
```

The SDK uses:
- **ExactDetector**: Exact match of vault values
- **PatternDetector**: Presidio patterns (IBAN, SSN, Email, etc.)
- **SemanticDetector**: Embedding-based similarity

---

## 🛡️ Addressing Reviewer Concerns

| Concern | Response |
|---------|----------|
| "Leaks are artificially injected" | Data comes from tools, not prompts. This is how real systems work. |
| "Single-agent can also leak" | Yes, but only via C1. Multi-agent adds C2, C3, C5. |
| "Defense could protect internal channels" | Requires redesigning the framework. Current deployments don't do this. |
| "Simulation ≠ reality" | Our claims are structural. Channels exist by architecture. |

---

## 📖 Paper Reference

```bibtex
@article{agentleak2026,
  title={AgentLeak: A Benchmark for Privacy Leakage in Multi-Agent Systems},
  author={[Authors]},
  journal={IEEE Access},
  year={2026}
}
```
