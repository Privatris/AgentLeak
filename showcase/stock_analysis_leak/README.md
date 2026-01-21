# 🔬 AgentLeak Showcase: Multi-Agent Privacy Vulnerability Demonstration

> **Paper Reference**: IEEE Access submission - "AgentLeak: A Benchmark for Privacy Leakage in Multi-Agent Systems"

## 📋 Executive Summary

This showcase provides **reproducible, scientifically defensible** evidence that multi-agent architectures introduce privacy leakage channels that do not exist in single-agent systems.

### Key Results (January 2026)

| Metric | Value | Significance |
|--------|-------|--------------|
| **Multi-Agent Penalty** | +31 leaks | vs single-agent baseline |
| **Audit Gap** | 87.5% | Leaks invisible to output-only monitoring |
| **Defense Bypass** | 93.3% | Internal channels unprotected |
| **Multi-Agent Specific Channels** | C2, C3, C5 | Do not exist in single-agent |

---

## ⚠️ Reviewer Concern Matrix

We address each potential criticism:

| Concern | Our Defense | Evidence |
|---------|-------------|----------|
| **"Leaks are artificially injected"** | Data enters via **tool responses**, not prompts | See [Methodology](#methodology) |
| **"Single-agent baseline is unfair"** | Single-agent has **equivalent** tool access | Code: `run_single_agent()` |
| **"Defense is designed to fail"** | Defense uses **generic patterns** (regex for IBAN, SSN, etc.) | Code: `OutputDefense.__init__()` |
| **"Simulation ≠ real behavior"** | Simulation models **structural properties**, not LLM specifics | See [Validity](#validity) |
| **"C3 exists in single-agent too"** | Single-agent tools are **internal**; multi-agent tools are **shared services** | See [Channel Definitions](#channels) |

---

## 🎯 Methodology

### Data Flow Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SINGLE-AGENT:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Agent Context                                │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │   │
│  │  │   Tool   │───>│ Internal │───>│  Output  │───> C1 (auditable)   │   │
│  │  │ Response │    │ Process  │    │          │                       │   │
│  │  └──────────┘    └──────────┘    └──────────┘                       │   │
│  │       ↑                                                              │   │
│  │  [Data stays INTERNAL - no shared logging]                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MULTI-AGENT:                                                               │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                 │
│  │   Agent 1     │──>│   Agent 2     │──>│   Agent 3     │                 │
│  │  (Research)   │C2 │  (Analysis)   │C2 │  (Advisory)   │───> C1         │
│  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                 │
│          │ C3                │ C3                │                          │
│          ▼                   ▼                   │                          │
│  ┌────────────────────────────────────────────┐  │                          │
│  │    SHARED TOOL SERVICE (centrally logged)  │  │                          │
│  └────────────────────────────────────────────┘  │                          │
│                      │                           │                          │
│                      ▼ C5                        │                          │
│  ┌────────────────────────────────────────────┐  │                          │
│  │    SHARED MEMORY (persistent, queryable)   │<─┘                          │
│  └────────────────────────────────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Is NOT Artificial

1. **Data enters through TOOLS, not prompts**
   - We do NOT inject PII into agent backstories
   - Tools (CRM, Portfolio API) return client data
   - This is how REAL systems work

2. **Single-agent has EQUIVALENT tool access**
   - Same tools available
   - Same data accessible
   - Difference is ARCHITECTURAL

3. **The "leak" is in SHARING, not ACCESS**
   - Single-agent: Tool response → Agent → Output
   - Multi-agent: Tool response → Agent 1 → Agent 2 → Agent 3 → Output
   - Each arrow is a potential leak point

---

## 📊 Channel Definitions {#channels}

| Channel | Single-Agent | Multi-Agent | Why Different |
|---------|--------------|-------------|---------------|
| **C1** (Output) | ✓ Exists | ✓ Exists | Final output - same for both |
| **C2** (Inter-agent) | ✗ None | ✓ **3 boundaries** | No agent-to-agent in single |
| **C3** (Tool I/O) | Internal only | Shared service | Multi-agent uses centralized tool orchestration |
| **C4** (Logs) | Minimal | Amplified | More agents = more log entries |
| **C5** (Memory) | ✗ None | ✓ Shared | Single-agent doesn't need shared state |
| **C6** (Artifacts) | ✓ Exists | ✓ Exists | Report files - similar for both |

### Critical Distinction: C3 Tool I/O

**Reviewer might ask**: "A single agent also calls tools - why is C3=0?"

**Answer**: In single-agent, tool calls are **internal function calls**:
```python
# Single-agent: direct call, no logging
response = crm_api.lookup(client_id)
```

In multi-agent, tools are **shared services** with orchestration:
```python
# Multi-agent: goes through tool service layer
response = tool_orchestrator.invoke(
    agent="research_analyst",
    tool="crm_api",
    params={"client_id": client_id}
)  # ← This is LOGGED for audit, billing, rate-limiting
```

---

## 🛡️ Defense Analysis

### What We Tested

```python
class OutputDefense:
    """Generic patterns - NOT hardcoded to our test data."""
    patterns = [
        r'FR\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{3}',  # IBAN
        r'\d[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{2}',  # French SSN
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Email
        r'\+\d{2}[-\s]?\d[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}',  # Phone
    ]
```

### Why Defense Fails on Internal Channels

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEFENSE COVERAGE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Agent 1 ──C2──> Agent 2 ──C2──> Agent 3 ──> [DEFENSE] ──> C1 │
│      │              │                                            │
│      └──C3──────────┴──C3──> [Tool Logs]    ← NOT FILTERED     │
│                                                                  │
│   [Memory Store C5]                          ← NOT FILTERED     │
│                                                                  │
│   Result: Defense only sees C1 (final output)                   │
│   Internal channels (C2, C3, C5) BYPASS defense                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Validity Discussion {#validity}

### What the Simulation Captures

| Aspect | Captured? | Notes |
|--------|-----------|-------|
| **Structural channels** | ✅ Yes | C1-C6 exist by architecture |
| **Data flow patterns** | ✅ Yes | Context passing is deterministic |
| **Defense limitations** | ✅ Yes | Output filter can't see internal |
| **LLM decision quality** | ⚠️ No | Would need real LLM runs |
| **Attack adversarial** | ⚠️ No | This shows benign case |

### What We Claim vs. What We Don't

✅ **We claim**: Multi-agent architecture creates new attack surfaces
✅ **We claim**: Output-only defense is insufficient  
✅ **We claim**: Audit gap exists

❌ **We do NOT claim**: Exact leak counts generalize to all systems
❌ **We do NOT claim**: LLMs will always include PII in context

### Why Simulation Is Valid for Our Claims

Our claims are about **architectural properties**, not LLM behavior:

1. **C2 exists because agents communicate** (structural)
2. **C3 exists because tools are shared** (structural)  
3. **C5 exists because memory is shared** (structural)
4. **Defense can't see internal channels** (structural)

These are true regardless of what the LLM decides to output.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install python-dotenv

# Run comparative analysis (recommended for paper)
python showcase_runner.py --mode comparative --stock AAPL

# Run defense analysis
python showcase_runner.py --mode defense --stock MSFT

# Run channel demonstration
python showcase_runner.py --mode channels
```

---

## 📁 File Structure

```
showcase/stock_analysis_leak/
├── showcase_runner.py          # ⭐ MAIN ENTRY POINT (unified)
├── README.md                   # This documentation
│
├── [Legacy - to be deprecated]
│   ├── run_rigorous_showcase.py
│   ├── run_multichannel_showcase.py
│   ├── run_claims_test.py
│   └── run_full_showcase.py
│
├── artifacts.py                # Report generator (C6)
└── tools/
    ├── client_crm.py           # CRM simulation (C3)
    ├── calculator.py           
    └── sec_api.py              
```

---

## 📖 Paper Claims Mapping

| Paper Claim | Showcase Evidence | Script Output |
|-------------|-------------------|---------------|
| **C1**: Multi-agent penalty exists | +31 leaks vs baseline | `Multi-Agent Penalty: +31` |
| **C2**: Internal > External | 28 internal vs 4 external | Channel breakdown |
| **C3**: Audit gap significant | 87.5% missed | `Audit Gap: 87.5%` |
| **C4**: Output defense insufficient | 93.3% bypass | `Defense Bypass: 93.3%` |

---

## 🔍 Reproducibility

Results are deterministic in simulation mode:

```bash
# Same results every run
python showcase_runner.py --mode comparative
python showcase_runner.py --mode comparative
# → Identical output
```

For LLM-based runs (future work), we would need:
- Fixed random seed
- Specific model version
- Temperature = 0

---

## 📝 Citation

If you use this showcase in your research:

```bibtex
@article{agentleak2026,
  title={AgentLeak: A Benchmark for Privacy Leakage in Multi-Agent Systems},
  author={[Authors]},
  journal={IEEE Access},
  year={2026}
}
```
