# AgentLeak - Résultats Finaux

**Date:** 2026-01-11  
**Tests:** 1,000 scénarios  
**Modèles:** 8 LLMs  
**Frameworks:** CrewAI, LangChain, AutoGPT, MetaGPT

---

## 🎯 Findings Principaux

### 1. Multi-Agent > Single-Agent Leakage

| Architecture | Tests | Leak Rate |
|--------------|-------|-----------|
| Single-agent | 400 | **16.0%** |
| Multi-agent (2) | 350 | **32.0%** |
| Multi-agent (3+) | 250 | **43.2%** |

**Conclusion:** Les systèmes multi-agents fuient **2-3× plus** que les single-agent.

---

### 2. Canaux Internes Non Protégés

| Type | Canaux | Leak Rate | Défenses |
|------|--------|-----------|----------|
| External | C1, C3, C4, C6, C7 | **3.8%** | Sanitizer, Prompt, CoT |
| **Internal** | **C2, C5** | **31.5%** | **AUCUNE** |

**Conclusion:** Les canaux internes ont un taux de fuite **8.3× supérieur** car aucune défense n'existe.

---

### 3. Efficacité des Défenses par Canal

| Défense | C1 (External) | C2/C5 (Internal) |
|---------|---------------|------------------|
| None | 48% | 31% |
| Privacy Prompt | 19% (-60%) | 29% (-6%) |
| CoT | 22% (-54%) | 31% (0%) |
| **Sanitizer** | **1%** (-98%) | **31%** (0%) |

**Conclusion:** Le sanitizer est **98% efficace sur C1** mais **0% sur C2/C5** - il n'opère pas sur les messages inter-agents.

---

### 4. Frameworks Sans Protection

| Framework | Internal Leak Rate | Mitigation |
|-----------|-------------------|------------|
| CrewAI | 33% | ❌ None |
| AutoGPT | 35% | ❌ None |
| LangChain | 29% | ❌ None |
| MetaGPT | 28% | ❌ None |

**Conclusion:** **Aucun framework majeur** n'implémente de mécanismes de privacy inter-agents.

---

## ✅ Claims Validées

1. ✅ **Multi-agent leak > Single-agent** (2.3× plus)
2. ✅ **C2/C5 non protégés** (31% vs 3.8%)
3. ✅ **Frameworks sans mécanismes** (28-35% leak rate)

---

## 📊 Statistiques pour le Paper

```
Overall leak rate: 28.4%
Single-agent rate: 16.0%
Multi-agent rate: 36.7%
Multi-agent increase: 2.3x

Internal channel rate: 31.5%
External channel rate: 3.8%
Internal/External ratio: 8.3x

Sanitizer on C1: 98% effective
Sanitizer on C2/C5: 0% effective
```
