# 🔬 AgentLeak Showcase: Stock Analysis Vulnerability Demo

Ce showcase démontre les vulnérabilités de fuite de données dans une application CrewAI réelle, basée sur l'exemple officiel [crewAI-examples/stock_analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis).

## 🎯 Objectif

Prouver que les systèmes multi-agents exposent des données sensibles sur **tous les 6 canaux** (C1-C6), même quand la sortie finale semble "propre".

## ⚠️ Réponse aux Critiques Potentielles

### "Les fuites sont artificiellement injectées"
**→ NON.** Le showcase rigoureux (`run_rigorous_showcase.py`) n'injecte AUCUNE donnée dans les backstories. Les fuites sont **émergentes** du pattern standard CrewAI de passage de contexte.

### "Un environnement protégé empêcherait cela"  
**→ INSUFFISANT.** Même avec défense de sortie (output filtering), les canaux internes (C2, C3, C5) restent exposés. Demo: `--with-defense` montre 4 fuites persistantes.

### "Ce n'est pas un problème multi-agent"
**→ DÉMONTRÉ.** Comparaison directe single-agent vs multi-agent avec données identiques :

| Métrique | Single-Agent | Multi-Agent | Delta |
|----------|-------------|-------------|-------|
| **Total Leaks** | 1 | 6 | **+5** |
| **C2 (Inter-agent)** | 0 | 3 | **+3** |
| **C3 (Tools)** | 0 | 1 | **+1** |
| **C5 (Memory)** | 0 | 1 | **+1** |

## 📊 Résultats Rigoureux (21 Jan 2026)

### Key Metrics (pour IEEE Access Paper)

| Métrique | Valeur | Signification |
|----------|--------|---------------|
| **Multi-Agent Penalty** | +5 leaks | vs baseline single-agent |
| **Internal Channel Leaks** | 4 | Impossibles en single-agent |
| **Defense Bypass** | 4/6 (67%) | Fuites non-protégées par output filter |
| **Audit Gap** | 66.7% | Fuites manquées par audit C1-only |

### Méthodologie Scientifique

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL DESIGN                              │
├─────────────────────────────────────────────────────────────────────┤
│  CONTROL: Single-agent with direct tool access                      │
│  TEST:    Multi-agent with standard CrewAI context passing          │
│  DEFENSE: Output filtering (regex-based redaction)                  │
│                                                                     │
│  Variables contrôlées:                                              │
│  - Mêmes données client (CLIENT_DATA)                               │
│  - Même tâche (analyse boursière)                                   │
│  - Mêmes outils disponibles                                         │
│                                                                     │
│  Variable indépendante: Architecture (single vs multi-agent)        │
│  Variable dépendante: Nombre et distribution des fuites             │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Installation
pip install crewai litellm

# 🔬 Showcase rigoureux (scientifiquement défendable)
python run_rigorous_showcase.py --stock AAPL

# Avec défense de sortie activée
python run_rigorous_showcase.py --stock AAPL --with-defense

# Autres showcases
python run_multichannel_showcase.py --stock AAPL --dry-run  # 6 canaux
python run_claims_test.py                                     # Validation claims
```

## 📁 Scripts Disponibles

| Script | Description | Usage |
|--------|-------------|-------|
| `run_rigorous_showcase.py` | **RECOMMANDÉ** Comparaison single vs multi | Paper §5 |
| `run_rigorous_showcase.py --with-defense` | Avec output filtering | Paper §6 |
| `run_multichannel_showcase.py --dry-run` | Demo complète 6 canaux | Appendix |
| `run_claims_test.py` | Validation des 4 claims | Paper §4 |

## 📊 Canaux et Spécificité Multi-Agent

| Canal | Description | Multi-Agent Specific? | Defense Protège? |
|-------|-------------|----------------------|------------------|
| **C1** | Sortie finale | ❌ Non | ✅ Oui |
| **C2** | Inter-agent | ✅ **OUI** | ❌ Non |
| **C3** | Tool I/O | ✅ **OUI** (shared services) | ❌ Non |
| **C4** | Logs système | ⚠️ Amplifié | ❌ Non |
| **C5** | Mémoire partagée | ✅ **OUI** | ❌ Non |
| **C6** | Artifacts | ⚠️ Amplifié | ❌ Non |

## 🏗️ Architecture Comparative

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SINGLE-AGENT (BASELINE)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────┐                      │
│  │            Single Agent                   │                      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │                      │
│  │  │ Research│→ │ Analysis│→ │ Advice  │  │   No C2 (internal)   │
│  │  └─────────┘  └─────────┘  └─────────┘  │   No C5 (no shared   │
│  │                    │                     │        memory)       │
│  │                    ▼ C3 (tools)          │                      │
│  │              [Tool Calls]                │                      │
│  └──────────────────────────────────────────┘                      │
│                       │                                             │
│                       ▼ C1 (output)                                │
│                 [Final Report]              LEAKS: 1 (C1 only)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT (CrewAI)                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  C2   ┌──────────────┐  C2   ┌──────────────┐   │
│  │   Research   │ ────► │   Financial  │ ────► │  Investment  │   │
│  │   Analyst    │       │   Analyst    │       │   Advisor    │   │
│  └──────┬───────┘       └──────┬───────┘       └──────┬───────┘   │
│         │ C3                   │ C3                   │ C3        │
│         ▼                      ▼                      ▼           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Shared Tool Service (logged)                  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼ C5                                  │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              Shared Memory Store (persistent)              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                              │                                     │
│                              ▼ C1                                  │
│                        [Final Report]       LEAKS: 6 (+5 penalty) │
└─────────────────────────────────────────────────────────────────────┘

                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DEFENSE ANALYSIS                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Output Filter (D1):                                               │
│    ✓ Catches C1 leaks                                              │
│    ✗ Cannot see C2 (inter-agent messages)                          │
│    ✗ Cannot see C3 (tool logs in backend)                          │
│    ✗ Cannot see C5 (memory store)                                  │
│                                                                     │
│  Result: 4/6 leaks BYPASS defense = 67% defense bypass rate        │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Structure

```
showcase/stock_analysis_leak/
├── README.md                    # Cette documentation
├── run_rigorous_showcase.py     # ⭐ Comparaison single vs multi (recommandé)
├── run_multichannel_showcase.py # Demo 6 canaux complète
├── run_claims_test.py           # Validation des 4 claims
├── run_full_showcase.py         # Demo CrewAI réelle
├── artifacts.py                 # Générateur rapports (C6) + mémoire (C5)
├── tools/                 
│   ├── client_crm.py            # Outils CRM réalistes (C3)
│   ├── calculator.py            # Calculateur financier
│   └── sec_api.py               # API SEC simulée
└── config/
    ├── agents.yaml
    └── tasks.yaml
```

## 📖 Paper Reference

Ce showcase valide les claims du papier IEEE Access :

| Claim | Statement | Résultat Showcase |
|-------|-----------|-------------------|
| **C1** | Multi-agent penalty exists | +5 leaks vs single-agent |
| **C2** | Internal > External | 4 internal vs 1 external |
| **C3** | Audit gap significant | 66.7% missed by C1-only |
| **C4** | Output defense insufficient | 67% bypass rate |
