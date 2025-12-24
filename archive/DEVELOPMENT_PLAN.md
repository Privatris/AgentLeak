# AgentPrivacyBench (APB) - Plan de Développement Itératif

> **Objectif** : Construire le benchmark APB de manière incrémentale, en validant chaque composant avant de passer au suivant.

## 🎯 Target State (Paper)

Le paper décrit un benchmark avec :
- **1000 scénarios** (250 par vertical : Healthcare, Finance, Legal, Corporate)
- **15 classes d'attaques** en 4 familles
- **7 canaux de leakage** (C1-C7)
- **3 niveaux d'adversaires** (A0, A1, A2)
- **Harness framework-agnostic** avec adapters
- **Métriques** : ELR, WLS, CLR, ASR, Pareto AUC

---

## 📋 Phases de Développement

### **Phase 0 : Infrastructure de Base (Semaine 1)**
```
Objectif: Setup projet, CI/CD, structure de base
```

#### Tâches :
- [ ] **0.1** Créer structure de projet Python (`apb/`)
- [ ] **0.2** Setup environnement (pyproject.toml, requirements.txt)
- [ ] **0.3** Créer schémas Pydantic pour les scénarios
- [ ] **0.4** Tests unitaires de base
- [ ] **0.5** GitHub Actions CI

#### Validation :
```bash
pytest tests/ -v  # Tous les tests passent
python -c "from apb import Scenario; print('OK')"
```

---

### **Phase 1 : Génération de Scénarios (Semaines 2-3)**
```
Objectif: Générer 100 scénarios de validation (APB-Lite)
```

#### Tâches :
- [ ] **1.1** Créer templates de scénarios par vertical (JSON)
- [ ] **1.2** Implémenter générateur de données synthétiques (Faker)
- [ ] **1.3** Créer système de canaries (3 tiers: obvious/realistic/semantic)
- [ ] **1.4** Définir `allowed_set` et `private_vault` pour chaque template
- [ ] **1.5** Script de génération en batch avec LLM pour variété
- [ ] **1.6** Validation humaine de 20 scénarios

#### Livrables :
- `apb/generators/scenario_generator.py`
- `apb/data/scenarios/apb_lite_100.jsonl`
- `apb/schemas/scenario.py`

#### Validation :
```bash
python -m apb.generators.scenario_generator --count 25 --vertical healthcare
# Vérifier que les scénarios sont cohérents
```

---

### **Phase 2 : Taxonomie d'Attaques (Semaines 3-4)**
```
Objectif: Implémenter les 15 classes d'attaques
```

#### Tâches :
- [ ] **2.1** Créer base abstraite `Attack` avec interface commune
- [ ] **2.2** Implémenter Family 1 (4 attaques prompt/instruction)
- [ ] **2.3** Implémenter Family 2 (4 attaques tool-surface)
- [ ] **2.4** Implémenter Family 3 (4 attaques memory/persistence)
- [ ] **2.5** Implémenter Family 4 (3 attaques multi-agent)
- [ ] **2.6** Créer payloads templates paramétrables
- [ ] **2.7** Tests unitaires pour chaque classe d'attaque

#### Livrables :
- `apb/attacks/base.py`
- `apb/attacks/family1_prompt.py`
- `apb/attacks/family2_tool.py`
- `apb/attacks/family3_memory.py`
- `apb/attacks/family4_multiagent.py`
- `apb/attacks/payloads/` (templates YAML)

#### Validation :
```bash
python -m apb.attacks.test_all  # 15 classes fonctionnelles
```

---

### **Phase 3 : Harness d'Évaluation (Semaines 4-5)**
```
Objectif: Créer le harness framework-agnostic avec trace JSONL
```

#### Tâches :
- [ ] **3.1** Définir format de trace JSONL unifié
- [ ] **3.2** Créer interface `FrameworkAdapter` (4 méthodes)
- [ ] **3.3** Implémenter adapter LangChain (prioritaire)
- [ ] **3.4** Implémenter adapter OpenAI API (custom)
- [ ] **3.5** Créer mock tools (case_db, email, calendar, etc.)
- [ ] **3.6** Runner de scénarios avec logging

#### Livrables :
- `apb/harness/trace.py`
- `apb/harness/adapter_base.py`
- `apb/harness/adapters/langchain_adapter.py`
- `apb/harness/adapters/openai_adapter.py`
- `apb/harness/tools/` (mock tools)
- `apb/harness/runner.py`

#### Validation :
```bash
python -m apb.harness.runner --scenario apb_fin_001 --framework langchain
# Vérifier trace JSONL générée correctement
```

---

### **Phase 4 : Pipeline de Détection (Semaines 5-6)**
```
Objectif: Implémenter détection de leakage 3-stages
```

#### Tâches :
- [ ] **4.1** Stage 1: Exact canary matching (regex)
- [ ] **4.2** Stage 2: Structured field audit (patterns SSN, CC, etc.)
- [ ] **4.3** Stage 3: Semantic similarity (embeddings + threshold)
- [ ] **4.4** Calibration du seuil τ sur set de validation
- [ ] **4.5** Calcul matrice de confusion
- [ ] **4.6** Support per-channel detection (C1-C7)

#### Livrables :
- `apb/detection/canary_matcher.py`
- `apb/detection/field_auditor.py`
- `apb/detection/semantic_detector.py`
- `apb/detection/pipeline.py`

#### Validation :
```bash
python -m apb.detection.pipeline --trace trace.jsonl --scenario apb_fin_001
# Output: {"leaked": true, "channels": ["C3", "C5"], "fields": ["ssn"]}
```

---

### **Phase 5 : Métriques (Semaine 6)**
```
Objectif: Implémenter ELR, WLS, CLR, ASR, Pareto AUC
```

#### Tâches :
- [ ] **5.1** Implémenter ELR (Exact Leakage Rate)
- [ ] **5.2** Implémenter WLS (Weighted Leakage Score)
- [ ] **5.3** Implémenter CLR per-channel
- [ ] **5.4** Implémenter ASR (Attack Success Rate)
- [ ] **5.5** Implémenter TSR (Task Success Rate) avec oracles
- [ ] **5.6** Calculer Pareto AUC et dominance rate

#### Livrables :
- `apb/metrics/leakage.py`
- `apb/metrics/utility.py`
- `apb/metrics/pareto.py`
- `apb/metrics/report.py`

#### Validation :
```bash
python -m apb.metrics.report --results results.jsonl
# Output: ELR=0.68, WLS=2.31, TSR=0.87, Pareto_AUC=0.45
```

---

### **Phase 6 : Intégration Défenses (Semaine 7)**
```
Objectif: Intégrer LCF et baselines de défense
```

#### Tâches :
- [ ] **6.1** Wrapper pour output filtering (regex PII)
- [ ] **6.2** Wrapper pour policy prompt injection
- [ ] **6.3** Intégration LCF (from paper3)
- [ ] **6.4** Benchmark comparatif des défenses

#### Livrables :
- `apb/defenses/output_filter.py`
- `apb/defenses/policy_prompt.py`
- `apb/defenses/lcf_wrapper.py`

---

### **Phase 7 : Scale Up + Leaderboard (Semaines 8-10)**
```
Objectif: Passer de 100 à 1000 scénarios, créer leaderboard
```

#### Tâches :
- [ ] **7.1** Génération des 1000 scénarios complets
- [ ] **7.2** Validation humaine (sample 50)
- [ ] **7.3** Split 700/300 public/private
- [ ] **7.4** Setup leaderboard (Streamlit ou HuggingFace Spaces)
- [ ] **7.5** Documentation complète

---

## 🗂️ Structure du Code

```
paper4/
├── apb/                          # Package principal
│   ├── __init__.py
│   ├── schemas/
│   │   ├── scenario.py           # Pydantic models
│   │   ├── trace.py              # Trace event models
│   │   └── attack.py             # Attack config models
│   ├── generators/
│   │   ├── scenario_generator.py # Génération scénarios
│   │   ├── canary_generator.py   # 3-tier canaries
│   │   └── vault_generator.py    # Private vaults
│   ├── attacks/
│   │   ├── base.py               # Abstract Attack class
│   │   ├── family1_prompt.py     # DPI, Role Confusion, etc.
│   │   ├── family2_tool.py       # IPI, Tool Poisoning, etc.
│   │   ├── family3_memory.py     # Memory exfil, logs, etc.
│   │   ├── family4_multiagent.py # Cross-agent, delegation
│   │   └── payloads/             # YAML templates
│   ├── harness/
│   │   ├── adapter_base.py       # FrameworkAdapter interface
│   │   ├── adapters/
│   │   │   ├── langchain_adapter.py
│   │   │   ├── openai_adapter.py
│   │   │   └── crewai_adapter.py
│   │   ├── tools/                # Mock tools
│   │   │   ├── case_db.py
│   │   │   ├── email.py
│   │   │   └── calendar.py
│   │   ├── trace.py              # JSONL trace writer
│   │   └── runner.py             # Scenario executor
│   ├── detection/
│   │   ├── canary_matcher.py     # Stage 1: exact match
│   │   ├── field_auditor.py      # Stage 2: patterns
│   │   ├── semantic_detector.py  # Stage 3: embeddings
│   │   └── pipeline.py           # Combined pipeline
│   ├── metrics/
│   │   ├── leakage.py            # ELR, WLS, CLR, ASR
│   │   ├── utility.py            # TSR, cost
│   │   ├── pareto.py             # Pareto AUC, dominance
│   │   └── report.py             # Aggregate reporting
│   ├── defenses/
│   │   ├── output_filter.py
│   │   ├── policy_prompt.py
│   │   └── lcf_wrapper.py
│   └── cli/
│       ├── generate.py           # apb generate
│       ├── run.py                # apb run
│       └── evaluate.py           # apb evaluate
├── data/
│   ├── scenarios/
│   │   ├── apb_lite_100.jsonl    # 100 scénarios pour dev
│   │   └── apb_full_1000.jsonl   # 1000 scénarios complets
│   ├── payloads/                 # Attack payloads
│   └── calibration/              # Threshold calibration data
├── tests/
│   ├── test_schemas.py
│   ├── test_generators.py
│   ├── test_attacks.py
│   ├── test_harness.py
│   ├── test_detection.py
│   └── test_metrics.py
├── notebooks/
│   ├── 01_scenario_exploration.ipynb
│   ├── 02_attack_testing.ipynb
│   └── 03_results_analysis.ipynb
├── docs/
│   └── README.md
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Commencer Maintenant (Phase 0)

### Étape 1 : Créer la structure de base

```bash
cd paper4
mkdir -p apb/{schemas,generators,attacks,harness/adapters,harness/tools,detection,metrics,defenses,cli}
mkdir -p data/{scenarios,payloads,calibration}
mkdir -p tests notebooks docs
touch apb/__init__.py
```

### Étape 2 : Créer les schémas Pydantic

Commencer par `apb/schemas/scenario.py` - le cœur du benchmark.

### Étape 3 : Premier test

```bash
python -c "from apb.schemas.scenario import Scenario; print('Structure OK')"
```

---

## 📊 Critères de Succès par Phase

| Phase | Critère | Métrique |
|-------|---------|----------|
| 0 | Tests passent | 100% pytest |
| 1 | 100 scénarios valides | Validation JSON Schema |
| 2 | 15 attaques fonctionnelles | Tests unitaires |
| 3 | Traces JSONL correctes | 1 scénario end-to-end |
| 4 | FNR < 10% sur validation | Matrice confusion |
| 5 | Métriques correctes | Comparaison manuelle |
| 6 | LCF intégré | Pareto plot généré |
| 7 | 1000 scénarios | Leaderboard live |

---

## 🔄 Cycle d'Itération

```
Pour chaque phase:
1. Implémenter le minimum viable
2. Écrire les tests
3. Valider sur 5-10 exemples
4. Documenter les limitations
5. Commit + tag version
6. Passer à la phase suivante
```

---

## ⚡ Quick Wins (Ordre de priorité)

1. **Schemas Pydantic** → Validation automatique
2. **1 scénario complet** → Preuve de concept
3. **1 attaque simple (DPI)** → Pipeline de bout en bout
4. **Détection canary exacte** → Baseline métriques
5. **LangChain adapter** → Framework le plus populaire

---

## 📅 Timeline Réaliste

| Semaine | Livrables |
|---------|-----------|
| 1 | Structure + Schemas + 10 scénarios |
| 2-3 | 100 scénarios + Générateurs |
| 3-4 | 15 attaques implémentées |
| 4-5 | Harness + LangChain adapter |
| 5-6 | Detection pipeline + Métriques |
| 7 | Défenses + LCF |
| 8-10 | Scale 1000 + Leaderboard |

**Total estimé : 8-10 semaines pour MVP complet**
