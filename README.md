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

### Pipeline de détection
- **Tier 1** : Marqueurs canari (correspondance exacte)
- **Tier 2** : Extraction de patterns (regex, détection PII via Presidio)
- **Tier 3** : Similarité sémantique (embeddings, LLM-as-Judge)

### Défenses évaluées
- **D1** : Assainissement de sortie (efficace sur canaux externes)
- **D2** : Validation d'entrée (protection modérée)
- **D3** : Protection mémoire (non implémentée par les frameworks)
- **D4** : Isolation des canaux (nécessite des changements architecturaux)

### Frameworks supportés

| Framework | Intégration | Canaux internes | Accès mémoire |
|-----------|-------------|-----------------|---------------|
| CrewAI | Native | C2, C5 | Complet |
| LangChain | Native | C2, C5 | Complet |
| AutoGPT | Adaptateur | C2, C5 | Limité |
| MetaGPT | Adaptateur | C2, C5 | Limité |

---

## Installation

### Prérequis
- Python 3.10 ou supérieur
- Clé API OpenRouter (ou fournisseur LLM compatible)

### Mise en place

```bash
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak

pip install -e .

# Configurer la clé API
export OPENROUTER_API_KEY=votre_cle
# ou créer un fichier .env
echo "OPENROUTER_API_KEY=votre_cle" > .env
```

### Vérification

```bash
pytest tests/ -v --tb=short
```

---

## Utilisation

### Lancer le benchmark

```bash
# Test rapide (validation)
python -m agentleak run --quick --dry-run

# Benchmark complet
python -m agentleak run --full

# Filtrer par famille d'attaque
python -m agentleak run --attack-family F4 --limit 50

# Avec défenses activées
python -m agentleak run --defense D1
```

### Reproduire les résultats

Le dossier `experiments/all_to_all/` contient les scripts de validation des claims du paper :

```bash
cd experiments/all_to_all

# Test rapide (smoke test)
python smoke_test.py --claims "1,2,3" --scenarios 10

# Benchmark complet
python master_benchmark.py --mode full
```

### Utilisation programmatique

```python
from agentleak import AgentLeakTester, DetectionMode

tester = AgentLeakTester(mode=DetectionMode.HYBRID)

result = tester.check(
    vault={"ssn": "123-45-6789", "email": "patient@hospital.com"},
    output="Le patient dont le SSN est 123-45-6789 a été traité.",
    channel="C1"
)

print(f"Fuite détectée : {result.leaked}")
print(f"Confiance : {result.confidence}")
```

---

## Structure du projet

```
AgentLeak/
├── agentleak/                  # Package principal
│   ├── catalog/                # Définitions canoniques
│   │   ├── attacks.py          # 32 classes d'attaques
│   │   ├── defenses.py         # Implémentations des défenses
│   │   └── channels.py         # 7 canaux de fuite
│   ├── detection/              # Pipeline de détection
│   │   ├── basic_detectors.py  # Détecteurs Tier 1-2
│   │   ├── llm_judge.py        # LLM-as-Judge
│   │   └── presidio_detector.py# Détection PII
│   ├── defenses/               # Mécanismes de défense
│   ├── integrations/           # Intégrations frameworks
│   └── metrics/                # Métriques d'évaluation
│
├── agentleak_data/             # Données du benchmark
│   ├── datasets/               # Scénarios
│   └── prompts/                # Prompts système
│
├── experiments/                # Scripts de validation
│   └── all_to_all/             # Benchmark complet
│
├── tests/                      # Suite de tests
├── docs/                       # Documentation détaillée
└── paper/                      # Sources du paper IEEE
```

---

## Résultats du benchmark

### Comparaison architecturale

| Architecture | Tests | Taux de fuite | IC 95% |
|--------------|-------|---------------|--------|
| Agent unique | 400 | 16.0% | [12.9%, 19.7%] |
| Multi-agent (2) | 350 | 32.0% | [27.3%, 37.1%] |
| Multi-agent (3+) | 250 | 43.2% | [37.1%, 49.5%] |

### Analyse par canal

| Canal | Type | Taux de fuite | Efficacité défense |
|-------|------|---------------|-------------------|
| C1 (Sortie finale) | Externe | 4.8% | 98% |
| C2 (Inter-agent) | Interne | 31.0% | 0% |
| C3 (Entrée outil) | Externe | 3.7% | 85% |
| C4 (Sortie outil) | Externe | 3.2% | 80% |
| C5 (Mémoire) | Interne | 32.0% | 0% |
| C6 (Logs) | Externe | 3.2% | 90% |
| C7 (Artefacts) | Externe | 4.2% | 75% |

### Performance par famille d'attaque

| Famille | Nom | Taux de succès | Canal privilégié |
|---------|-----|----------------|------------------|
| F1 | Prompt & Instruction | 62.2% | C1, C2 |
| F2 | Surface d'outil | 71.7% | C3, C4 |
| F3 | Mémoire & Persistance | 62.7% | C5 |
| F4 | Coordination multi-agent | 80.0% | C2 |
| F5 | Raisonnement & CoT | 62.7% | C2, C5 |
| F6 | Évasion & Obfuscation | 55.0% | C1 |

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

## Licence

MIT — voir [LICENSE](LICENSE).

---

## Contribution

Les contributions sont bienvenues. Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

```bash
pip install -e ".[dev]"
pytest tests/ -v
black agentleak/ tests/
```
