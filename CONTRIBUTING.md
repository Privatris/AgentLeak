# Contribution

Merci de votre intérêt pour AgentLeak. Ce document décrit les procédures pour contribuer au projet.

## Code de conduite

Nous attendons des interactions respectueuses et constructives de tous les contributeurs.

## Mise en place

```bash
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

## Tests

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_defenses.py -v

# Avec couverture
pytest tests/ --cov=agentleak --cov-report=html
```

## Style de code

```bash
# Formatage
black agentleak/ tests/

# Tri des imports
isort agentleak/ tests/

# Vérification des types
mypy agentleak/

# Linting
ruff agentleak/
```

## Signaler un bug

1. Vérifier les issues existantes
2. Inclure :
   - Version de Python
   - Système d'exploitation
   - Étapes pour reproduire
   - Messages d'erreur

## Proposer une fonctionnalité

1. Ouvrir une discussion sur GitHub
2. Décrire le problème résolu
3. Proposer une solution

## Pull requests

1. Fork le repository
2. Créer une branche : `git checkout -b feature/ma-fonctionnalite`
3. Coder et tester
4. Soumettre la PR

### Checklist

- [ ] Tests ajoutés pour le nouveau code
- [ ] Tous les tests passent
- [ ] Documentation mise à jour
- [ ] Code formaté avec black

## Structure du projet

Voir [AI_CONTEXT.md](AI_CONTEXT.md) pour une description détaillée de l'architecture.

## Questions

Ouvrir une issue sur GitHub avec le tag `question`.
