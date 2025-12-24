# Contributing to AgentLeak

Thank you for your interest in contributing to APB! This document provides guidelines for contributions.

## 🚀 Quick Setup

```bash
# Clone the repository
git clone https://github.com/YOURORG/AgentLeak.git
cd AgentLeak

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## 📁 Project Structure

```
apb/
├── schemas/          # Data models (Pydantic)
├── generators/       # Data generation
├── attacks/          # Attack implementations
├── harness/          # Execution harness + adapters
├── detection/        # Leakage detection
├── defenses/         # Defense implementations
├── metrics/          # Metric computation
└── utils/            # Utilities
```

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_schemas.py -v

# With coverage
pytest tests/ --cov=apb --cov-report=html
```

## 📝 Code Style

We use:
- **Black** for formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black apb/ tests/
isort apb/ tests/

# Check linting
flake8 apb/ tests/

# Type check
mypy apb/
```

## 🔧 Adding a New Framework Adapter

1. Create adapter file in `apb/harness/adapters/`
2. Implement `BaseAdapter` interface:

```python
from apb.harness.base_adapter import BaseAdapter, AdapterConfig

class MyFrameworkConfig(AdapterConfig):
    # Framework-specific settings
    pass

class MyFrameworkAdapter(BaseAdapter):
    def __init__(self, config: MyFrameworkConfig):
        self.config = config
    
    def execute(self, scenario: Scenario) -> ExecutionResult:
        # Implement execution logic
        pass
    
    def hook_messages(self) -> None:
        # Intercept messages
        pass
    
    def wrap_tools(self) -> None:
        # Instrument tool calls
        pass
    
    def intercept_memory(self) -> None:
        # Capture memory operations
        pass
    
    def export_trace(self) -> ExecutionTrace:
        # Export unified trace
        pass
```

3. Register in `apb/harness/adapters/__init__.py`
4. Add tests in `tests/test_harness.py`

## 🎭 Adding a New Attack Class

1. Define attack in `apb/attacks/attack_module.py`:

```python
class MyNewAttack(AttackClass):
    name = "my_new_attack"
    family = AttackFamily.TOOL  # or PROMPT, MEMORY, MULTI_AGENT
    injection_surface = ["tool_output"]
    target_channels = [Channel.C1, Channel.C3]
    
    def generate_payload(self, scenario: Scenario) -> str:
        return "Attack payload template"
```

2. Register in `ATTACK_REGISTRY`
3. Add test payloads in `apb/attacks/payloads/`
4. Add tests in `tests/test_attacks.py`

## 📊 Adding New Metrics

1. Implement in `apb/metrics/`:

```python
def compute_my_metric(
    scenarios: List[Scenario],
    traces: List[ExecutionTrace],
) -> float:
    """Compute my custom metric."""
    # Implementation
    return value
```

2. Add to `apb/metrics/__init__.py`
3. Add tests in `tests/test_metrics.py`

## 🛡️ Adding a New Defense

1. Create defense in `apb/defenses/`:

```python
from apb.defenses.base_defense import BaseDefense

class MyDefense(BaseDefense):
    def __init__(self, strength: float = 0.5):
        self.strength = strength
    
    def apply(self, embedding: np.ndarray, context: Dict) -> np.ndarray:
        # Apply defense transformation
        return protected_embedding
    
    def should_block(self, content: str, context: Dict) -> bool:
        # Decide whether to block content
        return False
```

2. Register in `apb/defenses/__init__.py`
3. Add to benchmark runner
4. Add tests

## 📚 Documentation

- Update docstrings with Google style
- Update README.md if adding major features
- Update `docs/PAPER_CODE_MAPPING.md` for paper-related changes

## 🔀 Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Ensure all tests pass: `pytest tests/ -v`
5. Format code: `black apb/ tests/ && isort apb/ tests/`
6. Commit with descriptive message
7. Push and create PR

## 📋 Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Refactoring
- `perf`: Performance
- `chore`: Maintenance

Examples:
```
feat(attacks): add retrieval trap attack class
fix(detection): handle empty traces in pipeline
docs(readme): update installation instructions
test(harness): add CrewAI adapter tests
```

## ❓ Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Email: faouzi.elyagoubi@polymtl.ca

Thank you for contributing! 🙏
