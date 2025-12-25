# Installation Guide

## Prerequisites
- Python 3.10+
- `pip` or `conda`

## Installation

### Option 1: Direct Installation (Recommended for Users)

```bash
# Clone the repository
git clone https://github.com/Privatris/AgentLeak.git
cd AgentLeak

# Install the package
pip install -e .
```

### Option 2: With Virtual Environment (Recommended for Development)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .
```

### Option 3: With Conda

```bash
conda create -n agentleak python=3.11
conda activate agentleak
pip install -e .
```

## Verify Installation

```bash
# Run the test suite
pytest tests/ -q

# Quick eval
python scripts/quick_eval.py --n 5
```

## Troubleshooting

### ModuleNotFoundError for agentleak

If you get `ModuleNotFoundError: No module named 'agentleak'`, ensure you ran:
```bash
pip install -e .
```

### Python Path Issues

If tests fail due to import errors, set PYTHONPATH:
```bash
PYTHONPATH=. pytest tests/
```

### Missing Dependencies

Install optional dependencies for framework adapters:

```bash
# For LangChain support
pip install langchain langchain-openai

# For CrewAI support  
pip install crewai

# For AutoGPT support
pip install autogpt

# All frameworks
pip install -e ".[frameworks]"
```

## Next Steps

See [README.md](README.md) for usage instructions.
