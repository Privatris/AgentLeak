#!/bin/bash
# agentleak Quick Setup Script
# Usage: curl -sSL https://raw.githubusercontent.com/YOURORG/AgentLeak/main/setup.sh | bash

set -e

echo "🔐 AgentLeak Setup"
echo "=========================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install package
echo "📦 Installing agentleak..."
pip install -e . -q

# Verify installation
echo "🧪 Verifying installation..."
python -c "from agentleak import __version__; print(f'✓ agentleak {__version__} installed')"

# Run quick test
echo "🧪 Running quick test..."
python scripts/quick_eval.py --n 5

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick commands:"
echo "  pytest tests/ -v              # Run tests"
echo "  python scripts/quick_eval.py  # Quick benchmark"
echo "  python -m agentleak.cli --help      # CLI help"
