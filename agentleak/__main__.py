#!/usr/bin/env python3
"""
AgentLeak v1.0 - Main Entry Point

Usage:
    # Interactive mode
    python -m agentleak

    # Or use the launcher script
    ./run_agentleak.sh

    # Quick test
    python -m agentleak --quick

    # With specific model
    python -m agentleak --model gpt-4o --scenarios 100
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cli.app import main

if __name__ == "__main__":
    main()
