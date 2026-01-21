#!/usr/bin/env python3
"""
AgentLeak - CLI entry point.

Usage:
    python -m agentleak list models
    python -m agentleak run -n 10 -m gpt-4o
    python -m agentleak check "SSN: 123-45-6789"
"""

from agentleak.cli import main

if __name__ == "__main__":
    main()
