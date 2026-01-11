#!/usr/bin/env python3
"""
AgentLeak v1.0 - Main Entry Point

Usage:
    # Interactive mode (default)
    python -m agentleak
    agentleak

    # Benchmark mode
    python -m agentleak benchmark --models gpt-4o-mini --n 10
    agentleak benchmark --reproduce-paper
    agentleak benchmark --compare agents

    # Quick test
    python -m agentleak --quick

    # With specific model
    python -m agentleak --model gpt-4o --scenarios 100

Commands:
    (default)   Interactive CLI with menu
    benchmark   Run benchmark tests (see: agentleak benchmark --help)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point with subcommand support."""
    # Check for subcommands
    if len(sys.argv) > 1:
        command = sys.argv[1]

        # Benchmark subcommand
        if command == "benchmark":
            from agentleak.cli.benchmark import run_benchmark_cli

            # Remove 'benchmark' from args before passing
            run_benchmark_cli(sys.argv[2:])
            return

        # Help for subcommands
        if command in ["--help", "-h"]:
            print(__doc__)
            print("\nFor benchmark options: agentleak benchmark --help")
            return

        # Version
        if command in ["--version", "-V"]:
            print("AgentLeak v1.0.0")
            return

    # Default: Interactive CLI
    from cli.app import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
