"""
APB CLI Module - Command-line interface for Agent Privacy Benchmark.

Usage:
    python -m apb.cli.run_apb --model gpt-4o --privacy-test

Inspired by AgentDAM's run_agentdam.py with enhanced features.
"""

from .run_apb import main, create_arg_parser, ResultTracker

__all__ = [
    "main",
    "create_arg_parser",
    "ResultTracker",
]