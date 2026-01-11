"""
AgentLeak CLI Module - Interactive command-line interface.

Features:
- Interactive menu with logo
- Real-time progress visualization
- Live results table
- Configuration selection
- Test comparison tools
- Unified benchmark CLI
"""


# Lazy imports to avoid loading all dependencies
def _get_app():
    from .app import AgentLeakCLI, main

    return AgentLeakCLI, main


def _get_display():
    from .display import Display, Logo

    return Display, Logo


def _get_menu():
    from .menu import Menu, MenuItem

    return Menu, MenuItem


def _get_progress():
    from .progress import ProgressManager, TestProgress

    return ProgressManager, TestProgress


def _get_benchmark():
    from .benchmark import BenchmarkConfig, BenchmarkEngine, run_benchmark_cli

    return run_benchmark_cli, BenchmarkConfig, BenchmarkEngine


__all__ = [
    "AgentLeakCLI",
    "main",
    "Display",
    "Logo",
    "Menu",
    "MenuItem",
    "ProgressManager",
    "TestProgress",
    "run_benchmark_cli",
    "BenchmarkConfig",
    "BenchmarkEngine",
]
