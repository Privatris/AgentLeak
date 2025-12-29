# AgentLeak Harness - Execution infrastructure
"""
Framework-agnostic harness for running agents on AgentLeak scenarios.

Components:
- BaseAdapter: Abstract interface for agent frameworks
- DryRunAdapter: Mock adapter for testing without LLM calls
- OpenRouterAdapter: Adapter for OpenRouter API (Qwen, etc.)
- TraceCollector: Unified trace collection across all channels
- MockTools: Simulated tools per vertical (Healthcare, Finance, Legal, Corporate)
"""

from .base_adapter import BaseAdapter, AdapterConfig, DryRunAdapter, AdapterStatus, ExecutionResult
from .trace_collector import TraceCollector, TraceBuffer
from .mock_tools import MockToolkit, MockTool

# Optional: OpenRouter adapter (requires httpx)
try:
    from .openrouter_adapter import (
        OpenRouterAdapter,
        OpenRouterConfig,
        create_qwen_adapter,
        QWEN_MODELS,
    )

    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False
    OpenRouterAdapter = None
    OpenRouterConfig = None
    create_qwen_adapter = None
    QWEN_MODELS = None

__all__ = [
    "BaseAdapter",
    "AdapterConfig",
    "DryRunAdapter",
    "AdapterStatus",
    "ExecutionResult",
    "TraceCollector",
    "TraceBuffer",
    "MockToolkit",
    "MockTool",
    # OpenRouter (optional)
    "OpenRouterAdapter",
    "OpenRouterConfig",
    "create_qwen_adapter",
    "QWEN_MODELS",
]
