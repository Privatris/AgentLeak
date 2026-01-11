"""
AgentLeak Utilities - Helper functions and utilities.

Provides:
- APIUsageTracker: Thread-safe token usage monitoring
- Retry utilities with exponential backoff
- LLM caching for development
- Progress tracking
"""

from .api_tracker import (
    APIUsageTracker,
    LLMCache,
    ProgressTracker,
    api_usage_tracker,
    append_jsonl,
    cached_llm_call,
    count_message_tokens,
    count_tokens,
    llm_cache,
    load_json,
    load_jsonl,
    print_api_usage,
    retry_with_backoff,
    save_json,
)

__all__ = [
    # API Tracking
    "APIUsageTracker",
    "api_usage_tracker",
    "print_api_usage",
    # Retry
    "retry_with_backoff",
    # Token counting
    "count_tokens",
    "count_message_tokens",
    # Caching
    "LLMCache",
    "llm_cache",
    "cached_llm_call",
    # Progress
    "ProgressTracker",
    # I/O
    "load_json",
    "save_json",
    "load_jsonl",
    "append_jsonl",
]
