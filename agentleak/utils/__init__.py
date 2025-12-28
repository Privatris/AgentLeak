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
    api_usage_tracker,
    print_api_usage,
    retry_with_backoff,
    count_tokens,
    count_message_tokens,
    LLMCache,
    llm_cache,
    cached_llm_call,
    ProgressTracker,
    load_json,
    save_json,
    load_jsonl,
    append_jsonl,
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
