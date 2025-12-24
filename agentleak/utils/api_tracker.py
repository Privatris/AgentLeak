"""
APB API Usage Tracker - Thread-safe token usage monitoring.

Provides utilities for tracking API usage across the codebase:
1. APIUsageTracker: Singleton class for token counting
2. @print_api_usage: Decorator for automatic usage printing
3. Retry utilities with exponential backoff
4. Caching for expensive LLM calls
"""

from __future__ import annotations
import functools
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Try to import tiktoken for token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


# =============================================================================
# API Usage Tracker (Singleton, Thread-Safe)
# =============================================================================

class APIUsageTracker:
    """
    A singleton class to track API usage across threads.
    
    Inspired by PrivacyLens's usage tracking for cost monitoring
    and debugging during evaluation runs.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._token_usage: Dict[str, Dict[str, int]] = {}
        self._call_count: Dict[str, int] = {}
        self._usage_lock = threading.Lock()
    
    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Get current token usage by model."""
        with self._usage_lock:
            return dict(self._token_usage)
    
    def get_call_count(self) -> Dict[str, int]:
        """Get API call count by model."""
        with self._usage_lock:
            return dict(self._call_count)
    
    def reset(self):
        """Reset all usage counters."""
        with self._usage_lock:
            self._token_usage = {}
            self._call_count = {}
    
    def increment_token_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Thread-safe increment of token usage."""
        with self._usage_lock:
            if model not in self._token_usage:
                self._token_usage[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            self._token_usage[model]["prompt_tokens"] += prompt_tokens
            self._token_usage[model]["completion_tokens"] += completion_tokens
            self._token_usage[model]["total_tokens"] += prompt_tokens + completion_tokens
            
            if model not in self._call_count:
                self._call_count[model] = 0
            self._call_count[model] += 1
    
    def estimate_cost(self, pricing: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """
        Estimate total cost based on token usage.
        
        Args:
            pricing: Dict mapping model to {"prompt": cost_per_1k, "completion": cost_per_1k}
        """
        if pricing is None:
            # Default pricing (approximate, as of 2024)
            pricing = {
                "gpt-4o": {"prompt": 0.005, "completion": 0.015},
                "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},
            }
        
        total_cost = 0.0
        with self._usage_lock:
            for model, usage in self._token_usage.items():
                if model in pricing:
                    cost = (
                        (usage["prompt_tokens"] / 1000) * pricing[model]["prompt"] +
                        (usage["completion_tokens"] / 1000) * pricing[model]["completion"]
                    )
                    total_cost += cost
        
        return total_cost
    
    def summary(self) -> str:
        """Get a formatted summary of API usage."""
        lines = ["API Usage Summary:"]
        lines.append("-" * 50)
        
        with self._usage_lock:
            for model, usage in self._token_usage.items():
                calls = self._call_count.get(model, 0)
                lines.append(f"{model}:")
                lines.append(f"  Calls: {calls}")
                lines.append(f"  Prompt tokens: {usage['prompt_tokens']:,}")
                lines.append(f"  Completion tokens: {usage['completion_tokens']:,}")
                lines.append(f"  Total tokens: {usage['total_tokens']:,}")
        
        lines.append("-" * 50)
        lines.append(f"Estimated cost: ${self.estimate_cost():.4f}")
        
        return "\n".join(lines)


# Global singleton instance
api_usage_tracker = APIUsageTracker()


# =============================================================================
# Decorator for Automatic Usage Printing
# =============================================================================

def print_api_usage(func: Callable) -> Callable:
    """
    Decorator to print API usage after function execution.
    
    Prints usage summary even if the function raises an exception,
    which is useful for debugging failed runs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"\n{func.__name__} raised an exception: {e}")
            raise
        finally:
            print(f"\n{api_usage_tracker.summary()}")
    
    return wrapper


# =============================================================================
# Retry Utilities with Exponential Backoff
# =============================================================================

def retry_with_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        jitter: Add random jitter to prevent thundering herd
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        actual_delay = delay
                        if jitter:
                            actual_delay *= (0.5 + random.random())
                        actual_delay = min(actual_delay, max_delay)
                        
                        print(f"Attempt {attempt + 1} failed: {e}. "
                              f"Retrying in {actual_delay:.1f}s...")
                        time.sleep(actual_delay)
                        delay *= backoff_factor
            
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Token Counting Utilities
# =============================================================================

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string.
    
    Uses tiktoken for accurate counting, falls back to estimate if unavailable.
    """
    if HAS_TIKTOKEN:
        try:
            # Map model names to encodings
            if "gpt-4" in model or "gpt-3.5" in model:
                encoding = tiktoken.encoding_for_model(model)
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    # Fallback: rough estimate (1 token ≈ 4 characters)
    return len(text) // 4


def count_message_tokens(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o"
) -> int:
    """Count tokens in a list of chat messages."""
    total = 0
    for msg in messages:
        # Each message has role and content
        total += count_tokens(msg.get("role", ""), model)
        total += count_tokens(msg.get("content", ""), model)
        total += 4  # Overhead per message
    total += 2  # Overhead for the prompt
    return total


# =============================================================================
# Simple Cache for LLM Calls
# =============================================================================

class LLMCache:
    """
    Simple file-based cache for LLM responses.
    
    Caches responses based on the hash of the request to avoid
    redundant API calls during development and testing.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".apb_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def _get_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate a cache key from the request."""
        # Create deterministic string representation
        request_str = json.dumps(request, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, request: Dict[str, Any]) -> Optional[str]:
        """Get cached response if available."""
        key = self._get_cache_key(request)
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        data = json.load(f)
                    return data.get("response")
                except Exception:
                    return None
        return None
    
    def set(self, request: Dict[str, Any], response: str):
        """Cache a response."""
        key = self._get_cache_key(request)
        cache_path = self._get_cache_path(key)
        
        with self._lock:
            try:
                with open(cache_path, "w") as f:
                    json.dump({"request": request, "response": response}, f)
            except Exception as e:
                print(f"Cache write error: {e}")
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


# Global cache instance
llm_cache = LLMCache()


def cached_llm_call(
    call_func: Callable,
    use_cache: bool = True,
) -> Callable:
    """
    Wrapper to add caching to an LLM call function.
    
    Args:
        call_func: The original LLM call function
        use_cache: Whether to use caching
    """
    @functools.wraps(call_func)
    def wrapper(engine: str, messages: List[Dict], **kwargs):
        if not use_cache:
            return call_func(engine, messages, **kwargs)
        
        # Build cache request
        cache_request = {
            "engine": engine,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k != "api_key"}
        }
        
        # Check cache
        cached = llm_cache.get(cache_request)
        if cached is not None:
            return cached
        
        # Make actual call
        response = call_func(engine, messages, **kwargs)
        
        # Cache response
        llm_cache.set(cache_request, response)
        
        return response
    
    return wrapper


# =============================================================================
# Progress Tracking
# =============================================================================

@dataclass
class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    total: int
    current: int = 0
    description: str = "Processing"
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        pct = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_len = 30
        filled = int(bar_len * self.current / self.total) if self.total > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r{self.description}: [{bar}] {self.current}/{self.total} ({pct:.1f}%)", end="")
        if self.current >= self.total:
            print()  # Newline at completion
    
    def finish(self):
        """Mark as complete."""
        self.current = self.total
        self._print_progress()


# =============================================================================
# Data I/O Utilities
# =============================================================================

def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)


def load_jsonl(filepath: str) -> List[Any]:
    """Load data from JSONL file."""
    results = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def append_jsonl(filepath: str, data: Any) -> None:
    """Append data to JSONL file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")
