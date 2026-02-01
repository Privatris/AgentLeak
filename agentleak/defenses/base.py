"""
AgentLeak Base Defense - Abstract base class for all defenses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..schemas.scenario import Channel


class FilterAction(str, Enum):
    """Action taken by a defense filter."""

    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


@dataclass
class DefenseConfig:
    """Base configuration for defenses."""

    enabled: bool = True
    log_decisions: bool = True
    channels_to_monitor: list[Channel] = field(default_factory=lambda: list(Channel))
    strict_mode: bool = False  # Block on any suspicion


@dataclass
class DefenseResult:
    """Result from applying a defense."""

    action: FilterAction
    original_content: str
    filtered_content: Optional[str] = None  # After redaction/modification
    confidence: float = 1.0
    reason: Optional[str] = None
    detected_patterns: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.action == FilterAction.BLOCK

    @property
    def modified(self) -> bool:
        return self.action == FilterAction.REDACT


class BaseDefense(ABC):
    """
    Abstract base class for all privacy defenses.

    Defenses intercept content before it leaves the agent system
    and can block, redact, or warn about potential privacy violations.
    """

    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        self._decision_log: list[DefenseResult] = []

    @abstractmethod
    def filter(self, content: str, channel: Channel, metadata: Optional[dict[str, Any]] = None) -> DefenseResult:
        """
        Apply defense filter to content.

        Args:
            content: Content to filter
            channel: Channel where content is going
            metadata: Additional context (e.g. source agent, target agent)

        Returns:
            DefenseResult with action and potentially modified content
        """
        pass

    def filter_batch(
        self,
        items: list[tuple[str, Channel, Optional[dict[str, Any]]]],
    ) -> list[DefenseResult]:
        """Filter multiple items."""
        return [self.filter(content, channel, metadata) for content, channel, metadata in items]

    def should_monitor(self, channel: Channel) -> bool:
        """Check if this channel should be monitored."""
        if not self.config.channels_to_monitor:
            return True  # Monitor all
        return channel in self.config.channels_to_monitor

    def get_decision_log(self) -> list[DefenseResult]:
        """Get log of all decisions made."""
        return self._decision_log.copy()

    def clear_log(self) -> None:
        """Clear decision log."""
        self._decision_log.clear()

    def _log_decision(self, result: DefenseResult) -> None:
        """Log a decision if logging is enabled."""
        if self.config.log_decisions:
            self._decision_log.append(result)

    @property
    @abstractmethod
    def name(self) -> str:
        """Defense name."""
        pass

    def stats(self) -> dict[str, Any]:
        """Get statistics about defense decisions."""
        if not self._decision_log:
            return {"total": 0, "blocked": 0, "redacted": 0, "allowed": 0}

        return {
            "total": len(self._decision_log),
            "blocked": sum(1 for d in self._decision_log if d.action == FilterAction.BLOCK),
            "redacted": sum(1 for d in self._decision_log if d.action == FilterAction.REDACT),
            "allowed": sum(1 for d in self._decision_log if d.action == FilterAction.ALLOW),
            "warned": sum(1 for d in self._decision_log if d.action == FilterAction.WARN),
            "avg_confidence": sum(d.confidence for d in self._decision_log)
            / len(self._decision_log),
            "avg_latency_ms": sum(d.latency_ms for d in self._decision_log)
            / len(self._decision_log),
        }
