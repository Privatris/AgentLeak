"""
AgentLeak Output Sanitizer - Rule-based output filtering.

A simpler, regex-based defense that doesn't require training.
Useful as a baseline or fallback.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ..schemas.scenario import Channel
from .base import BaseDefense, DefenseConfig, DefenseResult, FilterAction


@dataclass
class SanitizerConfig(DefenseConfig):
    """Configuration for Output Sanitizer."""

    # Pattern types to detect
    detect_ssn: bool = True
    detect_phone: bool = True
    detect_email: bool = True
    detect_credit_card: bool = True
    detect_canaries: bool = True
    detect_iban: bool = True
    detect_crypto: bool = True
    detect_api_keys: bool = True

    # Redaction settings
    redaction_string: str = "[REDACTED]"

    # Custom patterns
    custom_patterns: list[str] = field(default_factory=list)


@dataclass
class SanitizationResult:
    """Result from sanitization."""

    original: str
    sanitized: str
    redactions_made: int
    patterns_detected: list[str]
    latency_ms: float


class OutputSanitizer(BaseDefense):
    """
    Rule-based output sanitizer.

    Uses regex patterns to detect and redact PII from output.
    No training required - works out of the box.

    Example:
        sanitizer = OutputSanitizer()
        result = sanitizer.filter("SSN: 123-45-6789", Channel.C1_FINAL_OUTPUT)
        print(result.filtered_content)  # "SSN: [REDACTED]"
    """

    def __init__(self, config: Optional[SanitizerConfig] = None):
        super().__init__(config or SanitizerConfig())
        self.config: SanitizerConfig = self.config
        self._patterns = self._build_patterns()

    @property
    def name(self) -> str:
        return "OutputSanitizer"

    def _build_patterns(self) -> dict[str, re.Pattern]:
        """Build regex patterns based on config."""
        patterns = {}

        if self.config.detect_ssn:
            patterns["ssn"] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

        if self.config.detect_phone:
            patterns["phone"] = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")

        if self.config.detect_email:
            patterns["email"] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

        if self.config.detect_credit_card:
            patterns["credit_card"] = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")

        if self.config.detect_canaries:
            patterns["canary"] = re.compile(r"CANARY_[A-Z]+_[A-Z0-9]{8}")

        if self.config.detect_iban:
            # Simplified IBAN (2 letters, 2 digits, then alphanumeric up to 30 chars)
            patterns["iban"] = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
            
        if self.config.detect_crypto:
            # Ethereum-style (0x...) and Bitcoin-style (1..., 3..., bc1...) simple checks
            patterns["crypto"] = re.compile(r"\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59})\b")

        if self.config.detect_api_keys:
            # Generic API keys (sk-..., ghp_..., etc)
            patterns["api_key"] = re.compile(r"\b(sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}|mz_[a-zA-Z0-9]{24,})\b")

        # Add custom patterns
        for i, pattern_str in enumerate(self.config.custom_patterns):
            patterns[f"custom_{i}"] = re.compile(pattern_str)

        return patterns

    def filter(self, content: str, channel: Channel, metadata: Optional[dict[str, Any]] = None) -> DefenseResult:
        """
        Apply sanitization filter.
        
        Implements BaseDefense interface with optional metadata.
        """
        if not self.should_monitor(channel):
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
            )

        # Sanitize using build-in logic
        res = self.sanitize(content)
        
        # Prepare result
        if res.redactions_made > 0:
            result = DefenseResult(
                action=FilterAction.REDACT,
                original_content=res.original,
                filtered_content=res.sanitized,
                detected_patterns=res.patterns_detected,
                latency_ms=res.latency_ms,
                confidence=1.0,
                reason=f"Redacted {res.redactions_made} patterns"
            )
        else:
            result = DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
                latency_ms=res.latency_ms
            )
            
        self._log_decision(result)
        return result

    def sanitize(self, content: str) -> SanitizationResult:
        """
        Sanitize content by redacting detected patterns.

        Returns detailed SanitizationResult.
        """
        start_time = time.time()

        sanitized = content
        patterns_detected = []
        redactions = 0

        for pattern_name, pattern in self._patterns.items():
            matches = pattern.findall(content)
            if matches:
                patterns_detected.extend([f"{pattern_name}:{m}" for m in matches])
                redactions += len(matches)
                sanitized = pattern.sub(self.config.redaction_string, sanitized)

        latency_ms = (time.time() - start_time) * 1000

        return SanitizationResult(
            original=content,
            sanitized=sanitized,
            redactions_made=redactions,
            patterns_detected=patterns_detected,
            latency_ms=latency_ms,
        )

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a custom pattern at runtime."""
        self._patterns[name] = re.compile(pattern)

    def remove_pattern(self, name: str) -> None:
        """Remove a pattern."""
        if name in self._patterns:
            del self._patterns[name]
