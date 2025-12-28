"""
AgentLeak Output Sanitizer - Rule-based output filtering.

A simpler, regex-based defense that doesn't require training.
Useful as a baseline or fallback.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re
import time

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
            patterns["ssn"] = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        
        if self.config.detect_phone:
            patterns["phone"] = re.compile(r'\b\d{3}-\d{3}-\d{4}\b')
        
        if self.config.detect_email:
            patterns["email"] = re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            )
        
        if self.config.detect_credit_card:
            patterns["credit_card"] = re.compile(
                r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
            )
        
        if self.config.detect_canaries:
            patterns["canary"] = re.compile(r'CANARY_[A-Z]+_[A-Z0-9]{8}')
        
        # Add custom patterns
        for i, pattern_str in enumerate(self.config.custom_patterns):
            patterns[f"custom_{i}"] = re.compile(pattern_str)
        
        return patterns
    
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
    
    def filter(self, content: str, channel: Channel) -> DefenseResult:
        """
        Filter content using sanitization.
        
        Implements BaseDefense interface.
        """
        if not self.should_monitor(channel):
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
            )
        
        result = self.sanitize(content)
        
        if result.redactions_made > 0:
            action = FilterAction.REDACT
            reason = f"Redacted {result.redactions_made} patterns"
        else:
            action = FilterAction.ALLOW
            reason = None
        
        defense_result = DefenseResult(
            action=action,
            original_content=content,
            filtered_content=result.sanitized if result.redactions_made > 0 else None,
            confidence=1.0 if result.redactions_made > 0 else 0.0,
            reason=reason,
            detected_patterns=result.patterns_detected,
            latency_ms=result.latency_ms,
        )
        
        self._log_decision(defense_result)
        return defense_result
    
    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a custom pattern at runtime."""
        self._patterns[name] = re.compile(pattern)
    
    def remove_pattern(self, name: str) -> None:
        """Remove a pattern."""
        if name in self._patterns:
            del self._patterns[name]
