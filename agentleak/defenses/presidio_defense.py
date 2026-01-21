"""
AgentLeak Presidio Defense - PII detection and redaction using Presidio.

This defense leverages the Presidio detector from the detection module to:
1. Detect PII in agent inputs/outputs
2. Redact sensitive information before disclosure
3. Support all 7 channels (C1-C7) including internal channels

Based on the paper's finding that internal channels (C2, C5) have 8.3x higher
leak rates than external channels but NO existing defense covers them.

This defense bridges that gap by applying Presidio to ALL channels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..detection.presidio_detector import PresidioDetector, PresidioConfig, PresidioMatch
from ..schemas.scenario import Channel
from .base import BaseDefense, DefenseConfig, DefenseResult, FilterAction


class RedactionStyle(str, Enum):
    """Style for redacting sensitive data."""
    
    PLACEHOLDER = "placeholder"      # [REDACTED_SSN]
    ASTERISK = "asterisk"            # ****
    MASKED = "masked"                # XXX-XX-6789 (partial)
    HASH = "hash"                    # SHA256 hash
    CATEGORY = "category"            # [PII: SSN]
    REMOVE = "remove"                # Remove entirely


@dataclass
class PresidioDefenseConfig(DefenseConfig):
    """Configuration for Presidio-based defense."""
    
    # Detection settings
    score_threshold: float = 0.5
    entities_to_detect: Optional[list[str]] = None  # None = all entities
    
    # Redaction settings
    redaction_style: RedactionStyle = RedactionStyle.PLACEHOLDER
    custom_placeholder: str = "[REDACTED]"
    
    # Channel settings
    protect_internal_channels: bool = True  # C2, C5 (key gap identified in paper)
    protect_external_channels: bool = True  # C1, C3, C4, C6, C7
    
    # Performance
    cache_enabled: bool = True
    max_cache_size: int = 1000
    
    # Behavior
    block_on_high_severity: bool = False
    high_severity_entities: list[str] = field(default_factory=lambda: [
        "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "CRYPTO_WALLET",
        "MEDICAL_LICENSE", "US_PASSPORT", "IBAN_CODE"
    ])


@dataclass
class PresidioDefenseResult:
    """Detailed result from Presidio defense."""
    
    original: str
    sanitized: str
    detections: list[PresidioMatch]
    redactions_made: int
    latency_ms: float
    channel: Channel
    blocked: bool = False
    block_reason: Optional[str] = None
    
    @property
    def has_pii(self) -> bool:
        return len(self.detections) > 0
    
    @property
    def entity_types_found(self) -> list[str]:
        return list(set(d.entity_type for d in self.detections))


class PresidioDefense(BaseDefense):
    """
    Presidio-based privacy defense.
    
    Uses Microsoft Presidio (via our detection module) to detect and redact
    PII from agent communications across ALL channels.
    
    Key differentiator: Works on internal channels (C2, C5) which existing
    guardrails don't protect (paper finding: 31.5% leak rate on internal
    channels vs 3.8% on defended external channels).
    
    Example:
        defense = PresidioDefense()
        
        # Protect output (C1)
        result = defense.filter("SSN: 123-45-6789", Channel.C1_FINAL_OUTPUT)
        print(result.filtered_content)  # "SSN: [REDACTED_SSN]"
        
        # Protect inter-agent message (C2) - key gap in existing defenses
        result = defense.filter(
            "The patient's diagnosis is diabetes",
            Channel.C2_INTER_AGENT
        )
    """
    
    def __init__(self, config: Optional[PresidioDefenseConfig] = None):
        super().__init__(config or PresidioDefenseConfig())
        self.config: PresidioDefenseConfig = self.config
        
        # Initialize Presidio detector
        presidio_config = PresidioConfig(
            score_threshold=self.config.score_threshold,
            entities=self.config.entities_to_detect,
        )
        self._detector = PresidioDetector(presidio_config)
        
        # Cache for repeated content
        self._cache: dict[str, PresidioDefenseResult] = {}
    
    @property
    def name(self) -> str:
        return "PresidioDefense"
    
    def should_protect_channel(self, channel: Channel) -> bool:
        """Check if this channel should be protected."""
        internal_channels = {Channel.C2_INTER_AGENT, Channel.C5_MEMORY_WRITE}
        external_channels = {
            Channel.C1_FINAL_OUTPUT, Channel.C3_TOOL_INPUT,
            Channel.C4_TOOL_OUTPUT, Channel.C6_LOG, Channel.C7_ARTIFACT
        }
        
        if channel in internal_channels:
            return self.config.protect_internal_channels
        if channel in external_channels:
            return self.config.protect_external_channels
        
        return True  # Default: protect unknown channels
    
    def _apply_redaction(
        self,
        text: str,
        detections: list[PresidioMatch],
    ) -> str:
        """Apply redaction to detected PII."""
        if not detections:
            return text
        
        # Sort by position (reverse) to avoid offset issues
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)
        
        result = text
        for detection in sorted_detections:
            replacement = self._get_replacement(detection)
            result = result[:detection.start] + replacement + result[detection.end:]
        
        return result
    
    def _get_replacement(self, detection: PresidioMatch) -> str:
        """Get replacement string based on redaction style."""
        style = self.config.redaction_style
        
        if style == RedactionStyle.PLACEHOLDER:
            return f"[REDACTED_{detection.entity_type}]"
        
        elif style == RedactionStyle.ASTERISK:
            return "*" * len(detection.text)
        
        elif style == RedactionStyle.MASKED:
            # Partial masking - show last 4 chars for some types
            if len(detection.text) > 4:
                return "X" * (len(detection.text) - 4) + detection.text[-4:]
            return "X" * len(detection.text)
        
        elif style == RedactionStyle.HASH:
            import hashlib
            hash_val = hashlib.sha256(detection.text.encode()).hexdigest()[:8]
            return f"[HASH:{hash_val}]"
        
        elif style == RedactionStyle.CATEGORY:
            return f"[PII: {detection.entity_type}]"
        
        elif style == RedactionStyle.REMOVE:
            return ""
        
        else:
            return self.config.custom_placeholder
    
    def _should_block(self, detections: list[PresidioMatch]) -> tuple[bool, Optional[str]]:
        """Check if content should be blocked based on detections."""
        if not self.config.block_on_high_severity:
            return False, None
        
        high_severity = self.config.high_severity_entities
        for detection in detections:
            if detection.entity_type in high_severity:
                return True, f"High severity PII detected: {detection.entity_type}"
        
        return False, None
    
    def analyze(self, content: str, channel: Channel) -> PresidioDefenseResult:
        """
        Analyze content for PII and apply redaction.
        
        Returns detailed PresidioDefenseResult.
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{content}:{channel.value}"
        if self.config.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.latency_ms = 0.1  # Cache hit
            return cached
        
        # Detect PII
        detections = self._detector.analyze(content, channel)
        
        # Check for blocking
        blocked, block_reason = self._should_block(detections)
        
        # Apply redaction
        if blocked:
            sanitized = "[BLOCKED: Sensitive content detected]"
        else:
            sanitized = self._apply_redaction(content, detections)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = PresidioDefenseResult(
            original=content,
            sanitized=sanitized,
            detections=detections,
            redactions_made=len(detections),
            latency_ms=latency_ms,
            channel=channel,
            blocked=blocked,
            block_reason=block_reason,
        )
        
        # Cache result
        if self.config.cache_enabled and len(self._cache) < self.config.max_cache_size:
            self._cache[cache_key] = result
        
        return result
    
    def filter(self, content: str, channel: Channel) -> DefenseResult:
        """
        Filter content using Presidio detection.
        
        Implements BaseDefense interface.
        """
        if not self.should_protect_channel(channel):
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
            )
        
        result = self.analyze(content, channel)
        
        # Determine action
        if result.blocked:
            action = FilterAction.BLOCK
            reason = result.block_reason
        elif result.redactions_made > 0:
            action = FilterAction.REDACT
            reason = f"Redacted {result.redactions_made} PII entities: {result.entity_types_found}"
        else:
            action = FilterAction.ALLOW
            reason = None
        
        defense_result = DefenseResult(
            action=action,
            original_content=content,
            filtered_content=result.sanitized if result.redactions_made > 0 else None,
            confidence=max((d.score for d in result.detections), default=0.0),
            reason=reason,
            detected_patterns=[f"{d.entity_type}:{d.text}" for d in result.detections],
            latency_ms=result.latency_ms,
        )
        
        # Log decision
        if self.config.log_decisions:
            self._decision_log.append(defense_result)
        
        return defense_result
    
    def protect_message(
        self,
        message: str,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> tuple[str, bool]:
        """
        Protect an inter-agent message (C2).
        
        This is the key gap identified in the paper - internal channels
        have 8.3x higher leak rates but no defense covers them.
        
        Args:
            message: The message content
            source_agent: Optional source agent name
            target_agent: Optional target agent name
        
        Returns:
            Tuple of (sanitized_message, had_pii)
        """
        result = self.filter(message, Channel.C2_INTER_AGENT)
        return (
            result.filtered_content or message,
            result.action in [FilterAction.REDACT, FilterAction.BLOCK]
        )
    
    def protect_memory(
        self,
        content: str,
        memory_key: Optional[str] = None,
    ) -> tuple[str, bool]:
        """
        Protect content before writing to memory (C5).
        
        Another key internal channel with no existing defense coverage.
        
        Args:
            content: Content to store in memory
            memory_key: Optional memory key/label
        
        Returns:
            Tuple of (sanitized_content, had_pii)
        """
        result = self.filter(content, Channel.C5_MEMORY_WRITE)
        return (
            result.filtered_content or content,
            result.action in [FilterAction.REDACT, FilterAction.BLOCK]
        )
    
    def get_statistics(self) -> dict[str, Any]:
        """Get defense statistics."""
        if not self._decision_log:
            return {"total_checks": 0}
        
        total = len(self._decision_log)
        blocked = sum(1 for r in self._decision_log if r.action == FilterAction.BLOCK)
        redacted = sum(1 for r in self._decision_log if r.action == FilterAction.REDACT)
        allowed = sum(1 for r in self._decision_log if r.action == FilterAction.ALLOW)
        
        avg_latency = sum(r.latency_ms for r in self._decision_log) / total
        
        # Entity type distribution
        entity_counts: dict[str, int] = {}
        for result in self._decision_log:
            for pattern in result.detected_patterns:
                entity_type = pattern.split(":")[0]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return {
            "total_checks": total,
            "blocked": blocked,
            "redacted": redacted,
            "allowed": allowed,
            "block_rate": blocked / total if total > 0 else 0,
            "redaction_rate": redacted / total if total > 0 else 0,
            "avg_latency_ms": avg_latency,
            "entity_distribution": entity_counts,
            "cache_size": len(self._cache),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_presidio_defense(
    strict: bool = False,
    internal_only: bool = False,
    external_only: bool = False,
    redaction_style: str = "placeholder",
) -> PresidioDefense:
    """
    Factory function to create Presidio defense with common configurations.
    
    Args:
        strict: If True, block on high-severity PII
        internal_only: Only protect internal channels (C2, C5)
        external_only: Only protect external channels (C1, C3, C4, C6, C7)
        redaction_style: One of: placeholder, asterisk, masked, hash, category, remove
    
    Returns:
        Configured PresidioDefense instance
    """
    config = PresidioDefenseConfig(
        block_on_high_severity=strict,
        protect_internal_channels=not external_only,
        protect_external_channels=not internal_only,
        redaction_style=RedactionStyle(redaction_style),
    )
    return PresidioDefense(config)


def create_internal_channel_defense() -> PresidioDefense:
    """
    Create a defense specifically for internal channels (C2, C5).
    
    This addresses the key gap identified in the paper:
    - Internal channels have 31.5% leak rate (undefended)
    - External channels have 3.8% leak rate (defended)
    - 8.3x difference because no existing defense covers C2/C5
    
    Returns:
        PresidioDefense configured for internal channels only
    """
    config = PresidioDefenseConfig(
        protect_internal_channels=True,
        protect_external_channels=False,
        block_on_high_severity=True,  # Stricter for internal channels
        high_severity_entities=[
            "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "CRYPTO_WALLET",
            "MEDICAL_LICENSE", "US_PASSPORT", "IBAN_CODE", "PATIENT_ID",
            "DIAGNOSIS", "PRESCRIPTION"
        ],
    )
    return PresidioDefense(config)
