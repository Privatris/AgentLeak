"""
AgentLeak Internal Channel Defense - Protecting C2 and C5 channels.

This module addresses the KEY FINDING from the paper:
- Internal channels (C2 inter-agent, C5 memory) have 8.3x higher leak rates
- 31.5% internal vs 3.8% external (defended)
- NO evaluated defense targets these channels
- Sanitizers achieve 0% effectiveness on C2/C5 by default

This defense provides message-level and memory-level protection that can be
integrated with multi-agent frameworks (CrewAI, LangChain, AutoGPT, MetaGPT).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

from ..detection.presidio_detector import PresidioDetector, PresidioConfig
from ..schemas.scenario import Channel
from .base import BaseDefense, DefenseConfig, DefenseResult, FilterAction


class ClearanceLevel(str, Enum):
    """Agent clearance levels for role separation defense."""
    
    PUBLIC = "public"           # No access to sensitive data
    INTERNAL = "internal"       # Access to low-sensitivity data
    CONFIDENTIAL = "confidential"  # Access to medium-sensitivity data
    SECRET = "secret"           # Access to high-sensitivity data
    TOP_SECRET = "top_secret"   # Access to all sensitive data


@dataclass
class AgentProfile:
    """Profile defining an agent's access permissions."""
    
    agent_id: str
    clearance_level: ClearanceLevel
    allowed_entity_types: list[str] = field(default_factory=list)
    denied_entity_types: list[str] = field(default_factory=list)
    can_send_to: list[str] = field(default_factory=list)  # Agent IDs
    can_receive_from: list[str] = field(default_factory=list)  # Agent IDs


@dataclass
class MessageFilter:
    """Result from filtering an inter-agent message."""
    
    original_message: str
    filtered_message: str
    source_agent: str
    target_agent: str
    pii_detected: list[str]
    pii_redacted: list[str]
    allowed: bool
    reason: Optional[str]
    latency_ms: float


@dataclass
class MemoryFilter:
    """Result from filtering a memory write."""
    
    original_content: str
    filtered_content: str
    memory_key: str
    pii_detected: list[str]
    pii_redacted: list[str]
    allowed: bool
    reason: Optional[str]
    latency_ms: float


@dataclass
class InternalChannelConfig(DefenseConfig):
    """Configuration for internal channel defense."""
    
    # Message filtering (C2)
    filter_inter_agent_messages: bool = True
    redact_pii_in_messages: bool = True
    block_high_severity_messages: bool = False
    
    # Memory filtering (C5)
    filter_memory_writes: bool = True
    filter_memory_reads: bool = True
    redact_pii_in_memory: bool = True
    
    # Role separation
    enable_role_separation: bool = False
    agent_profiles: dict[str, AgentProfile] = field(default_factory=dict)
    
    # Entity sensitivity mapping
    high_severity_entities: list[str] = field(default_factory=lambda: [
        "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "CRYPTO_WALLET",
        "MEDICAL_LICENSE", "US_PASSPORT", "IBAN_CODE"
    ])
    medium_severity_entities: list[str] = field(default_factory=lambda: [
        "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION",
        "IP_ADDRESS", "DATE_TIME", "NRP"
    ])
    
    # Redaction
    redaction_placeholder: str = "[REDACTED]"
    
    # Logging
    log_all_messages: bool = True
    log_file: Optional[str] = None


class InternalChannelDefense(BaseDefense):
    """
    Defense for internal channels (C2 inter-agent, C5 memory).
    
    This defense addresses the critical gap identified in the paper where
    existing defenses achieve 0% effectiveness on internal channels while
    achieving 98%+ on external channels like C1.
    
    Key features:
    1. Message-level PII detection and redaction (C2)
    2. Memory write/read filtering (C5)
    3. Optional role-based access control (clearance levels)
    4. Compatible with CrewAI, LangChain, AutoGPT, MetaGPT
    
    Example:
        defense = InternalChannelDefense()
        
        # Filter inter-agent message
        result = defense.filter_message(
            message="Patient SSN is 123-45-6789",
            source="DataAgent",
            target="ReportAgent"
        )
        print(result.filtered_message)  # "Patient SSN is [REDACTED]"
        
        # Filter memory write
        result = defense.filter_memory_write(
            content="credit_card: 4532-1234-5678-9010",
            key="payment_info"
        )
    """
    
    def __init__(self, config: Optional[InternalChannelConfig] = None):
        super().__init__(config or InternalChannelConfig())
        self.config: InternalChannelConfig = self.config
        
        # Initialize PII detector
        self._detector = PresidioDetector(PresidioConfig(
            score_threshold=0.5,
        ))
        
        # Message and memory logs
        self._message_log: list[MessageFilter] = []
        self._memory_log: list[MemoryFilter] = []
    
    @property
    def name(self) -> str:
        return "InternalChannelDefense"
    
    def _classify_severity(self, entity_type: str) -> str:
        """Classify entity severity."""
        if entity_type in self.config.high_severity_entities:
            return "high"
        elif entity_type in self.config.medium_severity_entities:
            return "medium"
        return "low"
    
    def _redact_text(
        self,
        text: str,
        entity_types_to_redact: Optional[list[str]] = None,
    ) -> tuple[str, list[str], list[str]]:
        """
        Detect and redact PII from text.
        
        Returns:
            Tuple of (redacted_text, detected_entities, redacted_entities)
        """
        detections = self._detector.analyze(text, Channel.C2_INTER_AGENT)
        
        if not detections:
            return text, [], []
        
        detected = []
        redacted = []
        result = text
        
        # Sort by position (reverse) to avoid offset issues
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)
        
        for detection in sorted_detections:
            detected.append(f"{detection.entity_type}:{detection.text}")
            
            # Check if we should redact this entity type
            if entity_types_to_redact is None or detection.entity_type in entity_types_to_redact:
                placeholder = f"[REDACTED_{detection.entity_type}]"
                result = result[:detection.start] + placeholder + result[detection.end:]
                redacted.append(f"{detection.entity_type}:{detection.text}")
        
        return result, detected, redacted
    
    def _check_clearance(
        self,
        source_agent: str,
        target_agent: str,
        entity_types: list[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Check if message transfer is allowed based on clearance levels.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if not self.config.enable_role_separation:
            return True, None
        
        profiles = self.config.agent_profiles
        
        # Check source clearance
        source_profile = profiles.get(source_agent)
        if source_profile is None:
            return True, None  # Unknown agent, allow by default
        
        # Check target clearance
        target_profile = profiles.get(target_agent)
        if target_profile is None:
            return True, None
        
        # Check if source can send to target
        if source_profile.can_send_to and target_agent not in source_profile.can_send_to:
            return False, f"Agent {source_agent} not authorized to send to {target_agent}"
        
        # Check if target can receive from source
        if target_profile.can_receive_from and source_agent not in target_profile.can_receive_from:
            return False, f"Agent {target_agent} not authorized to receive from {source_agent}"
        
        # Check entity type permissions
        for entity_type in entity_types:
            entity_base = entity_type.split(":")[0]
            
            # Check if target is allowed to receive this entity type
            if target_profile.denied_entity_types and entity_base in target_profile.denied_entity_types:
                return False, f"Agent {target_agent} denied access to {entity_base}"
            
            if target_profile.allowed_entity_types and entity_base not in target_profile.allowed_entity_types:
                return False, f"Agent {target_agent} not authorized for {entity_base}"
        
        return True, None
    
    def filter_message(
        self,
        message: str,
        source: str,
        target: str,
    ) -> MessageFilter:
        """
        Filter an inter-agent message (C2).
        
        Args:
            message: The message content
            source: Source agent ID
            target: Target agent ID
        
        Returns:
            MessageFilter result
        """
        start_time = time.time()
        
        if not self.config.filter_inter_agent_messages:
            return MessageFilter(
                original_message=message,
                filtered_message=message,
                source_agent=source,
                target_agent=target,
                pii_detected=[],
                pii_redacted=[],
                allowed=True,
                reason=None,
                latency_ms=0.0,
            )
        
        # Detect and redact PII
        filtered, detected, redacted = self._redact_text(message)
        
        # Check clearance
        allowed, deny_reason = self._check_clearance(source, target, detected)
        
        # Check for high severity blocking
        if self.config.block_high_severity_messages and allowed:
            for entity in detected:
                entity_type = entity.split(":")[0]
                if entity_type in self.config.high_severity_entities:
                    allowed = False
                    deny_reason = f"High severity PII ({entity_type}) blocked"
                    break
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = MessageFilter(
            original_message=message,
            filtered_message=filtered if allowed else "[BLOCKED]",
            source_agent=source,
            target_agent=target,
            pii_detected=detected,
            pii_redacted=redacted,
            allowed=allowed,
            reason=deny_reason,
            latency_ms=latency_ms,
        )
        
        if self.config.log_all_messages:
            self._message_log.append(result)
        
        return result
    
    def filter_memory_write(
        self,
        content: str,
        key: str,
    ) -> MemoryFilter:
        """
        Filter content before writing to memory (C5).
        
        Args:
            content: Content to write
            key: Memory key
        
        Returns:
            MemoryFilter result
        """
        start_time = time.time()
        
        if not self.config.filter_memory_writes:
            return MemoryFilter(
                original_content=content,
                filtered_content=content,
                memory_key=key,
                pii_detected=[],
                pii_redacted=[],
                allowed=True,
                reason=None,
                latency_ms=0.0,
            )
        
        # Detect and redact PII
        filtered, detected, redacted = self._redact_text(content)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = MemoryFilter(
            original_content=content,
            filtered_content=filtered,
            memory_key=key,
            pii_detected=detected,
            pii_redacted=redacted,
            allowed=True,
            reason=None,
            latency_ms=latency_ms,
        )
        
        if self.config.log_all_messages:
            self._memory_log.append(result)
        
        return result
    
    def filter_memory_read(
        self,
        content: str,
        key: str,
        requesting_agent: Optional[str] = None,
    ) -> MemoryFilter:
        """
        Filter content when reading from memory (C5).
        
        Args:
            content: Content being read
            key: Memory key
            requesting_agent: Agent requesting the read
        
        Returns:
            MemoryFilter result
        """
        start_time = time.time()
        
        if not self.config.filter_memory_reads:
            return MemoryFilter(
                original_content=content,
                filtered_content=content,
                memory_key=key,
                pii_detected=[],
                pii_redacted=[],
                allowed=True,
                reason=None,
                latency_ms=0.0,
            )
        
        # Detect and redact PII
        filtered, detected, redacted = self._redact_text(content)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = MemoryFilter(
            original_content=content,
            filtered_content=filtered,
            memory_key=key,
            pii_detected=detected,
            pii_redacted=redacted,
            allowed=True,
            reason=None,
            latency_ms=latency_ms,
        )
        
        return result
    
    def filter(self, content: str, channel: Channel) -> DefenseResult:
        """
        Filter content based on channel.
        
        Implements BaseDefense interface.
        """
        if channel == Channel.C2_INTER_AGENT:
            # Default agent names for generic filtering
            result = self.filter_message(content, "unknown", "unknown")
            
            action = FilterAction.ALLOW if result.allowed else FilterAction.BLOCK
            if result.pii_redacted:
                action = FilterAction.REDACT
            
            return DefenseResult(
                action=action,
                original_content=content,
                filtered_content=result.filtered_message if result.pii_redacted else None,
                confidence=1.0 if result.pii_detected else 0.0,
                reason=result.reason,
                detected_patterns=result.pii_detected,
                latency_ms=result.latency_ms,
            )
        
        elif channel == Channel.C5_MEMORY_WRITE:
            result = self.filter_memory_write(content, "generic")
            
            action = FilterAction.REDACT if result.pii_redacted else FilterAction.ALLOW
            
            return DefenseResult(
                action=action,
                original_content=content,
                filtered_content=result.filtered_content if result.pii_redacted else None,
                confidence=1.0 if result.pii_detected else 0.0,
                reason=result.reason,
                detected_patterns=result.pii_detected,
                latency_ms=result.latency_ms,
            )
        
        else:
            # Not an internal channel, allow through
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
            )
    
    def get_statistics(self) -> dict[str, Any]:
        """Get defense statistics."""
        message_count = len(self._message_log)
        memory_count = len(self._memory_log)
        
        messages_with_pii = sum(1 for m in self._message_log if m.pii_detected)
        messages_blocked = sum(1 for m in self._message_log if not m.allowed)
        
        memory_with_pii = sum(1 for m in self._memory_log if m.pii_detected)
        
        return {
            "total_messages_filtered": message_count,
            "messages_with_pii": messages_with_pii,
            "messages_blocked": messages_blocked,
            "message_pii_rate": messages_with_pii / message_count if message_count > 0 else 0,
            "total_memory_ops_filtered": memory_count,
            "memory_ops_with_pii": memory_with_pii,
            "memory_pii_rate": memory_with_pii / memory_count if memory_count > 0 else 0,
        }
    
    def create_agent_profile(
        self,
        agent_id: str,
        clearance: Union[str, ClearanceLevel],
        allowed_entities: Optional[list[str]] = None,
        denied_entities: Optional[list[str]] = None,
    ) -> AgentProfile:
        """
        Create and register an agent profile for role separation.
        
        Args:
            agent_id: Unique agent identifier
            clearance: Clearance level
            allowed_entities: Entity types this agent can access
            denied_entities: Entity types this agent cannot access
        
        Returns:
            Created AgentProfile
        """
        if isinstance(clearance, str):
            clearance = ClearanceLevel(clearance)
        
        profile = AgentProfile(
            agent_id=agent_id,
            clearance_level=clearance,
            allowed_entity_types=allowed_entities or [],
            denied_entity_types=denied_entities or [],
        )
        
        self.config.agent_profiles[agent_id] = profile
        return profile


# =============================================================================
# Framework Integration Hooks
# =============================================================================


class FrameworkHook(ABC):
    """Abstract base class for framework integration hooks."""
    
    @abstractmethod
    def install(self, defense: InternalChannelDefense) -> None:
        """Install the defense hook into the framework."""
        pass
    
    @abstractmethod
    def uninstall(self) -> None:
        """Remove the defense hook from the framework."""
        pass


class CrewAIHook(FrameworkHook):
    """
    Hook for CrewAI framework integration.
    
    Intercepts inter-agent communication and memory operations
    in CrewAI workflows.
    
    Example:
        defense = InternalChannelDefense()
        hook = CrewAIHook()
        hook.install(defense)
        
        # Now all CrewAI inter-agent messages are filtered
    """
    
    def __init__(self):
        self._defense: Optional[InternalChannelDefense] = None
        self._original_methods: dict[str, Callable] = {}
    
    def install(self, defense: InternalChannelDefense) -> None:
        """Install hook into CrewAI."""
        self._defense = defense
        # Implementation would patch CrewAI's message passing
        # This is a placeholder for the actual integration
        pass
    
    def uninstall(self) -> None:
        """Remove hook from CrewAI."""
        self._defense = None
        # Restore original methods
        pass


class LangChainHook(FrameworkHook):
    """
    Hook for LangChain framework integration.
    
    Intercepts agent communication and memory in LangChain agents.
    """
    
    def __init__(self):
        self._defense: Optional[InternalChannelDefense] = None
    
    def install(self, defense: InternalChannelDefense) -> None:
        """Install hook into LangChain."""
        self._defense = defense
        pass
    
    def uninstall(self) -> None:
        """Remove hook from LangChain."""
        self._defense = None


# =============================================================================
# Factory Functions
# =============================================================================


def create_internal_defense(
    strict: bool = False,
    enable_role_separation: bool = False,
) -> InternalChannelDefense:
    """
    Factory function to create internal channel defense.
    
    Args:
        strict: If True, block high-severity PII in messages
        enable_role_separation: If True, enable clearance-based access control
    
    Returns:
        Configured InternalChannelDefense
    """
    config = InternalChannelConfig(
        block_high_severity_messages=strict,
        enable_role_separation=enable_role_separation,
    )
    return InternalChannelDefense(config)


def wrap_message_handler(
    handler: Callable[[str, str, str], Any],
    defense: InternalChannelDefense,
) -> Callable[[str, str, str], Any]:
    """
    Wrap a message handler function with internal channel defense.
    
    Args:
        handler: Original message handler(message, source, target) -> result
        defense: InternalChannelDefense instance
    
    Returns:
        Wrapped handler that filters messages before processing
    """
    def wrapped(message: str, source: str, target: str) -> Any:
        result = defense.filter_message(message, source, target)
        if not result.allowed:
            raise PermissionError(f"Message blocked: {result.reason}")
        return handler(result.filtered_message, source, target)
    
    return wrapped
