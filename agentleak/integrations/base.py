"""
AgentLeak SDK - Base Integration Module

This module provides the abstract base class for all framework integrations.
All framework-specific integrations (CrewAI, LangChain, AutoGPT, MetaGPT)
inherit from this base class to ensure consistent behavior.

Architecture:
- BaseIntegration: Abstract class defining the integration interface
- IntegrationConfig: Configuration dataclass for all integrations
- IntegrationResult: Standardized result format

Extending AgentLeak to new frameworks:
    1. Create a new module in agentleak/integrations/
    2. Inherit from BaseIntegration
    3. Implement the required abstract methods
    4. Register in __init__.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class DetectionMode(str, Enum):
    """Detection mode for leak checking."""
    FAST = "FAST"           # Regex only (~10ms, 75% accuracy)
    STANDARD = "STANDARD"   # Regex + NLP (~100ms, 85% accuracy)
    HYBRID = "HYBRID"       # Regex + NLP + LLM fallback (~200ms, 92% accuracy)
    LLM_ONLY = "LLM_ONLY"   # Full LLM judge (~2s, 98% accuracy)


@dataclass
class IntegrationConfig:
    """Configuration for AgentLeak integrations."""
    
    # Vault: sensitive data to protect
    vault: Dict[str, str] = field(default_factory=dict)
    
    # Detection settings
    mode: DetectionMode = DetectionMode.HYBRID
    alert_threshold: float = 0.7
    
    # Behavior on leak
    raise_on_leak: bool = False
    block_on_leak: bool = False
    redact_on_leak: bool = False
    
    # Logging
    log_dir: Optional[str] = None
    log_level: str = "WARNING"
    
    # Callbacks
    on_leak_callback: Optional[Callable[[Dict], None]] = None


@dataclass
class LeakIncident:
    """Represents a detected leak incident."""
    timestamp: str
    framework: str
    channel: str
    leaked_items: List[str]
    confidence: float
    detection_tier: str
    content_preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "framework": self.framework,
            "channel": self.channel,
            "leaked_items": self.leaked_items,
            "confidence": self.confidence,
            "detection_tier": self.detection_tier,
            "content_preview": self.content_preview,
            "metadata": self.metadata
        }


@dataclass 
class IntegrationStats:
    """Statistics for integration monitoring."""
    total_checks: int = 0
    leaks_detected: int = 0
    high_confidence_leaks: int = 0
    incidents: List[LeakIncident] = field(default_factory=list)
    
    @property
    def leak_rate(self) -> float:
        return self.leaks_detected / self.total_checks if self.total_checks > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "leaks_detected": self.leaks_detected,
            "high_confidence_leaks": self.high_confidence_leaks,
            "leak_rate": self.leak_rate,
            "recent_incidents": [i.to_dict() for i in self.incidents[-5:]]
        }


class BaseIntegration(ABC):
    """
    Abstract base class for AgentLeak framework integrations.
    
    All framework integrations must implement:
    - attach(): Attach monitoring to framework
    - check_output(): Check a single output for leaks
    - get_stats(): Return monitoring statistics
    
    Subclasses should call super().__init__() and use self._check_for_leaks()
    for consistent leak detection behavior.
    """
    
    FRAMEWORK_NAME: str = "base"
    FRAMEWORK_VERSION: str = "0.0.0"
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.stats = IntegrationStats()
        self._tester = None
        self._setup_logging()
        self._init_tester()
        
    def _setup_logging(self):
        """Configure logging for this integration."""
        if self.config.log_dir:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_path / f"{self.FRAMEWORK_NAME}.log")
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.config.log_level))
    
    def _init_tester(self):
        """Initialize the AgentLeak tester."""
        try:
            from agentleak import AgentLeakTester
            self._tester = AgentLeakTester(mode=self.config.mode)
        except ImportError:
            logger.warning("AgentLeakTester not available, using mock tester")
            self._tester = None
    
    def _check_for_leaks(self, content: str, channel: str = "C1", 
                         metadata: Optional[Dict] = None) -> Optional[LeakIncident]:
        """
        Check content for leaks using AgentLeak detection.
        
        Args:
            content: The text content to check
            channel: The channel type (C1-C6)
            metadata: Additional context
            
        Returns:
            LeakIncident if leak detected, None otherwise
        """
        self.stats.total_checks += 1
        
        if self._tester is None:
            # Fallback: simple substring matching
            return self._simple_leak_check(content, channel, metadata)
        
        try:
            result = self._tester.check(
                vault=self.config.vault,
                output=content,
                channel=channel
            )
            
            if result.leaked:
                incident = LeakIncident(
                    timestamp=datetime.now().isoformat(),
                    framework=self.FRAMEWORK_NAME,
                    channel=channel,
                    leaked_items=result.detected_items,
                    confidence=result.confidence,
                    detection_tier=str(result.tier_used),
                    content_preview=content[:200],
                    metadata=metadata or {}
                )
                self._handle_leak(incident)
                return incident
                
        except Exception as e:
            logger.error(f"Leak check failed: {e}")
            if self.config.raise_on_leak:
                raise
        
        return None
    
    def _simple_leak_check(self, content: str, channel: str,
                           metadata: Optional[Dict] = None) -> Optional[LeakIncident]:
        """Fallback leak detection using simple substring matching."""
        content_lower = content.lower()
        leaked_items = []
        
        for key, value in self.config.vault.items():
            if str(value).lower() in content_lower:
                leaked_items.append(value)
        
        if leaked_items:
            incident = LeakIncident(
                timestamp=datetime.now().isoformat(),
                framework=self.FRAMEWORK_NAME,
                channel=channel,
                leaked_items=leaked_items,
                confidence=1.0,
                detection_tier="EXACT_MATCH",
                content_preview=content[:200],
                metadata=metadata or {}
            )
            self._handle_leak(incident)
            return incident
        
        return None
    
    def _handle_leak(self, incident: LeakIncident):
        """Handle a detected leak incident."""
        self.stats.leaks_detected += 1
        
        if incident.confidence >= self.config.alert_threshold:
            self.stats.high_confidence_leaks += 1
            severity = "🚨 HIGH"
        else:
            severity = "⚠️  LOW"
        
        self.stats.incidents.append(incident)
        
        logger.warning(
            f"{severity} LEAK [{self.FRAMEWORK_NAME}] {incident.channel}\n"
            f"  Items: {incident.leaked_items[:3]}\n"
            f"  Confidence: {incident.confidence:.2f}"
        )
        
        # Save to log file
        if self.config.log_dir:
            self._save_incident(incident)
        
        # Custom callback
        if self.config.on_leak_callback:
            try:
                self.config.on_leak_callback(incident.to_dict())
            except Exception as e:
                logger.error(f"Leak callback failed: {e}")
        
        # Raise if configured
        if self.config.raise_on_leak and incident.confidence >= self.config.alert_threshold:
            raise SecurityError(
                f"Leak detected in {self.FRAMEWORK_NAME}: {incident.leaked_items}"
            )
    
    def _save_incident(self, incident: LeakIncident):
        """Save incident to log directory."""
        log_path = Path(self.config.log_dir)
        incident_file = log_path / f"incident_{incident.timestamp.replace(':', '-')}.json"
        with open(incident_file, "w") as f:
            json.dump(incident.to_dict(), f, indent=2)
    
    @abstractmethod
    def attach(self, framework_object: Any) -> Any:
        """
        Attach AgentLeak monitoring to a framework object.
        
        Args:
            framework_object: The framework-specific object (Crew, Agent, etc.)
            
        Returns:
            The modified/wrapped framework object
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return monitoring statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset monitoring statistics."""
        self.stats = IntegrationStats()


class SecurityError(Exception):
    """Exception raised when a high-confidence leak is detected."""
    pass
