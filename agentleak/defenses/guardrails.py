"""
AgentLeak Guardrails Integration - Wrappers for external guardrail systems.

This module provides integration with production-grade guardrail systems:
1. LLM Guard (ProtectAI) - Open-source, comprehensive
2. Guardrails AI - PII detection with fix actions
3. NeMo Guardrails (NVIDIA) - Policy-based
4. LlamaGuard 3 (Meta) - Safety classifier
5. Rebuff - Multi-layer defense

Based on the paper's evaluation of 5 guardrail systems in Section 9.3.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..schemas.scenario import Channel
from .base import BaseDefense, DefenseConfig, DefenseResult, FilterAction


class GuardrailType(str, Enum):
    """Types of external guardrail systems."""
    
    LLM_GUARD = "llm_guard"
    GUARDRAILS_AI = "guardrails_ai"
    NEMO_GUARDRAILS = "nemo_guardrails"
    LLAMA_GUARD = "llama_guard"
    REBUFF = "rebuff"
    LAKERA = "lakera"


@dataclass
class GuardrailConfig(DefenseConfig):
    """Configuration for guardrail integration."""
    
    guardrail_type: GuardrailType = GuardrailType.LLM_GUARD
    
    # LLM Guard settings
    llm_guard_entities: list[str] = field(default_factory=lambda: [
        "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE",
        "IP_ADDRESS", "PERSON", "PHONE_NUMBER", "US_SSN"
    ])
    llm_guard_threshold: float = 0.5
    llm_guard_use_faker: bool = False
    
    # Guardrails AI settings
    guardrails_ai_pii_entities: list[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"
    ])
    guardrails_ai_on_fail: str = "fix"  # fix, exception, filter
    
    # General settings
    fallback_to_presidio: bool = True  # Use Presidio if guardrail unavailable
    log_guardrail_calls: bool = True


@dataclass
class GuardrailResult:
    """Result from guardrail check."""
    
    original: str
    sanitized: str
    is_valid: bool
    risk_score: float
    scanner_results: dict[str, Any]
    latency_ms: float
    guardrail_type: GuardrailType


class GuardrailBase(ABC):
    """Abstract base class for guardrail integrations."""
    
    @abstractmethod
    def scan_input(self, prompt: str) -> GuardrailResult:
        """Scan input prompt."""
        pass
    
    @abstractmethod
    def scan_output(self, prompt: str, output: str) -> GuardrailResult:
        """Scan LLM output."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if guardrail is installed and available."""
        pass


class LLMGuardIntegration(GuardrailBase):
    """
    Integration with ProtectAI's LLM Guard.
    
    LLM Guard provides:
    - Anonymize scanner (input): Detect and replace PII
    - Sensitive scanner (output): Detect data leakage
    - PromptInjection scanner: Detect injection attacks
    - Toxicity scanner: Detect harmful content
    
    Requires: pip install llm-guard
    
    Example:
        guard = LLMGuardIntegration()
        
        # Scan input
        result = guard.scan_input("My SSN is 123-45-6789")
        print(result.sanitized)  # "My SSN is [REDACTED_US_SSN]"
        
        # Scan output
        result = guard.scan_output(prompt, "Your email is john@example.com")
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self._vault = None
        self._input_scanners = None
        self._output_scanners = None
        self._initialized = False
    
    def _initialize(self) -> bool:
        """Initialize LLM Guard components."""
        if self._initialized:
            return True
        
        try:
            from llm_guard import scan_prompt, scan_output
            from llm_guard.input_scanners import Anonymize, PromptInjection, Toxicity
            from llm_guard.output_scanners import Deanonymize, Sensitive
            from llm_guard.vault import Vault
            
            # Create vault for anonymization
            self._vault = Vault()
            
            # Configure input scanners
            self._input_scanners = [
                Anonymize(
                    self._vault,
                    entity_types=self.config.llm_guard_entities,
                    threshold=self.config.llm_guard_threshold,
                    use_faker=self.config.llm_guard_use_faker,
                ),
                PromptInjection(threshold=0.9),
                Toxicity(threshold=0.5),
            ]
            
            # Configure output scanners
            self._output_scanners = [
                Deanonymize(self._vault),
                Sensitive(
                    threshold=self.config.llm_guard_threshold,
                    entity_types=self.config.llm_guard_entities,
                    redact=True,
                ),
            ]
            
            self._scan_prompt = scan_prompt
            self._scan_output = scan_output
            self._initialized = True
            return True
            
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        """Check if LLM Guard is available."""
        try:
            import llm_guard
            return True
        except ImportError:
            return False
    
    def scan_input(self, prompt: str) -> GuardrailResult:
        """Scan input with LLM Guard."""
        start_time = time.time()
        
        if not self._initialize():
            return GuardrailResult(
                original=prompt,
                sanitized=prompt,
                is_valid=True,
                risk_score=0.0,
                scanner_results={"error": "LLM Guard not available"},
                latency_ms=0.0,
                guardrail_type=GuardrailType.LLM_GUARD,
            )
        
        sanitized, results_valid, results_score = self._scan_prompt(
            self._input_scanners,
            prompt
        )
        
        is_valid = all(results_valid.values())
        max_risk = max(results_score.values()) if results_score else 0.0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GuardrailResult(
            original=prompt,
            sanitized=sanitized,
            is_valid=is_valid,
            risk_score=max_risk,
            scanner_results={
                "valid": results_valid,
                "scores": results_score,
            },
            latency_ms=latency_ms,
            guardrail_type=GuardrailType.LLM_GUARD,
        )
    
    def scan_output(self, prompt: str, output: str) -> GuardrailResult:
        """Scan output with LLM Guard."""
        start_time = time.time()
        
        if not self._initialize():
            return GuardrailResult(
                original=output,
                sanitized=output,
                is_valid=True,
                risk_score=0.0,
                scanner_results={"error": "LLM Guard not available"},
                latency_ms=0.0,
                guardrail_type=GuardrailType.LLM_GUARD,
            )
        
        sanitized, results_valid, results_score = self._scan_output(
            self._output_scanners,
            prompt,
            output
        )
        
        is_valid = all(results_valid.values())
        max_risk = max(results_score.values()) if results_score else 0.0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return GuardrailResult(
            original=output,
            sanitized=sanitized,
            is_valid=is_valid,
            risk_score=max_risk,
            scanner_results={
                "valid": results_valid,
                "scores": results_score,
            },
            latency_ms=latency_ms,
            guardrail_type=GuardrailType.LLM_GUARD,
        )


class GuardrailsAIIntegration(GuardrailBase):
    """
    Integration with Guardrails AI.
    
    Guardrails AI provides:
    - DetectPII validator: Detect PII with configurable entities
    - Fix action: Automatically redact detected PII
    - Exception action: Raise error on PII detection
    
    Requires:
        pip install guardrails-ai
        pip install presidio-analyzer presidio-anonymizer
        guardrails hub install hub://guardrails/detect_pii
    
    Example:
        guard = GuardrailsAIIntegration()
        
        result = guard.scan_output("", "Email: john@example.com")
        print(result.sanitized)  # "Email: [REDACTED]"
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self._guard = None
        self._initialized = False
    
    def _initialize(self) -> bool:
        """Initialize Guardrails AI."""
        if self._initialized:
            return True
        
        try:
            from guardrails import Guard
            from guardrails.hub import DetectPII
            
            on_fail_action = self.config.guardrails_ai_on_fail
            
            self._guard = Guard().use(
                DetectPII(
                    pii_entities=self.config.guardrails_ai_pii_entities,
                    on_fail=on_fail_action
                )
            )
            
            self._initialized = True
            return True
            
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        """Check if Guardrails AI is available."""
        try:
            import guardrails
            return True
        except ImportError:
            return False
    
    def scan_input(self, prompt: str) -> GuardrailResult:
        """Scan input with Guardrails AI."""
        return self.scan_output("", prompt)  # Same logic for input
    
    def scan_output(self, prompt: str, output: str) -> GuardrailResult:
        """Scan output with Guardrails AI."""
        start_time = time.time()
        
        if not self._initialize():
            return GuardrailResult(
                original=output,
                sanitized=output,
                is_valid=True,
                risk_score=0.0,
                scanner_results={"error": "Guardrails AI not available"},
                latency_ms=0.0,
                guardrail_type=GuardrailType.GUARDRAILS_AI,
            )
        
        try:
            result = self._guard.validate(output)
            sanitized = result.validated_output or output
            is_valid = result.validation_passed
            
            latency_ms = (time.time() - start_time) * 1000
            
            return GuardrailResult(
                original=output,
                sanitized=sanitized,
                is_valid=is_valid,
                risk_score=0.0 if is_valid else 1.0,
                scanner_results={"validation_passed": is_valid},
                latency_ms=latency_ms,
                guardrail_type=GuardrailType.GUARDRAILS_AI,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return GuardrailResult(
                original=output,
                sanitized=output,
                is_valid=False,
                risk_score=1.0,
                scanner_results={"error": str(e)},
                latency_ms=latency_ms,
                guardrail_type=GuardrailType.GUARDRAILS_AI,
            )


class PresidioFallback(GuardrailBase):
    """
    Fallback to Presidio when external guardrails are not available.
    
    Uses the internal Presidio detector as a reliable fallback.
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self._detector = None
    
    def _initialize(self) -> bool:
        """Initialize Presidio detector."""
        if self._detector is None:
            from ..detection.presidio_detector import PresidioDetector, PresidioConfig
            self._detector = PresidioDetector(PresidioConfig(
                score_threshold=0.5,
            ))
        return True
    
    def is_available(self) -> bool:
        """Presidio is always available (internal)."""
        return True
    
    def _redact(self, text: str) -> tuple[str, list[Any]]:
        """Detect and redact PII."""
        self._initialize()
        detections = self._detector.analyze(text)
        
        if not detections:
            return text, []
        
        result = text
        sorted_detections = sorted(detections, key=lambda d: d.start, reverse=True)
        
        for detection in sorted_detections:
            placeholder = f"[REDACTED_{detection.entity_type}]"
            result = result[:detection.start] + placeholder + result[detection.end:]
        
        return result, detections
    
    def scan_input(self, prompt: str) -> GuardrailResult:
        """Scan input with Presidio."""
        start_time = time.time()
        
        sanitized, detections = self._redact(prompt)
        latency_ms = (time.time() - start_time) * 1000
        
        return GuardrailResult(
            original=prompt,
            sanitized=sanitized,
            is_valid=len(detections) == 0,
            risk_score=max((d.score for d in detections), default=0.0),
            scanner_results={
                "detections": [{"type": d.entity_type, "text": d.text} for d in detections]
            },
            latency_ms=latency_ms,
            guardrail_type=GuardrailType.LLM_GUARD,  # Fallback
        )
    
    def scan_output(self, prompt: str, output: str) -> GuardrailResult:
        """Scan output with Presidio."""
        return self.scan_input(output)


class GuardrailDefense(BaseDefense):
    """
    Unified guardrail defense with automatic fallback.
    
    Tries to use the configured guardrail system, falling back to
    Presidio if unavailable.
    
    Example:
        # Try LLM Guard, fall back to Presidio
        defense = GuardrailDefense(GuardrailConfig(
            guardrail_type=GuardrailType.LLM_GUARD,
            fallback_to_presidio=True
        ))
        
        result = defense.filter("SSN: 123-45-6789", Channel.C1_FINAL_OUTPUT)
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        super().__init__(config or GuardrailConfig())
        self.config: GuardrailConfig = self.config
        
        # Initialize guardrails
        self._guardrails: dict[GuardrailType, GuardrailBase] = {
            GuardrailType.LLM_GUARD: LLMGuardIntegration(self.config),
            GuardrailType.GUARDRAILS_AI: GuardrailsAIIntegration(self.config),
        }
        
        self._fallback = PresidioFallback(self.config)
        self._active_guardrail: Optional[GuardrailBase] = None
    
    @property
    def name(self) -> str:
        return f"GuardrailDefense({self.config.guardrail_type.value})"
    
    def _get_guardrail(self) -> GuardrailBase:
        """Get the active guardrail, with fallback."""
        if self._active_guardrail is not None:
            return self._active_guardrail
        
        # Try configured guardrail
        guardrail = self._guardrails.get(self.config.guardrail_type)
        if guardrail and guardrail.is_available():
            self._active_guardrail = guardrail
            return guardrail
        
        # Fallback to Presidio
        if self.config.fallback_to_presidio:
            self._active_guardrail = self._fallback
            return self._fallback
        
        raise RuntimeError(
            f"Guardrail {self.config.guardrail_type.value} not available "
            "and fallback disabled"
        )
    
    def filter(self, content: str, channel: Channel, metadata: Optional[dict[str, Any]] = None) -> DefenseResult:
        """Filter content using guardrail."""
        if not self.should_monitor(channel):
            return DefenseResult(
                action=FilterAction.ALLOW,
                original_content=content,
            )
        
        guardrail = self._get_guardrail()
        
        # Use input scan for incoming content, output scan for outgoing
        if channel in [Channel.C1_FINAL_OUTPUT, Channel.C2_INTER_AGENT]:
            result = guardrail.scan_output("", content)
        else:
            result = guardrail.scan_input(content)
        
        # Determine action
        if not result.is_valid and result.sanitized != content:
            action = FilterAction.REDACT
            reason = f"PII detected and redacted (risk: {result.risk_score:.2f})"
        elif not result.is_valid:
            action = FilterAction.BLOCK
            reason = f"Content blocked (risk: {result.risk_score:.2f})"
        else:
            action = FilterAction.ALLOW
            reason = None
        
        defense_result = DefenseResult(
            action=action,
            original_content=content,
            filtered_content=result.sanitized if result.sanitized != content else None,
            confidence=result.risk_score,
            reason=reason,
            detected_patterns=[],  # Guardrails don't always provide this
            latency_ms=result.latency_ms,
        )
        
        if self.config.log_decisions:
            self._log_decision(defense_result)
        
        return defense_result
    
    def get_available_guardrails(self) -> list[GuardrailType]:
        """List available guardrail systems."""
        available = []
        for gtype, guardrail in self._guardrails.items():
            if guardrail.is_available():
                available.append(gtype)
        return available


# =============================================================================
# Factory Functions
# =============================================================================


def create_guardrail_defense(
    guardrail_type: str = "llm_guard",
    fallback: bool = True,
) -> GuardrailDefense:
    """
    Factory function to create guardrail defense.
    
    Args:
        guardrail_type: One of: llm_guard, guardrails_ai, nemo_guardrails, etc.
        fallback: If True, fall back to Presidio if guardrail unavailable
    
    Returns:
        Configured GuardrailDefense
    """
    config = GuardrailConfig(
        guardrail_type=GuardrailType(guardrail_type),
        fallback_to_presidio=fallback,
    )
    return GuardrailDefense(config)


def get_best_available_guardrail() -> GuardrailDefense:
    """
    Get the best available guardrail defense.
    
    Tries guardrails in order of preference:
    1. LLM Guard (comprehensive, open-source)
    2. Guardrails AI (good PII detection)
    3. Presidio fallback (always available)
    
    Returns:
        GuardrailDefense with best available backend
    """
    preference_order = [
        GuardrailType.LLM_GUARD,
        GuardrailType.GUARDRAILS_AI,
    ]
    
    for gtype in preference_order:
        config = GuardrailConfig(
            guardrail_type=gtype,
            fallback_to_presidio=True,
        )
        defense = GuardrailDefense(config)
        if gtype in defense.get_available_guardrails():
            return defense
    
    # Fallback to Presidio
    return GuardrailDefense(GuardrailConfig(fallback_to_presidio=True))
