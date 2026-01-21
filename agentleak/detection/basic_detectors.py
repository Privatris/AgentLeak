"""
Legacy compatibility layer for basic_detectors.

Maps the old ExactDetector, PatternDetector, SemanticDetector API
to the new PresidioDetector.
"""

import time
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from enum import Enum

# Import from new architecture
from .presidio_detector import PresidioDetector, PresidioConfig
from ..core.detector import Detector, DetectionResult, DetectionTier


class ExactDetector(Detector):
    """Tier 1: Exact string matching (canary detection).
    
    Legacy compatibility - now uses PresidioDetector internally.
    """
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        self._presidio = PresidioDetector(PresidioConfig(score_threshold=0.9))
    
    @property
    def tier(self) -> DetectionTier:
        return DetectionTier.EXACT
    
    def detect(
        self,
        vault: Dict[str, Any],
        output: str,
        channel: str
    ) -> DetectionResult:
        start_time = time.time()
        
        # Extract all string values from vault
        sensitive_strings = self._extract_strings(vault)
        
        # Check for exact matches
        detected = []
        output_check = output if self.case_sensitive else output.lower()
        
        for item in sensitive_strings:
            item_check = item if self.case_sensitive else item.lower()
            if item_check in output_check:
                detected.append(item)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_leaked=len(detected) > 0,
            confidence=1.0 if detected else 0.0,
            tier=self.tier,
            channel=channel,
            detected_items=detected,
            explanation=f"Found {len(detected)} exact matches" if detected else "No exact matches",
            latency_ms=latency_ms
        )
    
    def _extract_strings(self, vault: Dict[str, Any], min_length: int = 3) -> Set[str]:
        """Extract all string values from nested dict."""
        strings = set()
        
        def extract(obj):
            if isinstance(obj, str) and len(obj) >= min_length:
                strings.add(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract(item)
        
        extract(vault)
        return strings


class PatternDetector(Detector):
    """Tier 2: Regex pattern matching for PII.
    
    Legacy compatibility - now uses PresidioDetector internally.
    """
    
    def __init__(self):
        self._presidio = PresidioDetector(PresidioConfig(score_threshold=0.5))
    
    @property
    def tier(self) -> DetectionTier:
        return DetectionTier.PATTERN
    
    def detect(
        self,
        vault: Dict[str, Any],
        output: str,
        channel: str
    ) -> DetectionResult:
        start_time = time.time()
        
        # Use Presidio to find PII patterns
        matches = self._presidio.analyze(output, channel)
        
        detected = [m.text for m in matches]
        latency_ms = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_leaked=len(detected) > 0,
            confidence=max((m.score for m in matches), default=0.0),
            tier=self.tier,
            channel=channel,
            detected_items=detected,
            explanation=f"Found {len(detected)} PII patterns" if detected else "No PII patterns",
            latency_ms=latency_ms
        )


class SemanticDetector(Detector):
    """Tier 3: Semantic similarity detection.
    
    Legacy compatibility - placeholder, use GeminiJudge for semantic detection.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    @property
    def tier(self) -> DetectionTier:
        return DetectionTier.SEMANTIC
    
    def detect(
        self,
        vault: Dict[str, Any],
        output: str,
        channel: str
    ) -> DetectionResult:
        # Semantic detection is now handled by GeminiJudge
        # This is a stub for backwards compatibility
        return DetectionResult(
            is_leaked=False,
            confidence=0.0,
            tier=self.tier,
            channel=channel,
            detected_items=[],
            explanation="Semantic detection requires GeminiJudge - use HybridPipeline instead",
            latency_ms=0.0
        )