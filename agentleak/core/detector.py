"""
Core detection interface for AgentLeak.

Unified detection API supporting multiple strategies:
- Tier 1: Exact matching (canary detection)
- Tier 2: Pattern matching (regex-based)
- Tier 3: Semantic similarity (embeddings)
- Tier 4: LLM-as-Judge (contextual reasoning)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class DetectionTier(str, Enum):
    """Detection tiers in order of sophistication."""
    EXACT = "exact"          # Tier 1: Canary/exact match
    PATTERN = "pattern"      # Tier 2: Regex patterns
    SEMANTIC = "semantic"    # Tier 3: Embedding similarity
    LLM = "llm"             # Tier 4: LLM reasoning


@dataclass
class DetectionResult:
    """Result from a privacy detection check."""
    is_leaked: bool
    confidence: float  # 0.0-1.0
    tier: DetectionTier
    channel: str
    detected_items: List[str]
    explanation: str
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Detector(ABC):
    """Base class for all detectors."""
    
    @property
    @abstractmethod
    def tier(self) -> DetectionTier:
        """Return the detection tier."""
        pass
    
    @abstractmethod
    def detect(
        self,
        vault: Dict[str, Any],
        output: str,
        channel: str
    ) -> DetectionResult:
        """
        Check if output leaks information from vault.
        
        Args:
            vault: Private/sensitive data store
            output: Agent output to check
            channel: Leakage channel (C1-C7)
            
        Returns:
            DetectionResult with verdict and evidence
        """
        pass
    
    def should_skip(self, previous_result: Optional[DetectionResult]) -> bool:
        """
        Determine if this detector should skip based on previous result.
        
        Default: Skip if previous tier already found a leak with high confidence.
        """
        if previous_result is None:
            return False
        
        # Skip if already found leak with confidence > 0.9
        return previous_result.is_leaked and previous_result.confidence > 0.9


class DetectorPipeline:
    """
    Sequential detection pipeline with early stopping.
    
    Runs detectors in order of increasing cost/sophistication:
    1. Exact match (fastest, cheapest)
    2. Pattern match (fast, cheap)
    3. Semantic similarity (moderate)
    4. LLM reasoning (slow, expensive)
    
    Stops early if high-confidence leak detected.
    """
    
    def __init__(
        self,
        detectors: List[Detector],
        enable_early_stopping: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            detectors: List of detectors in order of execution
            enable_early_stopping: Stop after high-confidence detection
        """
        self.detectors = sorted(detectors, key=lambda d: list(DetectionTier).index(d.tier))
        self.enable_early_stopping = enable_early_stopping
    
    def detect(
        self,
        vault: Dict[str, Any],
        output: str,
        channel: str
    ) -> DetectionResult:
        """
        Run detection pipeline.
        
        Args:
            vault: Private data store
            output: Agent output
            channel: Channel identifier
            
        Returns:
            Final detection result (from highest tier that ran)
        """
        previous_result = None
        final_result = None
        
        for detector in self.detectors:
            # Skip if previous tier already found high-confidence leak
            if self.enable_early_stopping and detector.should_skip(previous_result):
                continue
            
            result = detector.detect(vault, output, channel)
            final_result = result
            previous_result = result
            
            # Early stop if leak found with high confidence
            if self.enable_early_stopping and result.is_leaked and result.confidence > 0.9:
                break
        
        return final_result or DetectionResult(
            is_leaked=False,
            confidence=1.0,
            tier=DetectionTier.EXACT,
            channel=channel,
            detected_items=[],
            explanation="No leak detected (pipeline empty)"
        )
