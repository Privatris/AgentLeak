"""
Legacy compatibility layer for llm_judge.

Maps the old LLMJudge API to the new GeminiJudge.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import from new architecture
from .gemini_judge import GeminiJudge, GeminiJudgeConfig
from ..core.detector import Detector, DetectionResult, DetectionTier


@dataclass
class LLMConfig:
    """Configuration for LLM-as-Judge (legacy compatibility)."""
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 30
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENROUTER_API_KEY")


class LLMJudge(Detector):
    """Tier 4: LLM-based semantic analysis.
    
    Legacy compatibility - now uses GeminiJudge internally.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._gemini = GeminiJudge(GeminiJudgeConfig(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ))
    
    @property
    def tier(self) -> DetectionTier:
        """Return the detection tier."""
        return DetectionTier.LLM
    
    def detect(
        self,
        vault: Dict[str, Any],
        output: str,
        channel: str
    ) -> DetectionResult:
        """Analyze output for semantic leakage."""
        import time
        start = time.time()
        
        # Convert vault to format expected by GeminiJudge
        result = self._gemini.evaluate(
            agent_output=output,
            privacy_vault=vault,
            context=f"Channel: {channel}"
        )
        
        is_leaked = result.has_leakage if hasattr(result, 'has_leakage') else False
        confidence = result.confidence if hasattr(result, 'confidence') else 0.0
        detected_items = [l.leaked_value for l in result.leaks] if hasattr(result, 'leaks') else []
        explanation = result.explanation if hasattr(result, 'explanation') else "No leakage detected"
        
        return DetectionResult(
            is_leaked=is_leaked,
            confidence=confidence,
            tier=self.tier,
            channel=channel,
            detected_items=detected_items,
            explanation=explanation,
            latency_ms=(time.time() - start) * 1000
        )
