"""
AgentLeak Detection Results Schema - Data structures for leakage detection results.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .scenario import Channel, AttackClass, CanaryTier


class FieldLeak(BaseModel):
    """A single detected field leak."""

    field_path: str = Field(..., description="e.g., 'customer.ssn'")
    leaked_value: str
    channel: Channel
    event_sequence: int = Field(..., description="Which event in trace")
    detection_method: str = Field(..., description="canary|pattern|semantic")
    confidence: float = Field(ge=0.0, le=1.0)
    sensitivity_weight: float = Field(default=1.0)
    canary_tier: Optional[CanaryTier] = None

    model_config = {"extra": "forbid"}


class ChannelResult(BaseModel):
    """Detection results for a single channel."""

    channel: Channel
    leaked: bool
    leak_count: int = 0
    fields_leaked: list[str] = Field(default_factory=list)
    weighted_score: float = 0.0

    model_config = {"extra": "forbid"}


class DetectionResult(BaseModel):
    """
    Complete detection result for a scenario execution.

    This is what the detection pipeline produces.
    """

    # Identification
    scenario_id: str
    trace_id: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    # Overall results
    leaked: bool = Field(..., description="Any leakage detected?")
    total_leaks: int = 0

    # Per-channel breakdown
    channel_results: dict[Channel, ChannelResult] = Field(default_factory=dict)

    # Detailed leaks
    field_leaks: list[FieldLeak] = Field(default_factory=list)

    # Metrics
    exact_leakage_rate: float = Field(0.0, ge=0.0, le=1.0, alias="elr")
    weighted_leakage_score: float = Field(0.0, ge=0.0, alias="wls")

    # Attack-specific (if applicable)
    attack_class: Optional[AttackClass] = None
    attack_success: bool = False

    # Task success
    task_success: Optional[bool] = None
    task_success_reason: Optional[str] = None

    # Detection pipeline info
    detection_stages_used: list[str] = Field(default_factory=list)
    semantic_threshold: Optional[float] = None

    model_config = {"extra": "forbid", "populate_by_name": True}

    def get_leaked_channels(self) -> list[Channel]:
        """Get list of channels where leakage was detected."""
        return [ch for ch, result in self.channel_results.items() if result.leaked]

    def get_leaked_fields(self) -> list[str]:
        """Get list of all leaked field paths."""
        return list(set(leak.field_path for leak in self.field_leaks))


class BenchmarkResults(BaseModel):
    """
    Aggregate results for a benchmark run.
    """

    # Identification
    run_id: str
    benchmark_name: str = Field(default="agentleak")
    run_at: datetime = Field(default_factory=datetime.utcnow)

    # Configuration
    framework: str
    model: str
    defense: Optional[str] = None
    defense_params: dict = Field(default_factory=dict)

    # Counts
    total_scenarios: int
    completed_scenarios: int
    failed_scenarios: int = 0

    # Aggregate metrics
    task_success_rate: float = Field(0.0, ge=0.0, le=1.0, alias="tsr")
    exact_leakage_rate: float = Field(0.0, ge=0.0, le=1.0, alias="elr")
    weighted_leakage_score: float = Field(0.0, ge=0.0, alias="wls")
    attack_success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, alias="asr")

    # Per-channel leakage rates
    channel_leakage_rates: dict[str, float] = Field(default_factory=dict, alias="clr")

    # Pareto metrics
    pareto_auc: Optional[float] = None
    dominance_rate: Optional[float] = None

    # Cost metrics
    total_tokens: int = 0
    total_steps: int = 0
    avg_tokens_per_scenario: float = 0.0
    avg_steps_per_scenario: float = 0.0

    # Individual results
    scenario_results: list[DetectionResult] = Field(default_factory=list)

    model_config = {"extra": "forbid", "populate_by_name": True}


if __name__ == "__main__":
    # Quick test
    result = DetectionResult(
        scenario_id="agentleak_fin_00001",
        trace_id="trace_001",
        leaked=True,
        total_leaks=2,
        field_leaks=[
            FieldLeak(
                field_path="customer.ssn",
                leaked_value="CANARY_SSN_7F3Q",
                channel=Channel.C3_TOOL_INPUT,
                event_sequence=3,
                detection_method="canary",
                confidence=1.0,
                sensitivity_weight=5.0,
                canary_tier=CanaryTier.OBVIOUS,
            )
        ],
        weighted_leakage_score=5.0,
    )

    print(f"✓ Detection result: leaked={result.leaked}")
    print(f"  Total leaks: {result.total_leaks}")
    print(f"  WLS: {result.weighted_leakage_score}")
    print(f"  Leaked fields: {result.get_leaked_fields()}")
