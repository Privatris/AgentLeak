"""
AgentLeak Metrics Aggregator - Aggregate metrics across multiple runs.

Provides statistical aggregation with confidence intervals.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from ..schemas.scenario import Vertical, Channel, AttackClass
from ..schemas.results import DetectionResult
from .core import MetricsCalculator, BenchmarkMetrics, LeakageMetrics


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    mean: float
    std: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n: int  # Sample size
    
    @classmethod
    def from_values(cls, values: list[float], confidence: float = 0.95) -> "ConfidenceInterval":
        """Calculate confidence interval from values."""
        if not values:
            return cls(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0, n=0)
        
        n = len(values)
        mean = sum(values) / n
        
        if n == 1:
            return cls(mean=mean, std=0.0, ci_lower=mean, ci_upper=mean, n=n)
        
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        
        # Z-score for 95% confidence
        z = 1.96 if confidence == 0.95 else 2.576  # 99%
        margin = z * std / math.sqrt(n)
        
        return cls(
            mean=mean,
            std=std,
            ci_lower=max(0.0, mean - margin),
            ci_upper=min(1.0, mean + margin),
            n=n,
        )


@dataclass
class VerticalBreakdown:
    """Metrics breakdown by vertical."""
    vertical: Vertical
    scenario_count: int
    elr: ConfidenceInterval
    wls: ConfidenceInterval
    tsr: ConfidenceInterval
    asr: Optional[ConfidenceInterval] = None


@dataclass
class ChannelBreakdown:
    """Metrics breakdown by channel."""
    channel: Channel
    leak_count: int
    leak_rate: ConfidenceInterval
    weighted_score: ConfidenceInterval


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across multiple benchmark runs.
    
    Includes confidence intervals for all key metrics.
    """
    # Overall metrics with CIs
    elr: ConfidenceInterval
    wls: ConfidenceInterval
    tsr: ConfidenceInterval
    asr: ConfidenceInterval
    
    # Breakdowns
    by_vertical: list[VerticalBreakdown]
    by_channel: list[ChannelBreakdown]
    
    # Totals
    total_scenarios: int
    total_runs: int
    
    # Raw values for further analysis
    elr_values: list[float] = field(default_factory=list)
    tsr_values: list[float] = field(default_factory=list)


class MetricsAggregator:
    """
    Aggregate metrics across multiple benchmark runs.
    
    Provides:
    - Mean and std for all metrics
    - 95% confidence intervals
    - Breakdown by vertical and channel
    
    Example:
        aggregator = MetricsAggregator()
        
        # Add results from multiple runs
        for run in runs:
            aggregator.add_results(run.results, run.name)
        
        # Get aggregated metrics
        metrics = aggregator.aggregate()
        print(f"ELR: {metrics.elr.mean:.2%} ± {metrics.elr.std:.2%}")
    """
    
    def __init__(self):
        """Initialize aggregator."""
        self.calculator = MetricsCalculator()
        self.runs: list[tuple[str, list[DetectionResult]]] = []
    
    def add_results(
        self,
        results: list[DetectionResult],
        run_name: str = "run",
    ) -> None:
        """
        Add results from a benchmark run.
        
        Args:
            results: Detection results from the run
            run_name: Name/identifier for this run
        """
        self.runs.append((run_name, results))
    
    def clear(self) -> None:
        """Clear all accumulated results."""
        self.runs.clear()
    
    def aggregate(
        self,
        total_private_fields_per_scenario: int = 10,
    ) -> AggregatedMetrics:
        """
        Aggregate all accumulated results.
        
        Args:
            total_private_fields_per_scenario: Fields per scenario for ELR
            
        Returns:
            AggregatedMetrics with confidence intervals
        """
        if not self.runs:
            return self._empty_metrics()
        
        # Calculate per-run metrics
        run_metrics: list[BenchmarkMetrics] = []
        for run_name, results in self.runs:
            metrics = self.calculator.aggregate(results, total_private_fields_per_scenario)
            metrics.name = run_name
            run_metrics.append(metrics)
        
        # Aggregate across runs
        elr_values = [m.leakage.elr for m in run_metrics]
        wls_values = [m.leakage.wls for m in run_metrics]
        tsr_values = [m.tasks.tsr for m in run_metrics]
        asr_values = [m.attacks.asr for m in run_metrics if m.attacks.total_attacks > 0]
        
        # Per-vertical breakdown
        vertical_breakdown = self._aggregate_by_vertical(run_metrics)
        
        # Per-channel breakdown
        channel_breakdown = self._aggregate_by_channel(run_metrics)
        
        return AggregatedMetrics(
            elr=ConfidenceInterval.from_values(elr_values),
            wls=ConfidenceInterval.from_values(wls_values),
            tsr=ConfidenceInterval.from_values(tsr_values),
            asr=ConfidenceInterval.from_values(asr_values) if asr_values else ConfidenceInterval(0, 0, 0, 0, 0),
            by_vertical=vertical_breakdown,
            by_channel=channel_breakdown,
            total_scenarios=sum(m.total_scenarios for m in run_metrics),
            total_runs=len(self.runs),
            elr_values=elr_values,
            tsr_values=tsr_values,
        )
    
    def _aggregate_by_vertical(
        self,
        run_metrics: list[BenchmarkMetrics],
    ) -> list[VerticalBreakdown]:
        """Aggregate metrics by vertical."""
        # This would require scenario metadata
        # For now, return empty (will be filled when scenarios have vertical info)
        return []
    
    def _aggregate_by_channel(
        self,
        run_metrics: list[BenchmarkMetrics],
    ) -> list[ChannelBreakdown]:
        """Aggregate metrics by channel."""
        channel_data: dict[Channel, list[tuple[int, float]]] = {}
        
        for metrics in run_metrics:
            for ch_metrics in metrics.channels:
                if ch_metrics.channel not in channel_data:
                    channel_data[ch_metrics.channel] = []
                channel_data[ch_metrics.channel].append(
                    (ch_metrics.leak_count, ch_metrics.weighted_score)
                )
        
        breakdowns = []
        for channel, data in channel_data.items():
            leak_counts = [d[0] for d in data]
            weighted_scores = [d[1] for d in data]
            total_leaks = sum(leak_counts)
            
            # Normalize leak rates
            total_all_leaks = sum(
                sum(d[0] for d in channel_data.get(ch, []))
                for ch in channel_data
            )
            leak_rates = [lc / max(1, total_all_leaks) for lc in leak_counts]
            
            breakdowns.append(ChannelBreakdown(
                channel=channel,
                leak_count=total_leaks,
                leak_rate=ConfidenceInterval.from_values(leak_rates),
                weighted_score=ConfidenceInterval.from_values(weighted_scores),
            ))
        
        return breakdowns
    
    def _empty_metrics(self) -> AggregatedMetrics:
        """Return empty metrics when no runs available."""
        return AggregatedMetrics(
            elr=ConfidenceInterval(0, 0, 0, 0, 0),
            wls=ConfidenceInterval(0, 0, 0, 0, 0),
            tsr=ConfidenceInterval(0, 0, 0, 0, 0),
            asr=ConfidenceInterval(0, 0, 0, 0, 0),
            by_vertical=[],
            by_channel=[],
            total_scenarios=0,
            total_runs=0,
        )
    
    # ================================================================
    # Reporting
    # ================================================================
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        metrics = self.aggregate()
        
        lines = [
            "=" * 50,
            "agentleak Benchmark Metrics Summary",
            "=" * 50,
            "",
            f"Total Runs: {metrics.total_runs}",
            f"Total Scenarios: {metrics.total_scenarios}",
            "",
            "Core Metrics:",
            f"  ELR: {metrics.elr.mean:.2%} ± {metrics.elr.std:.2%} (95% CI: [{metrics.elr.ci_lower:.2%}, {metrics.elr.ci_upper:.2%}])",
            f"  WLS: {metrics.wls.mean:.3f} ± {metrics.wls.std:.3f}",
            f"  TSR: {metrics.tsr.mean:.2%} ± {metrics.tsr.std:.2%}",
            f"  ASR: {metrics.asr.mean:.2%} ± {metrics.asr.std:.2%}",
            "",
        ]
        
        if metrics.by_channel:
            lines.append("Channel Breakdown:")
            for ch in metrics.by_channel:
                lines.append(f"  {ch.channel.value}: {ch.leak_count} leaks ({ch.leak_rate.mean:.2%})")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary (for JSON export)."""
        metrics = self.aggregate()
        
        return {
            "elr": {"mean": metrics.elr.mean, "std": metrics.elr.std, "ci": [metrics.elr.ci_lower, metrics.elr.ci_upper]},
            "wls": {"mean": metrics.wls.mean, "std": metrics.wls.std},
            "tsr": {"mean": metrics.tsr.mean, "std": metrics.tsr.std, "ci": [metrics.tsr.ci_lower, metrics.tsr.ci_upper]},
            "asr": {"mean": metrics.asr.mean, "std": metrics.asr.std},
            "total_scenarios": metrics.total_scenarios,
            "total_runs": metrics.total_runs,
        }
