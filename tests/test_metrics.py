"""
Tests for agentleak Metrics Module.

Tests cover:
- MetricsCalculator (ELR, WLS, ASR, TSR)
- MetricsAggregator (confidence intervals, breakdowns)
- ParetoCalculator (frontier, AUC, hypervolume)
"""

import pytest
import math
from datetime import datetime

from agentleak.schemas.scenario import (
    Channel, AttackClass, AttackFamily, CanaryTier, Vertical,
)
from agentleak.schemas.results import DetectionResult, FieldLeak, ChannelResult

from agentleak.metrics.core import (
    MetricsCalculator,
    LeakageMetrics,
    ChannelMetrics,
    AttackMetrics,
    TaskMetrics,
    BenchmarkMetrics,
)
from agentleak.metrics.aggregator import (
    MetricsAggregator,
    AggregatedMetrics,
    ConfidenceInterval,
)
from agentleak.metrics.pareto import (
    ParetoCalculator,
    ParetoPoint,
    ParetoFrontier,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_field_leaks():
    """Create sample field leaks for testing."""
    return [
        FieldLeak(
            field_path="patient.ssn",
            leaked_value="CANARY_SSN_12345678",
            channel=Channel.C1_FINAL_OUTPUT,
            event_sequence=0,
            detection_method="canary",
            confidence=1.0,
            sensitivity_weight=1.0,
            canary_tier=CanaryTier.OBVIOUS,
        ),
        FieldLeak(
            field_path="patient.phone",
            leaked_value="555-0123",
            channel=Channel.C3_TOOL_INPUT,
            event_sequence=1,
            detection_method="pattern",
            confidence=0.9,
            sensitivity_weight=0.6,
            canary_tier=CanaryTier.REALISTIC,
        ),
        FieldLeak(
            field_path="patient.diagnosis",
            leaked_value="diabetes diagnosis",
            channel=Channel.C1_FINAL_OUTPUT,
            event_sequence=2,
            detection_method="semantic",
            confidence=0.85,
            sensitivity_weight=1.0,
            canary_tier=CanaryTier.SEMANTIC,
        ),
    ]


@pytest.fixture
def sample_detection_result(sample_field_leaks):
    """Create a sample detection result."""
    return DetectionResult(
        scenario_id="test_001",
        trace_id="trace_001",
        leaked=True,
        total_leaks=len(sample_field_leaks),
        field_leaks=sample_field_leaks,
        exact_leakage_rate=0.3,
        weighted_leakage_score=2.5,
        channel_results={
            Channel.C1_FINAL_OUTPUT: ChannelResult(
                channel=Channel.C1_FINAL_OUTPUT,
                leaked=True,
                leak_count=2,
                fields_leaked=["patient.ssn", "patient.diagnosis"],
                weighted_score=1.85,
            ),
            Channel.C3_TOOL_INPUT: ChannelResult(
                channel=Channel.C3_TOOL_INPUT,
                leaked=True,
                leak_count=1,
                fields_leaked=["patient.phone"],
                weighted_score=0.378,
            ),
        },
    )


@pytest.fixture
def multiple_detection_results():
    """Create multiple detection results for aggregation."""
    results = []
    
    # Result 1: Some leakage, task success
    results.append(DetectionResult(
        scenario_id="scen_001",
        trace_id="trace_001",
        leaked=True,
        total_leaks=2,
        field_leaks=[
            FieldLeak(
                field_path="ssn",
                leaked_value="123",
                channel=Channel.C1_FINAL_OUTPUT,
                event_sequence=0,
                detection_method="canary",
                confidence=1.0,
            ),
            FieldLeak(
                field_path="phone",
                leaked_value="456",
                channel=Channel.C3_TOOL_INPUT,
                event_sequence=1,
                detection_method="pattern",
                confidence=0.8,
            ),
        ],
        exact_leakage_rate=0.2,
        weighted_leakage_score=1.5,
        task_success=True,
    ))
    
    # Result 2: No leakage, task success
    results.append(DetectionResult(
        scenario_id="scen_002",
        trace_id="trace_002",
        leaked=False,
        total_leaks=0,
        field_leaks=[],
        exact_leakage_rate=0.0,
        weighted_leakage_score=0.0,
        task_success=True,
    ))
    
    # Result 3: Leakage with attack, task failed
    results.append(DetectionResult(
        scenario_id="scen_003",
        trace_id="trace_003",
        leaked=True,
        total_leaks=1,
        field_leaks=[
            FieldLeak(
                field_path="account",
                leaked_value="789",
                channel=Channel.C2_INTER_AGENT,
                event_sequence=0,
                detection_method="canary",
                confidence=1.0,
            ),
        ],
        exact_leakage_rate=0.1,
        weighted_leakage_score=0.8,
        attack_class=AttackClass.DPI,
        attack_success=True,
        task_success=False,
    ))
    
    return results


# ============================================================
# MetricsCalculator Tests
# ============================================================

class TestMetricsCalculator:
    """Tests for the core metrics calculator."""
    
    def test_create_calculator(self):
        """Test calculator creation."""
        calc = MetricsCalculator()
        assert calc is not None
        assert len(calc.channel_weights) == 7
    
    def test_calculate_leakage(self, sample_detection_result):
        """Test ELR and WLS calculation."""
        calc = MetricsCalculator()
        
        leakage = calc.calculate_leakage(sample_detection_result, total_private_fields=10)
        
        assert isinstance(leakage, LeakageMetrics)
        assert leakage.elr == 0.3  # 3 leaks / 10 fields
        assert leakage.leaked_field_count == 3
        assert leakage.total_private_fields == 10
        assert leakage.wls > 0
    
    def test_calculate_leakage_no_leaks(self):
        """Test leakage calculation with no leaks."""
        calc = MetricsCalculator()
        
        result = DetectionResult(
            scenario_id="test",
            trace_id="trace",
            leaked=False,
            field_leaks=[],
        )
        
        leakage = calc.calculate_leakage(result, total_private_fields=10)
        
        assert leakage.elr == 0.0
        assert leakage.wls == 0.0
        assert leakage.leaked_field_count == 0
    
    def test_wls_respects_channel_weights(self, sample_field_leaks):
        """Test that WLS uses channel weights correctly."""
        # Custom weights: C1 = 1.0, C3 = 0.5
        calc = MetricsCalculator(channel_weights={
            Channel.C1_FINAL_OUTPUT: 1.0,
            Channel.C3_TOOL_INPUT: 0.5,
        })
        
        result = DetectionResult(
            scenario_id="test",
            trace_id="trace",
            leaked=True,
            field_leaks=sample_field_leaks,
        )
        
        leakage = calc.calculate_leakage(result, total_private_fields=10)
        
        # Should weight C1 more than C3
        assert leakage.wls > 0
    
    def test_tier_breakdown(self, sample_field_leaks):
        """Test breakdown by canary tier."""
        calc = MetricsCalculator()
        
        result = DetectionResult(
            scenario_id="test",
            trace_id="trace",
            leaked=True,
            field_leaks=sample_field_leaks,
        )
        
        leakage = calc.calculate_leakage(result, total_private_fields=10)
        
        assert "obvious" in leakage.tier_breakdown
        assert "realistic" in leakage.tier_breakdown
        assert "semantic" in leakage.tier_breakdown
        assert leakage.tier_breakdown["obvious"] == 1
    
    def test_calculate_channel_metrics(self, sample_detection_result):
        """Test per-channel metrics calculation."""
        calc = MetricsCalculator()
        
        channels = calc.calculate_channel_metrics(sample_detection_result)
        
        assert len(channels) == 2  # C1 and C3
        
        c1 = next(c for c in channels if c.channel == Channel.C1_FINAL_OUTPUT)
        assert c1.leak_count == 2
        assert c1.leak_rate > 0
    
    def test_calculate_asr(self, multiple_detection_results):
        """Test Attack Success Rate calculation."""
        calc = MetricsCalculator()
        
        # Only attack scenarios
        attack_results = [r for r in multiple_detection_results if r.attack_class]
        asr = calc.calculate_asr(attack_results)
        
        assert isinstance(asr, AttackMetrics)
        assert asr.total_attacks == 1
        assert asr.successful_attacks == 1
        assert asr.asr == 1.0
    
    def test_calculate_tsr(self, multiple_detection_results):
        """Test Task Success Rate calculation."""
        calc = MetricsCalculator()
        
        tsr = calc.calculate_tsr(multiple_detection_results)
        
        assert isinstance(tsr, TaskMetrics)
        assert tsr.total_tasks == 3
        assert tsr.successful_tasks == 2
        assert abs(tsr.tsr - 2/3) < 0.01
    
    def test_aggregate_all_metrics(self, multiple_detection_results):
        """Test full aggregation."""
        calc = MetricsCalculator()
        
        benchmark = calc.aggregate(multiple_detection_results)
        
        assert isinstance(benchmark, BenchmarkMetrics)
        assert benchmark.total_scenarios == 3
        assert benchmark.leakage.leaked_field_count == 3
        assert benchmark.tasks.tsr > 0


# ============================================================
# ConfidenceInterval Tests
# ============================================================

class TestConfidenceInterval:
    """Tests for confidence interval calculation."""
    
    def test_from_single_value(self):
        """Test CI from single value."""
        ci = ConfidenceInterval.from_values([0.5])
        
        assert ci.mean == 0.5
        assert ci.std == 0.0
        assert ci.n == 1
    
    def test_from_multiple_values(self):
        """Test CI from multiple values."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        ci = ConfidenceInterval.from_values(values)
        
        assert abs(ci.mean - 0.3) < 0.01
        assert ci.std > 0
        assert ci.ci_lower < ci.mean < ci.ci_upper
        assert ci.n == 5
    
    def test_empty_values(self):
        """Test CI with empty values."""
        ci = ConfidenceInterval.from_values([])
        
        assert ci.mean == 0.0
        assert ci.n == 0
    
    def test_ci_bounds_in_range(self):
        """Test that CI bounds are in [0, 1]."""
        values = [0.9, 0.95, 0.92, 0.88, 0.91]
        ci = ConfidenceInterval.from_values(values)
        
        assert 0 <= ci.ci_lower <= 1
        assert 0 <= ci.ci_upper <= 1


# ============================================================
# MetricsAggregator Tests
# ============================================================

class TestMetricsAggregator:
    """Tests for metrics aggregation across runs."""
    
    def test_create_aggregator(self):
        """Test aggregator creation."""
        agg = MetricsAggregator()
        assert len(agg.runs) == 0
    
    def test_add_results(self, multiple_detection_results):
        """Test adding results."""
        agg = MetricsAggregator()
        
        agg.add_results(multiple_detection_results, "run_1")
        
        assert len(agg.runs) == 1
    
    def test_aggregate_single_run(self, multiple_detection_results):
        """Test aggregation with single run."""
        agg = MetricsAggregator()
        agg.add_results(multiple_detection_results, "run_1")
        
        metrics = agg.aggregate()
        
        assert isinstance(metrics, AggregatedMetrics)
        assert metrics.total_runs == 1
        assert metrics.total_scenarios == 3
    
    def test_aggregate_multiple_runs(self, multiple_detection_results):
        """Test aggregation with multiple runs."""
        agg = MetricsAggregator()
        
        agg.add_results(multiple_detection_results, "run_1")
        agg.add_results(multiple_detection_results, "run_2")
        agg.add_results(multiple_detection_results, "run_3")
        
        metrics = agg.aggregate()
        
        assert metrics.total_runs == 3
        assert metrics.elr.n == 3
        assert metrics.elr.std >= 0  # Same data, std might be 0
    
    def test_clear(self, multiple_detection_results):
        """Test clearing accumulated results."""
        agg = MetricsAggregator()
        agg.add_results(multiple_detection_results, "run_1")
        
        agg.clear()
        
        assert len(agg.runs) == 0
    
    def test_summary_output(self, multiple_detection_results):
        """Test summary generation."""
        agg = MetricsAggregator()
        agg.add_results(multiple_detection_results, "run_1")
        
        summary = agg.summary()
        
        assert "agentleak Benchmark" in summary
        assert "ELR" in summary
        assert "TSR" in summary
    
    def test_to_dict(self, multiple_detection_results):
        """Test dictionary export."""
        agg = MetricsAggregator()
        agg.add_results(multiple_detection_results, "run_1")
        
        data = agg.to_dict()
        
        assert "elr" in data
        assert "tsr" in data
        assert "mean" in data["elr"]


# ============================================================
# ParetoPoint Tests
# ============================================================

class TestParetoPoint:
    """Tests for Pareto points."""
    
    def test_create_point(self):
        """Test point creation."""
        p = ParetoPoint(name="test", privacy_cost=0.1, utility=0.9)
        
        assert p.name == "test"
        assert p.privacy_cost == 0.1
        assert p.utility == 0.9
    
    def test_dominates(self):
        """Test Pareto dominance."""
        # p1 dominates p2 (better privacy AND utility)
        p1 = ParetoPoint(name="p1", privacy_cost=0.1, utility=0.9)
        p2 = ParetoPoint(name="p2", privacy_cost=0.2, utility=0.8)
        
        assert p1.dominates(p2)
        assert not p2.dominates(p1)
    
    def test_no_dominance(self):
        """Test non-dominated points."""
        # p1: better privacy, worse utility
        # p2: worse privacy, better utility
        p1 = ParetoPoint(name="p1", privacy_cost=0.1, utility=0.7)
        p2 = ParetoPoint(name="p2", privacy_cost=0.3, utility=0.9)
        
        assert not p1.dominates(p2)
        assert not p2.dominates(p1)
    
    def test_distance_to(self):
        """Test Euclidean distance."""
        p1 = ParetoPoint(name="p1", privacy_cost=0.0, utility=0.0)
        p2 = ParetoPoint(name="p2", privacy_cost=0.3, utility=0.4)
        
        distance = p1.distance_to(p2)
        assert abs(distance - 0.5) < 0.01  # 3-4-5 triangle


# ============================================================
# ParetoCalculator Tests
# ============================================================

class TestParetoCalculator:
    """Tests for Pareto frontier calculation."""
    
    def test_create_calculator(self):
        """Test calculator creation."""
        calc = ParetoCalculator()
        assert len(calc.points) == 0
    
    def test_add_point(self):
        """Test adding points."""
        calc = ParetoCalculator()
        
        p = calc.add_point("agent_1", privacy_cost=0.1, utility=0.9)
        
        assert len(calc.points) == 1
        assert p.name == "agent_1"
    
    def test_compute_frontier_simple(self):
        """Test frontier computation with clear dominance."""
        calc = ParetoCalculator()
        
        # p1 dominates p2
        calc.add_point("p1", privacy_cost=0.1, utility=0.9)
        calc.add_point("p2", privacy_cost=0.2, utility=0.8)
        
        frontier = calc.compute_frontier()
        
        assert len(frontier.points) == 1
        assert frontier.points[0].name == "p1"
    
    def test_compute_frontier_multiple_optimal(self):
        """Test frontier with multiple non-dominated points."""
        calc = ParetoCalculator()
        
        # All on frontier (tradeoff)
        calc.add_point("p1", privacy_cost=0.1, utility=0.7)
        calc.add_point("p2", privacy_cost=0.2, utility=0.8)
        calc.add_point("p3", privacy_cost=0.3, utility=0.9)
        
        frontier = calc.compute_frontier()
        
        assert len(frontier.points) == 3
    
    def test_compute_frontier_mixed(self):
        """Test frontier with mix of dominated and non-dominated."""
        calc = ParetoCalculator()
        
        calc.add_point("optimal_1", privacy_cost=0.1, utility=0.9)
        calc.add_point("optimal_2", privacy_cost=0.3, utility=0.95)
        calc.add_point("dominated", privacy_cost=0.2, utility=0.8)  # Dominated by optimal_1
        
        frontier = calc.compute_frontier()
        
        assert len(frontier.points) == 2
        names = [p.name for p in frontier.points]
        assert "optimal_1" in names
        assert "optimal_2" in names
        assert "dominated" not in names
    
    def test_compute_auc(self):
        """Test AUC calculation."""
        calc = ParetoCalculator()
        
        calc.add_point("p1", privacy_cost=0.0, utility=0.8)
        calc.add_point("p2", privacy_cost=0.5, utility=0.9)
        calc.add_point("p3", privacy_cost=1.0, utility=0.95)
        
        auc = calc.compute_auc()
        
        assert 0 < auc < 1
    
    def test_compute_auc_perfect(self):
        """Test AUC for perfect agent (0 leakage, 1 utility)."""
        calc = ParetoCalculator()
        
        calc.add_point("perfect", privacy_cost=0.0, utility=1.0)
        
        auc = calc.compute_auc()
        
        assert auc == 1.0
    
    def test_compute_hypervolume(self):
        """Test hypervolume calculation."""
        calc = ParetoCalculator()
        
        calc.add_point("p1", privacy_cost=0.2, utility=0.8)
        calc.add_point("p2", privacy_cost=0.4, utility=0.9)
        
        hv = calc.compute_hypervolume()
        
        assert hv > 0
    
    def test_rank_points(self):
        """Test point ranking."""
        calc = ParetoCalculator()
        
        calc.add_point("optimal", privacy_cost=0.1, utility=0.9)
        calc.add_point("rank2", privacy_cost=0.2, utility=0.8)
        calc.add_point("rank3", privacy_cost=0.3, utility=0.7)
        
        rankings = calc.rank_points()
        
        assert len(rankings) == 3
        
        # First point should be rank 1 (frontier)
        rank1_points = [name for rank, p in rankings if rank == 1 for name in [p.name]]
        assert "optimal" in rank1_points
    
    def test_get_optimal_point(self):
        """Test getting optimal point with constraints."""
        calc = ParetoCalculator()
        
        calc.add_point("low_privacy", privacy_cost=0.05, utility=0.7)
        calc.add_point("balanced", privacy_cost=0.15, utility=0.85)
        calc.add_point("high_utility", privacy_cost=0.4, utility=0.95)
        
        frontier = calc.compute_frontier()
        
        # Constraint: max privacy cost 0.2
        best = frontier.get_optimal_point(max_privacy_cost=0.2)
        assert best is not None
        assert best.name in ["low_privacy", "balanced"]
    
    def test_summary_output(self):
        """Test summary generation."""
        calc = ParetoCalculator()
        
        calc.add_point("agent_a", privacy_cost=0.1, utility=0.9)
        calc.add_point("agent_b", privacy_cost=0.2, utility=0.85)
        
        summary = calc.summary()
        
        assert "Pareto Analysis" in summary
        assert "Frontier Points" in summary
        assert "agent_a" in summary
    
    def test_clear(self):
        """Test clearing points."""
        calc = ParetoCalculator()
        calc.add_point("test", privacy_cost=0.1, utility=0.9)
        
        calc.clear()
        
        assert len(calc.points) == 0


# ============================================================
# Integration Tests
# ============================================================

class TestMetricsIntegration:
    """Integration tests for the complete metrics pipeline."""
    
    def test_full_benchmark_flow(self, multiple_detection_results):
        """Test complete flow from detection to Pareto."""
        # Step 1: Calculate metrics
        calc = MetricsCalculator()
        benchmark = calc.aggregate(multiple_detection_results)
        
        # Step 2: Aggregate across "runs"
        agg = MetricsAggregator()
        agg.add_results(multiple_detection_results, "baseline")
        aggregated = agg.aggregate()
        
        # Step 3: Pareto analysis
        pareto = ParetoCalculator()
        pareto.add_point(
            "baseline",
            privacy_cost=aggregated.elr.mean,
            utility=aggregated.tsr.mean,
        )
        pareto.add_point(
            "defense_lcf",
            privacy_cost=aggregated.elr.mean * 0.3,  # 70% reduction
            utility=aggregated.tsr.mean * 0.95,  # 5% utility drop
        )
        
        frontier = pareto.compute_frontier()
        auc = pareto.compute_auc()
        
        assert len(frontier.points) >= 1
        assert 0 <= auc <= 1
    
    def test_compare_agents(self):
        """Test comparing multiple agents."""
        pareto = ParetoCalculator()
        
        # Simulate results from different agents
        agents = [
            ("gpt-4", 0.15, 0.92),
            ("gpt-4+lcf", 0.05, 0.88),
            ("claude-3", 0.12, 0.89),
            ("claude-3+lcf", 0.04, 0.85),
            ("llama-70b", 0.25, 0.82),
        ]
        
        for name, privacy, utility in agents:
            pareto.add_point(name, privacy_cost=privacy, utility=utility)
        
        frontier = pareto.compute_frontier()
        
        # At least 2 points should be on frontier
        assert len(frontier.points) >= 2
        
        # LCF variants should generally be on frontier (better privacy)
        frontier_names = [p.name for p in frontier.points]
        assert any("lcf" in name for name in frontier_names)
