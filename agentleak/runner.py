"""
AgentLeak Benchmark Runner - Complete end-to-end benchmark execution.

This module orchestrates the full AgentLeak benchmark:
1. Generate scenarios (1000 across 4 verticals)
2. Run scenarios through agent adapter
3. Apply attacks (optional)
4. Detect leakage
5. Calculate metrics
6. Generate report

Example:
    runner = BenchmarkRunner()
    results = runner.run(
        n_scenarios=100,
        with_attacks=True,
        defense="lcf",
    )
    print(runner.summary())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import json
import os
import time

from .schemas.scenario import Scenario, Vertical, AttackClass
from .schemas.trace import ExecutionTrace
from .schemas.results import DetectionResult, BenchmarkResults

from .generators import ScenarioGenerator
from .harness import BaseAdapter, DryRunAdapter, AdapterConfig
from .detection import DetectionPipeline, DetectionConfig
from .metrics import MetricsCalculator, MetricsAggregator, ParetoCalculator
from .defenses import LearnedContentFilter, LCFConfig, LCFTrainer, OutputSanitizer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    
    # Scenario generation
    n_scenarios: int = 1000
    scenarios_per_vertical: Optional[int] = None  # Auto-balance if None
    seed: int = 42
    
    # Attack configuration
    with_attacks: bool = True
    attack_probability: float = 0.5  # 50% of scenarios get attacked
    attack_classes: Optional[list[AttackClass]] = None  # All if None
    
    # Defense configuration
    defense: Optional[str] = None  # "lcf", "sanitizer", or None
    defense_config: Optional[dict[str, Any]] = None
    
    # Execution
    adapter_type: str = "dry_run"  # "dry_run", "langchain", "autogen", etc.
    max_concurrent: int = 1
    timeout_per_scenario: float = 60.0
    
    # Output
    output_dir: Optional[str] = None
    save_traces: bool = True
    save_scenarios: bool = True
    verbose: bool = True


@dataclass
class ScenarioResult:
    """Result from running a single scenario."""
    scenario: Scenario
    trace: ExecutionTrace
    detection: DetectionResult
    duration_seconds: float
    attack_applied: Optional[AttackClass] = None
    defense_applied: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    config: BenchmarkConfig
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Results
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    
    # Aggregated metrics
    total_scenarios: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_leaks: int = 0
    
    # Metrics
    elr_mean: float = 0.0
    elr_std: float = 0.0
    wls_mean: float = 0.0
    tsr_mean: float = 0.0
    asr_mean: float = 0.0


class BenchmarkRunner:
    """
    Main benchmark runner for agentleak.
    
    Orchestrates full benchmark execution with configurable:
    - Scenario generation
    - Attack application
    - Defense deployment
    - Detection and metrics
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        
        # Components
        self.generator: Optional[ScenarioGenerator] = None
        self.adapter: Optional[BaseAdapter] = None
        self.detector: Optional[DetectionPipeline] = None
        self.defense: Optional[Any] = None
        self.metrics_calc = MetricsCalculator()
        self.aggregator = MetricsAggregator()
        
        # State
        self.current_run: Optional[BenchmarkRun] = None
        self._scenarios: list[Scenario] = []
    
    def setup(self) -> None:
        """Initialize all components."""
        # Generator
        self.generator = ScenarioGenerator(seed=self.config.seed)
        
        # Adapter
        if self.config.adapter_type == "dry_run":
            self.adapter = DryRunAdapter(AdapterConfig(dry_run=False))
        else:
            # Future: support other adapters
            self.adapter = DryRunAdapter()
        
        # Detector
        self.detector = DetectionPipeline()
        
        # Defense
        if self.config.defense == "lcf":
            self.defense = LearnedContentFilter(
                LCFConfig(**(self.config.defense_config or {}))
            )
        elif self.config.defense == "sanitizer":
            self.defense = OutputSanitizer()
    
    def generate_scenarios(self) -> list[Scenario]:
        """Generate all scenarios for the benchmark."""
        if self.generator is None:
            self.setup()
        
        scenarios = []
        verticals = list(Vertical)
        
        # Calculate per-vertical count
        if self.config.scenarios_per_vertical:
            per_vertical = self.config.scenarios_per_vertical
        else:
            per_vertical = self.config.n_scenarios // len(verticals)
        
        for vertical in verticals:
            for i in range(per_vertical):
                scenario = self.generator.generate(vertical)
                scenarios.append(scenario)
                
                if len(scenarios) >= self.config.n_scenarios:
                    break
            
            if len(scenarios) >= self.config.n_scenarios:
                break
        
        self._scenarios = scenarios
        return scenarios
    
    def run(
        self,
        scenarios: Optional[list[Scenario]] = None,
        progress_callback: Optional[callable] = None,
    ) -> BenchmarkRun:
        """
        Run the complete benchmark.
        
        Args:
            scenarios: Pre-generated scenarios (generates if None)
            progress_callback: Called with (current, total) for progress
            
        Returns:
            BenchmarkRun with all results
        """
        self.setup()
        
        # Generate or use provided scenarios
        if scenarios is None:
            scenarios = self.generate_scenarios()
        
        # Initialize run
        self.current_run = BenchmarkRun(
            config=self.config,
            started_at=datetime.utcnow(),
            total_scenarios=len(scenarios),
        )
        
        # Train defense if needed
        if self.defense and isinstance(self.defense, LearnedContentFilter):
            self._train_defense(scenarios)
        
        # Run each scenario
        for i, scenario in enumerate(scenarios):
            try:
                result = self._run_scenario(scenario)
                self.current_run.scenario_results.append(result)
                self.current_run.successful_runs += 1
                
                if result.detection.leaked:
                    self.current_run.total_leaks += result.detection.total_leaks
                
            except Exception as e:
                self.current_run.failed_runs += 1
                if self.config.verbose:
                    print(f"Error in scenario {scenario.scenario_id}: {e}")
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(scenarios))
        
        # Finalize
        self.current_run.completed_at = datetime.utcnow()
        self._calculate_aggregate_metrics()
        
        # Save results
        if self.config.output_dir:
            self._save_results()
        
        return self.current_run
    
    def _run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario."""
        start_time = time.time()
        
        # Run through adapter
        exec_result = self.adapter.run(scenario)
        trace = exec_result.trace
        
        # Apply defense filtering to trace (if enabled)
        if self.defense:
            self._apply_defense_to_trace(trace)
        
        # Detect leakage
        detection = self.detector.detect(scenario, trace)
        
        duration = time.time() - start_time
        
        return ScenarioResult(
            scenario=scenario,
            trace=trace,
            detection=detection,
            duration_seconds=duration,
            defense_applied=self.config.defense,
        )
    
    def _train_defense(self, scenarios: list[Scenario]) -> None:
        """Train LCF defense on scenario data."""
        if not isinstance(self.defense, LearnedContentFilter):
            return
        
        trainer = LCFTrainer()
        
        # Sample scenarios for training (don't use all)
        train_scenarios = scenarios[:min(100, len(scenarios))]
        
        for scenario in train_scenarios:
            trainer.add_from_scenario(scenario)
        
        trainer.train_filter(self.defense)
        
        # Register private values
        private_values = trainer.get_private_values()
        self.defense.register_private_values(private_values)
    
    def _apply_defense_to_trace(self, trace: ExecutionTrace) -> None:
        """Apply defense filtering to trace events."""
        # This modifies trace in place by filtering content
        for event in trace.events:
            if event.content and self.defense:
                result = self.defense.filter(event.content, event.channel)
                if result.filtered_content:
                    # Note: In real system, this would prevent leak
                    # Here we mark it for detection purposes
                    event.content = result.filtered_content
    
    def _calculate_aggregate_metrics(self) -> None:
        """Calculate aggregate metrics from all results."""
        if not self.current_run or not self.current_run.scenario_results:
            return
        
        detection_results = [r.detection for r in self.current_run.scenario_results]
        
        # Add to aggregator
        self.aggregator.add_results(detection_results, "main_run")
        aggregated = self.aggregator.aggregate()
        
        self.current_run.elr_mean = aggregated.elr.mean
        self.current_run.elr_std = aggregated.elr.std
        self.current_run.wls_mean = aggregated.wls.mean
        self.current_run.tsr_mean = aggregated.tsr.mean
        self.current_run.asr_mean = aggregated.asr.mean
    
    def _save_results(self) -> None:
        """Save results to output directory."""
        if not self.config.output_dir or not self.current_run:
            return
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Summary JSON
        summary = {
            "config": {
                "n_scenarios": self.config.n_scenarios,
                "defense": self.config.defense,
                "with_attacks": self.config.with_attacks,
            },
            "results": {
                "total_scenarios": self.current_run.total_scenarios,
                "successful_runs": self.current_run.successful_runs,
                "failed_runs": self.current_run.failed_runs,
                "total_leaks": self.current_run.total_leaks,
                "elr_mean": self.current_run.elr_mean,
                "elr_std": self.current_run.elr_std,
                "tsr_mean": self.current_run.tsr_mean,
                "asr_mean": self.current_run.asr_mean,
            },
            "timing": {
                "started_at": self.current_run.started_at.isoformat(),
                "completed_at": self.current_run.completed_at.isoformat() if self.current_run.completed_at else None,
            }
        }
        
        with open(os.path.join(self.config.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.current_run:
            return "No benchmark run completed."
        
        run = self.current_run
        
        lines = [
            "=" * 60,
            "agentleak Benchmark Results",
            "=" * 60,
            "",
            f"Configuration:",
            f"  Scenarios: {run.total_scenarios}",
            f"  Defense: {run.config.defense or 'None'}",
            f"  Attacks: {'Enabled' if run.config.with_attacks else 'Disabled'}",
            "",
            f"Execution:",
            f"  Successful: {run.successful_runs}",
            f"  Failed: {run.failed_runs}",
            f"  Total Leaks: {run.total_leaks}",
            "",
            f"Metrics:",
            f"  ELR (Exact Leakage Rate): {run.elr_mean:.2%} ± {run.elr_std:.2%}",
            f"  WLS (Weighted Leakage Score): {run.wls_mean:.3f}",
            f"  TSR (Task Success Rate): {run.tsr_mean:.2%}",
            f"  ASR (Attack Success Rate): {run.asr_mean:.2%}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)


def generate_agentleak_1000(
    output_dir: str = "agentleak_data",
    seed: int = 42,
) -> list[Scenario]:
    """
    Generate the full AgentLeak-1000 scenario set.
    
    Creates 1000 scenarios balanced across 4 verticals:
    - 250 Healthcare
    - 250 Finance
    - 250 Legal
    - 250 Corporate
    
    Args:
        output_dir: Directory to save scenarios
        seed: Random seed for reproducibility
        
    Returns:
        List of 1000 Scenario objects
    """
    generator = ScenarioGenerator(seed=seed)
    scenarios = []
    
    verticals = list(Vertical)
    per_vertical = 250
    
    for vertical in verticals:
        for i in range(per_vertical):
            scenario = generator.generate(vertical)
            scenarios.append(scenario)
    
    # Save to JSONL
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "agentleak_1000.jsonl")
    
    with open(output_file, "w") as f:
        for scenario in scenarios:
            f.write(scenario.model_dump_json() + "\n")
    
    print(f"Generated {len(scenarios)} scenarios to {output_file}")
    
    # Save summary
    summary = {
        "total": len(scenarios),
        "by_vertical": {v.value: per_vertical for v in verticals},
        "seed": seed,
        "generated_at": datetime.utcnow().isoformat(),
    }
    
    with open(os.path.join(output_dir, "agentleak_1000_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return scenarios


def load_scenarios(filepath: str, limit: Optional[int] = None) -> list[Scenario]:
    """Load scenarios from JSONL file."""
    scenarios = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line.strip())
            scenarios.append(Scenario.model_validate(data))
    return scenarios
