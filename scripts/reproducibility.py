"""
Reproducibility Infrastructure for AgentLeak.

This module provides comprehensive reproducibility support:
1. Fixed seeds for all randomness
2. Full JSONL logging of all API calls
3. Cost tracking and reporting
4. Experiment versioning and checkpointing

Addresses reviewer concern: "Improve reproducibility with seeds, logs, costs"
"""

from __future__ import annotations
import json
import os
import hashlib
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import uuid


# =============================================================================
# Seed Management
# =============================================================================

@dataclass
class SeedConfig:
    """Configuration for reproducible random seeds."""
    
    # Master seed from which all others are derived
    master_seed: int = 42
    
    # Component-specific seeds (derived from master)
    scenario_seed: int = field(init=False)
    sampling_seed: int = field(init=False)
    evaluation_seed: int = field(init=False)
    shuffle_seed: int = field(init=False)
    
    def __post_init__(self):
        # Derive component seeds deterministically from master
        rng = random.Random(self.master_seed)
        self.scenario_seed = rng.randint(0, 2**32 - 1)
        self.sampling_seed = rng.randint(0, 2**32 - 1)
        self.evaluation_seed = rng.randint(0, 2**32 - 1)
        self.shuffle_seed = rng.randint(0, 2**32 - 1)
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "master_seed": self.master_seed,
            "scenario_seed": self.scenario_seed,
            "sampling_seed": self.sampling_seed,
            "evaluation_seed": self.evaluation_seed,
            "shuffle_seed": self.shuffle_seed,
        }


class SeedManager:
    """Manage reproducible random number generation."""
    
    def __init__(self, config: SeedConfig = None):
        self.config = config or SeedConfig()
        self._rngs: Dict[str, random.Random] = {}
    
    def get_rng(self, component: str) -> random.Random:
        """Get a random number generator for a specific component."""
        if component not in self._rngs:
            if component == "scenario":
                self._rngs[component] = random.Random(self.config.scenario_seed)
            elif component == "sampling":
                self._rngs[component] = random.Random(self.config.sampling_seed)
            elif component == "evaluation":
                self._rngs[component] = random.Random(self.config.evaluation_seed)
            elif component == "shuffle":
                self._rngs[component] = random.Random(self.config.shuffle_seed)
            else:
                # Derive from master for unknown components
                self._rngs[component] = random.Random(
                    hash(component) ^ self.config.master_seed
                )
        return self._rngs[component]
    
    def reset(self):
        """Reset all RNGs to their initial state."""
        self._rngs.clear()


# =============================================================================
# API Call Logging
# =============================================================================

@dataclass
class APICallLog:
    """Log entry for a single API call."""
    
    # Unique identifiers
    call_id: str
    experiment_id: str
    
    # Request details
    timestamp: str
    provider: str  # openai, anthropic, google, etc.
    model: str
    endpoint: str  # chat, completions, embeddings
    
    # Request content
    messages: List[Dict[str, str]]
    temperature: float
    max_tokens: int
    
    # Response details
    response_id: Optional[str] = None
    response_content: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # Usage and cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    
    # Performance
    latency_ms: float = 0.0
    
    # Error handling
    error: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class APILogger:
    """Logger for all API calls with JSONL output."""
    
    # Cost per 1M tokens (approximate, as of late 2024)
    COST_PER_1M_TOKENS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "text-embedding-3-small": {"input": 0.02, "output": 0.00},
        "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    }
    
    def __init__(
        self,
        experiment_id: str = None,
        output_dir: str = "logs",
        buffer_size: int = 10,
    ):
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self._buffer: List[APICallLog] = []
        self._total_cost = 0.0
        self._total_calls = 0
        
        # Output file
        self.log_file = self.output_dir / f"api_calls_{self.experiment_id}.jsonl"
    
    def log_call(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        response: Dict[str, Any] = None,
        latency_ms: float = 0.0,
        error: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> APICallLog:
        """Log an API call."""
        
        # Extract usage from response
        usage = response.get("usage", {}) if response else {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Calculate cost
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Extract response content
        response_content = None
        finish_reason = None
        response_id = None
        
        if response:
            response_id = response.get("id")
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                response_content = message.get("content")
                finish_reason = choices[0].get("finish_reason")
        
        log = APICallLog(
            call_id=str(uuid.uuid4()),
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            endpoint="chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_id=response_id,
            response_content=response_content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            error=error,
            success=error is None,
        )
        
        self._buffer.append(log)
        self._total_cost += cost
        self._total_calls += 1
        
        if len(self._buffer) >= self.buffer_size:
            self.flush()
        
        return log
    
    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost in USD for an API call."""
        rates = self.COST_PER_1M_TOKENS.get(model, {"input": 1.0, "output": 1.0})
        
        input_cost = (prompt_tokens / 1_000_000) * rates["input"]
        output_cost = (completion_tokens / 1_000_000) * rates["output"]
        
        return input_cost + output_cost
    
    def flush(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        with open(self.log_file, 'a') as f:
            for log in self._buffer:
                f.write(json.dumps(log.to_dict()) + '\n')
        
        self._buffer.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all logged calls."""
        self.flush()
        
        # Read all logs
        logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
        
        # Aggregate statistics
        total_tokens = sum(l.get("total_tokens", 0) for l in logs)
        total_cost = sum(l.get("cost_usd", 0) for l in logs)
        total_calls = len(logs)
        success_rate = sum(1 for l in logs if l.get("success", False)) / total_calls if total_calls > 0 else 0
        
        # Per-model breakdown
        model_stats = {}
        for log in logs:
            model = log.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = {"calls": 0, "tokens": 0, "cost": 0.0}
            model_stats[model]["calls"] += 1
            model_stats[model]["tokens"] += log.get("total_tokens", 0)
            model_stats[model]["cost"] += log.get("cost_usd", 0)
        
        # Latency stats
        latencies = [l.get("latency_ms", 0) for l in logs if l.get("latency_ms", 0) > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "experiment_id": self.experiment_id,
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "model_breakdown": model_stats,
            "log_file": str(self.log_file),
        }
    
    def __del__(self):
        """Ensure buffer is flushed on cleanup."""
        self.flush()


# =============================================================================
# Experiment Configuration and Versioning
# =============================================================================

@dataclass
class ExperimentConfig:
    """Full experiment configuration for reproducibility."""
    
    # Identification
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    experiment_name: str = "agentleak_evaluation"
    version: str = "1.0.0"
    
    # Seeds
    seeds: SeedConfig = field(default_factory=SeedConfig)
    
    # Evaluation parameters
    n_scenarios_per_model: int = 100
    models: List[str] = field(default_factory=lambda: ["gpt-4o-mini"])
    domains: List[str] = field(default_factory=lambda: ["healthcare", "finance", "legal", "corporate"])
    
    # Detection parameters
    semantic_threshold: float = 0.72
    use_hybrid_detection: bool = True
    
    # Defense parameters
    defenses: List[str] = field(default_factory=lambda: ["none", "output_filter", "zero_trust"])
    
    # Resource limits
    max_api_calls: int = 1000
    timeout_seconds: int = 60
    max_cost_usd: float = 100.0
    
    # Output
    output_dir: str = "outputs"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "version": self.version,
            "seeds": self.seeds.to_dict(),
            "n_scenarios_per_model": self.n_scenarios_per_model,
            "models": self.models,
            "domains": self.domains,
            "semantic_threshold": self.semantic_threshold,
            "use_hybrid_detection": self.use_hybrid_detection,
            "defenses": self.defenses,
            "max_api_calls": self.max_api_calls,
            "timeout_seconds": self.timeout_seconds,
            "max_cost_usd": self.max_cost_usd,
            "output_dir": self.output_dir,
        }
    
    def get_hash(self) -> str:
        """Get a hash of the configuration for change detection."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def save(self, path: str = None):
        """Save configuration to JSON."""
        if path is None:
            path = Path(self.output_dir) / f"config_{self.experiment_id}.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        seeds = SeedConfig(master_seed=data["seeds"]["master_seed"])
        
        return cls(
            experiment_id=data["experiment_id"],
            experiment_name=data["experiment_name"],
            version=data["version"],
            seeds=seeds,
            n_scenarios_per_model=data["n_scenarios_per_model"],
            models=data["models"],
            domains=data["domains"],
            semantic_threshold=data["semantic_threshold"],
            use_hybrid_detection=data["use_hybrid_detection"],
            defenses=data["defenses"],
            max_api_calls=data["max_api_calls"],
            timeout_seconds=data["timeout_seconds"],
            max_cost_usd=data["max_cost_usd"],
            output_dir=data["output_dir"],
        )


# =============================================================================
# Checkpointing
# =============================================================================

@dataclass
class Checkpoint:
    """Checkpoint for experiment state."""
    experiment_id: str
    timestamp: str
    
    # Progress
    completed_scenarios: int
    total_scenarios: int
    
    # State
    last_scenario_id: str
    current_model: str
    current_domain: str
    
    # Partial results
    results_so_far: List[Dict[str, Any]]
    
    # Costs
    total_cost_so_far: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """Manage experiment checkpoints for resumability."""
    
    def __init__(self, experiment_id: str, output_dir: str = "checkpoints"):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.output_dir / f"checkpoint_{experiment_id}.json"
    
    def save(self, checkpoint: Checkpoint):
        """Save a checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    def load(self) -> Optional[Checkpoint]:
        """Load the latest checkpoint."""
        if not self.checkpoint_file.exists():
            return None
        
        with open(self.checkpoint_file, 'r') as f:
            data = json.load(f)
        
        return Checkpoint(**data)
    
    def clear(self):
        """Remove checkpoint after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# =============================================================================
# Cost Estimator
# =============================================================================

class CostEstimator:
    """Estimate and track experiment costs."""
    
    # Average tokens per scenario (empirical)
    TOKENS_PER_SCENARIO = {
        "system_prompt": 200,
        "user_query": 50,
        "response": 300,
        "detection": 100,
    }
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def estimate_cost(self) -> Dict[str, float]:
        """Estimate total cost before running experiment."""
        estimates = {}
        
        total_scenarios = self.config.n_scenarios_per_model * len(self.config.models)
        tokens_per_scenario = sum(self.TOKENS_PER_SCENARIO.values())
        
        for model in self.config.models:
            rates = APILogger.COST_PER_1M_TOKENS.get(model, {"input": 1.0, "output": 1.0})
            
            # Estimate input/output split (70/30)
            input_tokens = total_scenarios * tokens_per_scenario * 0.7
            output_tokens = total_scenarios * tokens_per_scenario * 0.3
            
            input_cost = (input_tokens / 1_000_000) * rates["input"]
            output_cost = (output_tokens / 1_000_000) * rates["output"]
            
            estimates[model] = input_cost + output_cost
        
        estimates["total"] = sum(estimates.values())
        
        return estimates
    
    def format_estimate(self) -> str:
        """Format cost estimate as readable string."""
        estimates = self.estimate_cost()
        
        lines = ["Cost Estimate", "=" * 40]
        
        for model, cost in estimates.items():
            if model != "total":
                lines.append(f"  {model}: ${cost:.2f}")
        
        lines.append("-" * 40)
        lines.append(f"  TOTAL: ${estimates['total']:.2f}")
        
        return "\n".join(lines)


# =============================================================================
# Full Reproducibility Report
# =============================================================================

def generate_reproducibility_report(
    config: ExperimentConfig,
    api_logger: APILogger,
    results: Dict[str, Any],
) -> str:
    """Generate a full reproducibility report."""
    
    summary = api_logger.get_summary()
    
    lines = [
        "# Reproducibility Report",
        "",
        f"**Experiment ID**: {config.experiment_id}",
        f"**Version**: {config.version}",
        f"**Config Hash**: {config.get_hash()}",
        "",
        "## Random Seeds",
        "",
        "```json",
        json.dumps(config.seeds.to_dict(), indent=2),
        "```",
        "",
        "## API Calls",
        "",
        f"- **Total Calls**: {summary['total_calls']}",
        f"- **Total Tokens**: {summary['total_tokens']:,}",
        f"- **Total Cost**: ${summary['total_cost_usd']:.4f}",
        f"- **Success Rate**: {summary['success_rate']:.1%}",
        f"- **Avg Latency**: {summary['avg_latency_ms']:.0f}ms",
        "",
        "### Per-Model Breakdown",
        "",
        "| Model | Calls | Tokens | Cost |",
        "|-------|-------|--------|------|",
    ]
    
    for model, stats in summary.get("model_breakdown", {}).items():
        lines.append(
            f"| {model} | {stats['calls']} | {stats['tokens']:,} | ${stats['cost']:.4f} |"
        )
    
    lines.extend([
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config.to_dict(), indent=2),
        "```",
        "",
        "## Reproduction Instructions",
        "",
        "To reproduce these results:",
        "",
        "1. Install dependencies: `pip install -r requirements.txt`",
        "2. Set API keys in environment variables",
        f"3. Load config: `ExperimentConfig.load('config_{config.experiment_id}.json')`",
        "4. Run evaluation with the same config",
        "",
        "The master seed ensures deterministic scenario generation.",
        "API responses may vary due to model updates, but the evaluation framework remains constant.",
        "",
        "## Log Files",
        "",
        f"- **API Calls**: `{summary['log_file']}`",
        f"- **Config**: `benchmark_results/raw_runs/config_{config.experiment_id}.json`",
    ])
    
    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Demo: Full reproducibility infrastructure
    print("=" * 60)
    print("Reproducibility Infrastructure Demo")
    print("=" * 60)
    
    # 1. Create experiment configuration
    config = ExperimentConfig(
        experiment_name="agentleak_main_evaluation",
        seeds=SeedConfig(master_seed=42),
        n_scenarios_per_model=100,
        models=["gpt-4o-mini", "claude-3-haiku-20240307"],
        domains=["healthcare", "finance"],
    )
    
    print(f"\nExperiment ID: {config.experiment_id}")
    print(f"Config Hash: {config.get_hash()}")
    
    # Save config
    config_path = config.save()
    print(f"Config saved to: {config_path}")
    
    # 2. Initialize seed manager
    seed_manager = SeedManager(config.seeds)
    scenario_rng = seed_manager.get_rng("scenario")
    
    print(f"\nFirst 5 scenario random values:")
    for i in range(5):
        print(f"  {scenario_rng.random():.6f}")
    
    # Reset and verify reproducibility
    seed_manager.reset()
    scenario_rng = seed_manager.get_rng("scenario")
    
    print(f"\nAfter reset (should be same):")
    for i in range(5):
        print(f"  {scenario_rng.random():.6f}")
    
    # 3. Initialize API logger
    api_logger = APILogger(
        experiment_id=config.experiment_id,
        output_dir="logs",
    )
    
    # Log some mock API calls
    for i in range(5):
        api_logger.log_call(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Test query {i}"}],
            response={
                "id": f"chatcmpl-{i}",
                "choices": [{"message": {"content": f"Response {i}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            },
            latency_ms=random.uniform(200, 500),
        )
    
    summary = api_logger.get_summary()
    print(f"\nAPI Call Summary:")
    print(f"  Total calls: {summary['total_calls']}")
    print(f"  Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Log file: {summary['log_file']}")
    
    # 4. Cost estimation
    estimator = CostEstimator(config)
    print(f"\n{estimator.format_estimate()}")
    
    # 5. Generate reproducibility report
    report = generate_reproducibility_report(config, api_logger, {})
    report_path = Path("outputs") / f"reproducibility_{config.experiment_id}.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReproducibility report saved to: {report_path}")
