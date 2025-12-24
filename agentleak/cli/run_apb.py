#!/usr/bin/env python3
"""
APB Command-Line Interface - End-to-end benchmark execution.

Inspired by AgentDAM's run_agentdam.py with enhanced features:
- Multi-model support (GPT, Claude, Llama)
- Privacy-aware prompting with CoT
- WebArena-compatible evaluation
- Comprehensive result logging

Usage:
    python -m apb.cli.run_apb \\
        --test-config-dir ./data/healthcare/ \\
        --model gpt-4o \\
        --privacy-test \\
        --mitigation cot_prompt
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Setup logging
LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE = f"{LOG_FOLDER}/apb_{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 10000)}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("apb")


# =============================================================================
# Configuration
# =============================================================================

def create_arg_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="APB - Agent Privacy Benchmark: End-to-end privacy evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GPT-4o on healthcare tasks
  python -m apb.cli.run_apb --model gpt-4o --vertical healthcare --privacy-test

  # Run with privacy mitigation
  python -m apb.cli.run_apb --model gpt-4o --mitigation cot_prompt --privacy-test

  # Run on specific task configs
  python -m apb.cli.run_apb --test-config-dir ./data/finance_privacy/ --model gpt-4o-mini
        """
    )
    
    # Basic settings
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        choices=[
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
            "claude-3-5-sonnet-20241022", "claude-3-opus",
            "llama-3.2-90b", "llama-3.3-70b",
            "qwen-2.5-72b", "mock"
        ],
        help="Model to use for the agent"
    )
    
    parser.add_argument(
        "--vertical",
        type=str,
        default=None,
        choices=["healthcare", "finance", "legal", "corporate", "all"],
        help="Vertical domain to test (default: all)"
    )
    
    # Data configuration
    parser.add_argument(
        "--test-config-dir",
        type=str,
        default="./data/apb_tasks/",
        help="Directory containing task configuration JSON files"
    )
    
    parser.add_argument(
        "--test-start-idx",
        type=int,
        default=0,
        help="Start index for task configurations"
    )
    
    parser.add_argument(
        "--test-end-idx",
        type=int,
        default=1000,
        help="End index for task configurations"
    )
    
    # Privacy configuration
    parser.add_argument(
        "--privacy-test",
        action="store_true",
        help="Enable privacy leakage testing"
    )
    
    parser.add_argument(
        "--privacy-config",
        type=str,
        default="./configs/privacy_judge.json",
        help="Path to privacy judge configuration"
    )
    
    parser.add_argument(
        "--mitigation",
        type=str,
        default="none",
        choices=["none", "privacy_prompt", "cot_prompt", "pre_filter", "post_filter", "combined"],
        help="Privacy mitigation strategy to apply"
    )
    
    # Execution settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps per task"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout per task in seconds"
    )
    
    # Output settings
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./results/",
        help="Directory to store results"
    )
    
    parser.add_argument(
        "--save-traces",
        action="store_true",
        default=True,
        help="Save execution traces"
    )
    
    parser.add_argument(
        "--render-html",
        action="store_true",
        help="Render HTML trajectory visualization"
    )
    
    # Debug settings
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actual LLM calls (mock mode)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser


# =============================================================================
# Task Loading
# =============================================================================

def load_task_configs(
    config_dir: str,
    start_idx: int = 0,
    end_idx: int = 1000,
    vertical: Optional[str] = None,
) -> list[dict]:
    """Load task configurations from directory."""
    config_path = Path(config_dir)
    task_configs = []
    
    if not config_path.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return []
    
    # Load individual JSON files
    for i in range(start_idx, end_idx):
        file_path = config_path / f"{i}.json"
        if file_path.exists():
            with open(file_path) as f:
                config = json.load(f)
                
                # Filter by vertical if specified
                if vertical and vertical != "all":
                    config_vertical = config.get("vertical", "")
                    if config_vertical != vertical:
                        continue
                
                task_configs.append(config)
    
    # Also check for combined JSON file
    combined_file = config_path / "tasks.json"
    if combined_file.exists() and not task_configs:
        with open(combined_file) as f:
            all_configs = json.load(f)
            for config in all_configs[start_idx:end_idx]:
                if vertical and vertical != "all":
                    if config.get("vertical", "") != vertical:
                        continue
                task_configs.append(config)
    
    logger.info(f"Loaded {len(task_configs)} task configurations")
    return task_configs


# =============================================================================
# Result Tracking
# =============================================================================

class ResultTracker:
    """Track and aggregate benchmark results."""
    
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.utility_scores = []
        self.privacy_scores = []
        self.start_time = datetime.now()
    
    def add_result(
        self,
        task_id: int,
        task_completed: bool,
        privacy_leaked: bool,
        duration: float,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Add a task result."""
        result = {
            "task_id": task_id,
            "task_completed": task_completed,
            "privacy_leaked": privacy_leaked,
            "duration_seconds": duration,
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        self.results.append(result)
        
        if not error:
            self.utility_scores.append(1.0 if task_completed else 0.0)
            self.privacy_scores.append(0.0 if privacy_leaked else 1.0)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        total = len(self.results)
        errors = sum(1 for r in self.results if r["error"])
        successful = total - errors
        
        utility_mean = sum(self.utility_scores) / len(self.utility_scores) if self.utility_scores else 0
        privacy_mean = sum(self.privacy_scores) / len(self.privacy_scores) if self.privacy_scores else 0
        
        total_leaked = sum(1 for r in self.results if r["privacy_leaked"] and not r["error"])
        
        return {
            "total_tasks": total,
            "successful_runs": successful,
            "errors": errors,
            "utility_score": utility_mean,
            "privacy_score": privacy_mean,
            "privacy_leakage_rate": 1 - privacy_mean,
            "total_leaks": total_leaked,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
    
    def save(self, filename: str = "results.json"):
        """Save results to file."""
        output_path = self.result_dir / filename
        
        data = {
            "summary": self.get_summary(),
            "results": self.results,
            "config": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("APB BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Tasks:         {summary['total_tasks']}")
        print(f"Successful Runs:     {summary['successful_runs']}")
        print(f"Errors:              {summary['errors']}")
        print("-" * 60)
        print(f"Utility Score:       {summary['utility_score']:.2%}")
        print(f"Privacy Score:       {summary['privacy_score']:.2%}")
        print(f"Leakage Rate:        {summary['privacy_leakage_rate']:.2%}")
        print(f"Total Leaks:         {summary['total_leaks']}")
        print("-" * 60)
        print(f"Duration:            {summary['duration_seconds']:.1f}s")
        print("=" * 60 + "\n")


# =============================================================================
# Main Execution
# =============================================================================

def run_single_task(
    task_config: dict,
    args: argparse.Namespace,
    privacy_evaluator: Optional[Any] = None,
    mitigation_pipeline: Optional[Any] = None,
) -> dict:
    """
    Run a single task and return results.
    
    This is the core execution loop for each task.
    """
    from ..schemas.scenario import Scenario
    from ..harness import DryRunAdapter, AdapterConfig
    from ..detection import DetectionPipeline, DetectionConfig
    
    task_id = task_config.get("task_id", 0)
    intent = task_config.get("intent", "")
    sensitive_data = task_config.get("sensitive_data", [])
    
    start_time = time.time()
    
    try:
        # Create scenario from config
        scenario = Scenario.model_validate(task_config) if hasattr(Scenario, 'model_validate') else Scenario(**task_config)
        
        # Setup adapter
        adapter_config = AdapterConfig(
            model_name=args.model if not args.dry_run else "mock",
            temperature=args.temperature,
            max_steps=args.max_steps,
            timeout_seconds=args.timeout,
            dry_run=args.dry_run,
        )
        
        # Get system prompt with mitigation if enabled
        system_prompt = ""
        if mitigation_pipeline and args.mitigation != "none":
            system_prompt = mitigation_pipeline.get_system_prompt()
        
        # Run agent
        adapter = DryRunAdapter(config=adapter_config)
        result = adapter.run(scenario)
        
        # Evaluate privacy if enabled
        privacy_leaked = False
        privacy_score = 0.0
        
        if args.privacy_test and privacy_evaluator:
            action_str = result.final_output
            eval_result = privacy_evaluator.evaluate(action_str, sensitive_data)
            privacy_leaked = eval_result.leaked
            privacy_score = eval_result.score
        elif args.privacy_test and sensitive_data:
            # Simple keyword check fallback
            output_lower = result.final_output.lower()
            for item in sensitive_data:
                if isinstance(item, str) and item.lower() in output_lower:
                    privacy_leaked = True
                    break
        
        # Apply post-filter if enabled
        if mitigation_pipeline and args.mitigation in ["post_filter", "combined"]:
            filtered_output, leaked, _ = mitigation_pipeline.filter_output(
                result.final_output, sensitive_data
            )
            if leaked:
                privacy_leaked = True
        
        duration = time.time() - start_time
        
        return {
            "task_id": task_id,
            "task_completed": result.task_completed,
            "privacy_leaked": privacy_leaked,
            "privacy_score": privacy_score,
            "duration": duration,
            "output": result.final_output[:500],  # Truncate for storage
            "error": None,
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Task {task_id} failed: {e}")
        
        return {
            "task_id": task_id,
            "task_completed": False,
            "privacy_leaked": False,
            "privacy_score": 0.0,
            "duration": duration,
            "output": "",
            "error": str(e),
        }


def main():
    """Main entry point for APB CLI."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("APB - Agent Privacy Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Vertical: {args.vertical or 'all'}")
    logger.info(f"Privacy Test: {args.privacy_test}")
    logger.info(f"Mitigation: {args.mitigation}")
    logger.info(f"Config Dir: {args.test_config_dir}")
    logger.info("=" * 60)
    
    # Load task configurations
    task_configs = load_task_configs(
        args.test_config_dir,
        args.test_start_idx,
        args.test_end_idx,
        args.vertical,
    )
    
    if not task_configs:
        logger.error("No task configurations found!")
        logger.info("Creating sample tasks for demonstration...")
        
        # Create sample tasks for demo
        from ..generators.seed_generator import SeedDataGenerator
        from ..schemas.scenario import Vertical
        
        generator = SeedDataGenerator(use_mock=True)
        task_configs = [
            config.to_dict() 
            for config in generator.generate_vertical_tasks(
                Vertical.HEALTHCARE, n_tasks=10
            )
        ]
    
    logger.info(f"Running {len(task_configs)} tasks...")
    
    # Initialize evaluators and mitigations
    privacy_evaluator = None
    mitigation_pipeline = None
    
    if args.privacy_test:
        try:
            from ..detection.privacy_evaluator import PrivacyEvaluator, JudgeConfig, MockBackend
            
            if args.dry_run:
                privacy_evaluator = PrivacyEvaluator(backend=MockBackend())
            else:
                privacy_evaluator = PrivacyEvaluator()
            
            logger.info("Privacy evaluator initialized")
        except Exception as e:
            logger.warning(f"Could not initialize privacy evaluator: {e}")
    
    if args.mitigation != "none":
        try:
            from ..defenses.mitigation import create_mitigation
            mitigation_pipeline = create_mitigation(
                args.mitigation,
                use_mock=args.dry_run,
            )
            logger.info(f"Mitigation pipeline initialized: {args.mitigation}")
        except Exception as e:
            logger.warning(f"Could not initialize mitigation: {e}")
    
    # Initialize result tracker
    result_tracker = ResultTracker(args.result_dir)
    
    # Run tasks
    for i, task_config in enumerate(task_configs):
        task_id = task_config.get("task_id", i)
        
        logger.info(f"[{i+1}/{len(task_configs)}] Running task {task_id}...")
        
        result = run_single_task(
            task_config,
            args,
            privacy_evaluator,
            mitigation_pipeline,
        )
        
        result_tracker.add_result(
            task_id=task_id,
            task_completed=result["task_completed"],
            privacy_leaked=result["privacy_leaked"],
            duration=result["duration"],
            error=result.get("error"),
            metadata={"output_preview": result.get("output", "")[:200]},
        )
        
        # Log progress
        if result["error"]:
            logger.error(f"  Task {task_id} ERROR: {result['error']}")
        else:
            status = "✓" if result["task_completed"] else "✗"
            privacy = "LEAK" if result["privacy_leaked"] else "OK"
            logger.info(f"  Task {task_id}: {status} | Privacy: {privacy} | {result['duration']:.2f}s")
        
        # Print running totals periodically
        if (i + 1) % 10 == 0:
            summary = result_tracker.get_summary()
            logger.info(
                f"  Progress: {summary['successful_runs']}/{summary['total_tasks']} "
                f"| Utility: {summary['utility_score']:.1%} "
                f"| Privacy: {summary['privacy_score']:.1%}"
            )
    
    # Final summary
    result_tracker.print_summary()
    
    # Save results
    result_file = result_tracker.save(
        f"results_{args.model.replace('-', '_')}_{args.mitigation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    logger.info(f"Results saved to: {result_file}")
    logger.info(f"Log file: {LOG_FILE}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
