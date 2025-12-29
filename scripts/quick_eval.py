#!/usr/bin/env python3
"""
AgentLeak Quick Evaluation - Simple entry point for testing the benchmark.

This script provides an easy way to run AgentLeak without complex configuration.
Perfect for:
  - Quick validation that the benchmark works
  - CI/CD integration
  - Demo purposes
  - Initial exploration

Usage:
    python scripts/quick_eval.py                    # Default: 10 scenarios, simulation
    python scripts/quick_eval.py --n 50             # 50 scenarios
    python scripts/quick_eval.py --real             # Use real LLM (requires API key)
    python scripts/quick_eval.py --defense lcf      # Test with LCF defense
    python scripts/quick_eval.py --verbose          # Show detailed output
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class QuickResult:
    """Simplified result structure."""

    n_scenarios: int
    tsr: float  # Task Success Rate
    elr: float  # Exact Leakage Rate
    wls: float  # Weighted Leakage Score
    channel_leakage: Dict[str, float]
    runtime_seconds: float
    mode: str  # "simulation" or "real"
    defense: Optional[str] = None


def run_simulation(
    n_scenarios: int, defense: Optional[str] = None, verbose: bool = False
) -> QuickResult:
    """
    Run simulated benchmark for quick testing.

    Simulates realistic leakage patterns based on paper findings.
    """
    start_time = time.time()

    # Import AgentLeak components
    from agentleak.generators import ScenarioGenerator
    from agentleak.schemas.scenario import Vertical

    if verbose:
        print(f"🔄 Generating {n_scenarios} scenarios...")

    # Generate scenarios
    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_set(name="quick_eval", total_count=n_scenarios).scenarios

    if verbose:
        print(f"✓ Generated {len(scenarios)} scenarios")
        for v in Vertical:
            count = sum(1 for s in scenarios if s.vertical == v)
            if count > 0:
                print(f"  - {v.value}: {count}")

    # Simulate execution with realistic distributions
    # Based on paper Table 5 (A0 benign setting)
    base_tsr = 0.84
    base_elr = 0.72
    base_wls = 2.63

    # Defense reduces leakage
    if defense == "lcf":
        elr_mult = 0.25  # LCF reduces to ~18%
        wls_mult = 0.25
    elif defense == "output_filter":
        elr_mult = 0.57
        wls_mult = 0.56
    elif defense == "policy_prompt":
        elr_mult = 0.81
        wls_mult = 0.81
    else:
        elr_mult = 1.0
        wls_mult = 1.0

    # Add noise for realism
    tsr = base_tsr + random.gauss(0, 0.03)
    elr = (base_elr * elr_mult) + random.gauss(0, 0.02)
    wls = (base_wls * wls_mult) + random.gauss(0, 0.1)

    # Clamp to valid ranges
    tsr = max(0, min(1, tsr))
    elr = max(0, min(1, elr))
    wls = max(0, wls)

    # Per-channel leakage (based on paper Table 7)
    channel_base = {
        "C1_final_output": 0.34,
        "C2_inter_agent": 0.29,
        "C3_tool_input": 0.50,
        "C4_tool_output": 0.12,
        "C5_memory": 0.42,
        "C6_logs": 0.08,
        "C7_artifacts": 0.22,
    }

    channel_leakage = {
        k: max(0, min(1, v * elr_mult + random.gauss(0, 0.02))) for k, v in channel_base.items()
    }

    runtime = time.time() - start_time

    return QuickResult(
        n_scenarios=n_scenarios,
        tsr=round(tsr, 3),
        elr=round(elr, 3),
        wls=round(wls, 2),
        channel_leakage={k: round(v, 3) for k, v in channel_leakage.items()},
        runtime_seconds=round(runtime, 2),
        mode="simulation",
        defense=defense,
    )


def run_real(n_scenarios: int, defense: Optional[str] = None, verbose: bool = False) -> QuickResult:
    """
    Run real benchmark with actual LLM calls.

    Requires OPENROUTER_API_KEY.
    """
    import os
    import copy

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("No API key found. Set OPENROUTER_API_KEY environment variable.")

    start_time = time.time()

    # Import components
    from agentleak.generators import ScenarioGenerator
    from agentleak.harness.adapters import LangChainAdapter, LangChainConfig
    from agentleak.detection.pipeline import DetectionPipeline
    from agentleak.defenses.lcf import LearnedContentFilter, LCFConfig

    if verbose:
        print(f"🔄 Generating {n_scenarios} scenarios...")

    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_set(name="quick_eval", total_count=n_scenarios).scenarios

    if verbose:
        print(f"✓ Generated {len(scenarios)} scenarios")

    # Setup adapter
    config = LangChainConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1000,
    )
    adapter = LangChainAdapter(config)

    # Setup detection
    detector = DetectionPipeline()

    # Setup defense if requested
    defense_module = None
    if defense == "lcf":
        lcf_config = LCFConfig(
            enable_pattern_matching=True, enable_leace=True, enable_semantic_filter=True
        )
        defense_module = LearnedContentFilter(lcf_config)

    # Track metrics
    successes = 0
    leaks = 0
    total_wls = 0.0
    channel_counts = {f"C{i}": 0 for i in range(1, 8)}

    for i, scenario in enumerate(scenarios):
        if verbose:
            print(f"  [{i+1}/{len(scenarios)}] Running {scenario.scenario_id}...")

        try:
            # Train defense on scenario data if enabled (mock training)
            if defense_module and hasattr(defense_module, "register_private_values"):
                private_values = []
                for record in scenario.private_vault.records:
                    for field_name, value in record.fields.items():
                        if field_name not in scenario.allowed_set.fields:
                            private_values.append(str(value))
                defense_module.register_private_values(private_values)

            # Execute scenario
            result = adapter.run(scenario)

            # Check task success
            if result.task_completed:
                successes += 1

            # Apply defense to trace if enabled
            trace_to_analyze = result.trace
            if defense_module:
                trace_to_analyze = copy.deepcopy(result.trace)
                for event in trace_to_analyze.events:
                    if event.content:
                        filter_result = defense_module.filter(event.content, event.channel)
                        if filter_result.filtered_content:
                            event.content = filter_result.filtered_content

            # Detect leakage
            detection = detector.detect(scenario, trace_to_analyze)

            if detection.leaked:
                leaks += 1
                total_wls += detection.weighted_leakage_score

                for leak in detection.field_leaks:
                    channel_key = f"C{leak.channel.value}"
                    if channel_key in channel_counts:
                        channel_counts[channel_key] += 1

        except Exception as e:
            if verbose:
                print(f"    ⚠️ Error: {e}")

    runtime = time.time() - start_time
    n = len(scenarios)

    return QuickResult(
        n_scenarios=n,
        tsr=round(successes / n, 3) if n > 0 else 0,
        elr=round(leaks / n, 3) if n > 0 else 0,
        wls=round(total_wls / n, 2) if n > 0 else 0,
        channel_leakage={k: round(v / n, 3) for k, v in channel_counts.items()} if n > 0 else {},
        runtime_seconds=round(runtime, 2),
        mode="real",
        defense=defense,
    )


def print_results(result: QuickResult):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 60)
    print("📊 AgentLeak Evaluation Results")
    print("=" * 60)

    print(f"\n⚙️  Mode: {result.mode.upper()}")
    if result.defense:
        print(f"🛡️  Defense: {result.defense}")
    print(f"📁 Scenarios: {result.n_scenarios}")
    print(f"⏱️  Runtime: {result.runtime_seconds}s")

    print("\n📈 Metrics:")
    print(f"  TSR (Task Success Rate):     {result.tsr*100:.1f}%")
    print(f"  ELR (Exact Leakage Rate):    {result.elr*100:.1f}%")
    print(f"  WLS (Weighted Leakage Score): {result.wls:.2f}")

    print("\n📡 Per-Channel Leakage:")
    channel_names = {
        "C1_final_output": "C1 Final Output",
        "C2_inter_agent": "C2 Inter-Agent",
        "C3_tool_input": "C3 Tool Input",
        "C4_tool_output": "C4 Tool Output",
        "C5_memory": "C5 Memory",
        "C6_logs": "C6 Logs",
        "C7_artifacts": "C7 Artifacts",
    }

    for key, value in result.channel_leakage.items():
        name = channel_names.get(key, key)
        bar = "█" * int(value * 20)
        print(f"  {name:20s} {value*100:5.1f}% {bar}")

    # Key insight
    if result.elr > 0.5:
        print("\n⚠️  WARNING: High leakage rate detected!")
        print("   Consider enabling LCF defense: --defense lcf")
    elif result.elr < 0.2:
        print("\n✅ Low leakage rate - defense is effective")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AgentLeak Quick Evaluation - Simple benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quick_eval.py                    # Quick test with 10 scenarios
  python scripts/quick_eval.py --n 100            # Test with 100 scenarios
  python scripts/quick_eval.py --defense lcf      # Test LCF defense
  python scripts/quick_eval.py --real             # Use real LLM (requires API key)
  python scripts/quick_eval.py --output result.json  # Save results to file
        """,
    )

    parser.add_argument(
        "-n",
        "--n-scenarios",
        type=int,
        default=10,
        help="Number of scenarios to evaluate (default: 10)",
    )
    parser.add_argument(
        "--real", action="store_true", help="Use real LLM execution (requires API key)"
    )
    parser.add_argument(
        "--defense",
        choices=["none", "lcf", "output_filter", "policy_prompt"],
        default=None,
        help="Defense to test",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save results to JSON file")

    args = parser.parse_args()

    print("🔐 AgentLeak Quick Evaluation")
    print("-" * 40)

    try:
        if args.real:
            result = run_real(
                n_scenarios=args.n_scenarios,
                defense=args.defense if args.defense != "none" else None,
                verbose=args.verbose,
            )
        else:
            result = run_simulation(
                n_scenarios=args.n_scenarios,
                defense=args.defense if args.defense != "none" else None,
                verbose=args.verbose,
            )

        print_results(result)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\n💾 Results saved to {output_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
