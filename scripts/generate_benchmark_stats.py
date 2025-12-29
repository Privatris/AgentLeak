#!/usr/bin/env python3
"""
AgentLeak Benchmark Statistics Generator.

Generates all statistics for README and paper consistency.
"""

import json
import sys

sys.path.insert(0, ".")

from agentleak.generators import ScenarioGenerator
from agentleak.generators.hard_scenarios import HardScenarioGenerator
from agentleak.generators.cross_agent_scenarios import generate_all_cross_agent_scenarios
from agentleak.harness import DryRunAdapter, AdapterConfig
from agentleak.detection import DetectionPipeline
from agentleak.metrics import MetricsCalculator
from agentleak.schemas import Vertical, AttackFamily, AttackClass, Channel


def main():
    print("=" * 60)
    print("AGENTLEAK BENCHMARK STATISTICS GENERATION")
    print("=" * 60)

    # 1. Scenario Generation Stats
    print("\n📊 1. SCENARIO GENERATION")
    gen = ScenarioGenerator(seed=42)

    # Use generate_set which includes multi-agent scenarios (60% multi-agent as claimed in paper)
    scenario_set = gen.generate_set(
        name="agentleak_full_1000",
        total_count=1000,
        multi_agent_ratio=0.6,  # Paper claims 60% multi-agent
    )
    scenarios = scenario_set.scenarios

    print(f"  Total scenarios: {len(scenarios)}")
    print(f"  By vertical:")
    for v in Vertical:
        count = sum(1 for s in scenarios if s.vertical == v)
        print(f"    - {v.value}: {count}")

    # 2. Attack Taxonomy Stats
    print("\n🔬 2. ATTACK TAXONOMY")
    print(f"  Attack Families: {len(AttackFamily)}")
    for af in AttackFamily:
        print(f"    - {af.name}: {af.value}")
    print(f"  Attack Classes: {len(AttackClass)}")

    # 3. Channel Stats
    print("\n📡 3. LEAKAGE CHANNELS")
    print(f"  Total channels: {len(Channel)}")
    for c in Channel:
        print(f"    - {c.name}: {c.value}")

    # 4. Multi-Agent Distribution
    print("\n🤖 4. MULTI-AGENT DISTRIBUTION")
    single_agent = sum(1 for s in scenarios if len(s.agents) == 1)
    two_agent = sum(1 for s in scenarios if len(s.agents) == 2)
    three_plus = sum(1 for s in scenarios if len(s.agents) >= 3)
    print(f"  Single-agent: {single_agent} ({single_agent/len(scenarios)*100:.1f}%)")
    print(f"  2-agent: {two_agent} ({two_agent/len(scenarios)*100:.1f}%)")
    print(f"  3+ agent: {three_plus} ({three_plus/len(scenarios)*100:.1f}%)")

    # 5. Cross-Agent Scenarios
    print("\n⚔️ 5. CROSS-AGENT ATTACK SCENARIOS")
    try:
        cross_scenarios = generate_all_cross_agent_scenarios()
        print(f"  Cross-agent scenarios: {len(cross_scenarios)}")
    except Exception as e:
        cross_scenarios = []
        print(f"  Error generating: {e}")

    # 6. Hard Scenarios
    print("\n💪 6. HARD SCENARIOS")
    hard_gen = HardScenarioGenerator(seed=42)
    hard_scenarios = []
    for v in Vertical:
        for _ in range(10):
            hard_scenarios.append(hard_gen.generate(v))
    print(f"  Hard scenarios: {len(hard_scenarios)}")

    # 7. DryRun Benchmark
    print("\n🏃 7. DRYRUN BENCHMARK (100 scenarios)")
    adapter = DryRunAdapter(AdapterConfig())
    pipeline = DetectionPipeline()
    results = []
    for i, s in enumerate(scenarios[:100]):
        result = adapter.run(s)
        if result.trace:
            detection = pipeline.detect(s, result.trace)
            results.append(
                {
                    "scenario_id": s.scenario_id,
                    "leaked": detection.leaked,
                    "n_leaks": len(detection.field_leaks),
                    "task_completed": result.task_completed,
                }
            )

    n_leaked = sum(1 for r in results if r["leaked"])
    n_task_ok = sum(1 for r in results if r["task_completed"])
    print(f"  Scenarios tested: {len(results)}")
    print(f"  Leaks detected: {n_leaked} ({n_leaked/len(results)*100:.1f}%)")
    print(f"  Tasks completed: {n_task_ok} ({n_task_ok/len(results)*100:.1f}%)")

    # Summary JSON
    print("\n" + "=" * 60)
    print("SUMMARY (for README/paper)")
    print("=" * 60)
    summary = {
        "total_scenarios": len(scenarios),
        "attack_families": len(AttackFamily),
        "attack_classes": len(AttackClass),
        "leakage_channels": len(Channel),
        "verticals": len(Vertical),
        "single_agent_pct": single_agent / len(scenarios) * 100,
        "two_agent_pct": two_agent / len(scenarios) * 100,
        "three_plus_agent_pct": three_plus / len(scenarios) * 100,
        "multi_agent_pct": (two_agent + three_plus) / len(scenarios) * 100,
        "cross_agent_scenarios": len(cross_scenarios),
        "hard_scenarios": len(hard_scenarios),
        "dryrun_leak_rate": n_leaked / len(results) * 100,
        "dryrun_tsr": n_task_ok / len(results) * 100,
    }
    print(json.dumps(summary, indent=2))

    # Save to file
    with open("benchmark_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n📁 Saved to benchmark_stats.json")


if __name__ == "__main__":
    main()
