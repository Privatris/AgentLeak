import os
import sys
import time
import copy
from pathlib import Path
from dataclasses import asdict
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentleak.generators import ScenarioGenerator
from agentleak.harness.adapters import LangChainAdapter, LangChainConfig
from agentleak.detection.pipeline import DetectionPipeline
from agentleak.defenses.sanitizer import OutputSanitizer, SanitizerConfig
from agentleak.schemas.scenario import Vertical


def run_comparison(n_scenarios=3):
    print("⚔️  AgentLeak Defense Comparison (Agnostic Mode)")
    print("------------------------------------------------")

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY not set")
        return
    # Set OPENAI_API_KEY for framework compatibility (uses OpenRouter)
    import os

    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    # 1. Generate Scenarios
    print(f"🔄 Generating {n_scenarios} scenarios...")
    generator = ScenarioGenerator(seed=42)
    scenarios = generator.generate_set(name="compare", total_count=n_scenarios).scenarios
    print(f"✓ Generated {len(scenarios)} scenarios")

    # 2. Setup Components
    adapter_config = LangChainConfig(model_name="gpt-4o-mini", temperature=0.0)
    adapter = LangChainAdapter(adapter_config)
    detector = DetectionPipeline()

    # 3. Define Defenses to Compare
    defenses = {
        "Baseline (None)": None,
        "Regex Sanitizer": OutputSanitizer(SanitizerConfig(
            detect_ssn=True, detect_email=True, detect_canaries=True
        ))
    }

    results = {name: {"leaks": 0, "total_wls": 0.0} for name in defenses}

    # 4. Run Benchmark
    for i, scenario in enumerate(scenarios):
        print(f"\n📝 Scenario {i+1}/{n_scenarios}: {scenario.scenario_id}")
        
        # Execute Agent (Once)
        try:
            exec_result = adapter.run(scenario)
        except Exception as e:
            print(f"  ❌ Execution failed: {e}")
            continue

        # Evaluate against each defense
        for defense_name, defense_module in defenses.items():
            # Copy trace to avoid side effects
            trace_copy = copy.deepcopy(exec_result.trace)

            # Apply Defense (if any)
            if defense_module:
                filtered_count = 0
                for event in trace_copy.events:
                    if event.content:
                        res = defense_module.filter(event.content, event.channel)
                        if res.filtered_content:
                            event.content = res.filtered_content
                            filtered_count += 1
                # print(f"  [{defense_name}] Filtered {filtered_count} events")

            # Detect Leakage
            detection = detector.detect(scenario, trace_copy)

            if detection.leaked:
                results[defense_name]["leaks"] += 1
                results[defense_name]["total_wls"] += detection.weighted_leakage_score

            print(
                f"  - {defense_name:20s}: WLS={detection.weighted_leakage_score:.2f} | Leaked={detection.leaked}"
            )

    # 5. Summary
    print("\n📊 Final Comparison Results")
    print("========================================")
    print(f"{'Defense':<20} | {'Leak Rate':<10} | {'Avg WLS':<10} | {'Reduction'}")
    print("-" * 60)

    baseline_wls = results["Baseline (None)"]["total_wls"] / n_scenarios

    for name, metrics in results.items():
        avg_wls = metrics["total_wls"] / n_scenarios
        leak_rate = (metrics["leaks"] / n_scenarios) * 100

        reduction = 0.0
        if baseline_wls > 0:
            reduction = ((baseline_wls - avg_wls) / baseline_wls) * 100

        print(f"{name:<20} | {leak_rate:5.1f}%    | {avg_wls:5.2f}      | -{reduction:.1f}%")


if __name__ == "__main__":
    run_comparison()
