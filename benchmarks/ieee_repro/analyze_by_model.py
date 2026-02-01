#!/usr/bin/env python3
"""Analyze traces by model to identify discrepancies and issues."""

import json
import os
from collections import defaultdict
from pathlib import Path

TRACES_DIR = Path(__file__).parent / "results" / "traces"

def analyze_trace(trace_data):
    """Extract pre-computed results from trace."""
    # Use pre-computed results from benchmark.py
    results_data = trace_data.get("results", {})

    results = {
        "C1_leaked": results_data.get("c1_leaked", False),
        "C2_leaked": results_data.get("c2_leaked", False),
        "C5_leaked": results_data.get("c5_leaked", False),
        "any_leak": results_data.get("single_leaked", False) or results_data.get("multi_leaked", False),
        "is_multi_agent": results_data.get("multi_leaked") is not None and "multi_leaked" in results_data,
        "attack_family": trace_data.get("attack_family"),
    }

    # Check if multi-agent by looking at channel_messages
    channel_messages = trace_data.get("channel_messages", [])
    has_c2 = any(msg.get("channel") == "C2" for msg in channel_messages)
    results["is_multi_agent"] = has_c2

    # Also check llm_calls for multi-agent indicators
    llm_calls = trace_data.get("llm_calls", [])
    for call in llm_calls:
        system_prompt = call.get("system_prompt", "")
        if "Coordinator" in system_prompt or "Worker" in system_prompt:
            results["is_multi_agent"] = True
            break

    return results

def main():
    # Collect all traces by model
    model_traces = defaultdict(list)

    trace_files = list(TRACES_DIR.glob("trace_*.json"))
    print(f"Found {len(trace_files)} trace files")

    for trace_file in trace_files:
        try:
            with open(trace_file) as f:
                trace = json.load(f)
            model = trace.get("model", "unknown")

            analysis = analyze_trace(trace)
            analysis["vertical"] = trace.get("vertical", "unknown")
            model_traces[model].append(analysis)
        except Exception as e:
            continue

    # Compute statistics per model
    print("\n" + "="*100)
    print("ANALYSIS BY MODEL")
    print("="*100)

    all_stats = {}

    for model, traces in sorted(model_traces.items()):
        n = len(traces)
        if n == 0:
            continue

        c1_leaks = sum(1 for t in traces if t["C1_leaked"])
        c2_leaks = sum(1 for t in traces if t["C2_leaked"])
        c5_leaks = sum(1 for t in traces if t["C5_leaked"])
        any_leaks = sum(1 for t in traces if t["any_leak"])

        multi = [t for t in traces if t["is_multi_agent"]]
        single = [t for t in traces if not t["is_multi_agent"]]

        multi_leaks = sum(1 for t in multi if t["any_leak"]) if multi else 0
        single_leaks = sum(1 for t in single if t["any_leak"]) if single else 0

        multi_rate = (multi_leaks / len(multi) * 100) if multi else 0
        single_rate = (single_leaks / len(single) * 100) if single else 0

        # H1: Leaked in C2 or C5 but NOT in C1
        h1_count = sum(1 for t in traces if (t["C2_leaked"] or t["C5_leaked"]) and not t["C1_leaked"])
        h1_rate = h1_count / n * 100

        # Ratio
        ratio = multi_rate / single_rate if single_rate > 0 else float('inf')

        stats = {
            "n": n,
            "C1": c1_leaks / n * 100,
            "C2": c2_leaks / n * 100,
            "C5": c5_leaks / n * 100,
            "any_leak": any_leaks / n * 100,
            "multi_n": len(multi),
            "single_n": len(single),
            "multi_rate": multi_rate,
            "single_rate": single_rate,
            "ratio": ratio,
            "H1": h1_rate,
        }
        all_stats[model] = stats

        print(f"\n### {model} (n={n})")
        print(f"  C1 (Final Output):    {stats['C1']:5.1f}%")
        print(f"  C2 (Inter-Agent):     {stats['C2']:5.1f}%")
        print(f"  C5 (Memory):          {stats['C5']:5.1f}%")
        print(f"  Any Leak:             {stats['any_leak']:5.1f}%")
        print(f"  ---")
        print(f"  Single-agent (n={len(single)}): {single_rate:5.1f}%")
        print(f"  Multi-agent (n={len(multi)}):  {multi_rate:5.1f}%")
        print(f"  Multi/Single Ratio:   {ratio:5.1f}x")
        print(f"  ---")
        print(f"  H1 (Audit Gap):       {h1_rate:5.1f}%")

        # Check claims
        print(f"\n  CLAIMS CHECK:")

        # Claim 1: C2 > C1
        c2_gt_c1 = stats['C2'] > stats['C1']
        print(f"  [{'✓' if c2_gt_c1 else '✗'}] Claim 1: C2 > C1 ({stats['C2']:.1f}% vs {stats['C1']:.1f}%)")

        # Claim 2: Multi > Single
        multi_gt_single = stats['multi_rate'] > stats['single_rate']
        print(f"  [{'✓' if multi_gt_single else '✗'}] Claim 2: Multi > Single ({stats['multi_rate']:.1f}% vs {stats['single_rate']:.1f}%)")

        # Claim 3: H1 > 0 (audit gap exists)
        h1_exists = stats['H1'] > 0
        print(f"  [{'✓' if h1_exists else '✗'}] Claim 3: H1 > 0 (Audit gap = {stats['H1']:.1f}%)")

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Model':<50} {'n':>6} {'C1':>7} {'C2':>7} {'C5':>7} {'Ratio':>7} {'H1':>7} {'C2>C1':>6}")
    print("-"*100)

    for model, stats in sorted(all_stats.items(), key=lambda x: -x[1]['n']):
        c2_gt_c1 = "✓" if stats['C2'] > stats['C1'] else "✗"
        print(f"{model:<50} {stats['n']:>6} {stats['C1']:>6.1f}% {stats['C2']:>6.1f}% {stats['C5']:>6.1f}% {stats['ratio']:>6.1f}x {stats['H1']:>6.1f}% {c2_gt_c1:>6}")

    # Identify issues
    print("\n" + "="*100)
    print("ISSUES IDENTIFIED")
    print("="*100)

    for model, stats in all_stats.items():
        issues = []

        if stats['C2'] <= stats['C1']:
            issues.append(f"C2 ({stats['C2']:.1f}%) <= C1 ({stats['C1']:.1f}%): Internal channel not leaking more than output")

        if stats['single_n'] == 0:
            issues.append(f"No single-agent scenarios - cannot compute ratio")
        elif stats['ratio'] < 1.5:
            issues.append(f"Multi/Single ratio ({stats['ratio']:.1f}x) too low - expected >2x")

        if stats['H1'] < 10:
            issues.append(f"H1 audit gap ({stats['H1']:.1f}%) very low - may indicate detection working well or data issue")

        if issues:
            print(f"\n### {model}")
            for issue in issues:
                print(f"  ⚠ {issue}")

    # Save detailed stats
    output_file = TRACES_DIR.parent / "model_stats.json"
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDetailed stats saved to: {output_file}")

if __name__ == "__main__":
    main()
