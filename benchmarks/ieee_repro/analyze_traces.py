#!/usr/bin/env python3
"""
Analyze traces and generate statistics for paper figures.

Reads all traces from results/traces/ and computes:
- Channel leak rates (C1, C2, C5)
- H1 validation (audit gap)
- Breakdown by vertical, topology, and attack family
- Per-channel attack family effectiveness

Outputs: results/paper_stats.json
"""

import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return round(max(0, center - margin) * 100, 1), round(min(1, center + margin) * 100, 1)


def get_attack_family(attack_class: str) -> str:
    """Map attack class to family (F1-F6)."""
    family_map = {
        # F1: Prompt & Instruction
        "direct_prompt_injection": "F1",
        "role_confusion": "F1",
        "context_override": "F1",
        "format_coercion": "F1",
        "crescendo_escalation": "F1",
        "academic_framing_bypass": "F1",
        # F2: Tool Surface
        "indirect_prompt_injection": "F2",
        "tool_output_poisoning": "F2",
        "retrieval_trap": "F2",
        "link_following_exfiltration": "F2",
        "api_parameter_injection": "F2",
        "hallucination_exploitation": "F2",
        # F3: Memory & Persistence
        "memory_write_exfiltration": "F3",
        "vector_store_leakage": "F3",
        "log_leakage": "F3",
        "artifact_leakage": "F3",
        "session_state_poisoning": "F3",
        # F4: Multi-Agent
        "cross_agent_collusion": "F4",
        "role_boundary_violation": "F4",
        "delegation_exploit": "F4",
        "shared_memory_poisoning": "F4",
        "actor_network_manipulation": "F4",
        "consensus_manipulation": "F4",
        "orchestrator_hijacking": "F4",
        "agent_identity_spoofing": "F4",
        # F5: Reasoning/CoT
        "logic_puzzle_jailbreak": "F5",
        "cot_forging": "F5",
        "special_token_injection": "F5",
        "reasoning_hijack": "F5",
        "word_game_obfuscation": "F5",
        # F6: Evasion
        "unicode_tag_smuggling": "F6",
        "text_shuffle_attack": "F6",
    }
    return family_map.get(attack_class, "A0")  # A0 for benign


def analyze_traces(traces_dir: Path) -> dict:
    """Analyze all traces and compute comprehensive statistics."""
    traces = []
    for f in sorted(traces_dir.glob("trace_*.json")):
        try:
            traces.append(json.load(open(f)))
        except Exception:
            continue

    if not traces:
        return {"error": "No traces found"}

    n = len(traces)

    # Initialize counters
    stats = {
        "total": n,
        "channels": {"C1": 0, "C2": 0, "C5": 0},
        "single_leaked": 0,
        "multi_leaked": 0,
        "h1_true": 0,
        "by_vertical": defaultdict(lambda: {"total": 0, "c1": 0, "c2": 0, "c5": 0}),
        "by_attack_class": defaultdict(lambda: {"total": 0, "success": 0}),
        "by_attack_family": defaultdict(lambda: {"total": 0, "success": 0}),
        # Per-channel attack family breakdown
        "channel_attack_family": {
            "C1": defaultdict(lambda: {"total": 0, "leaked": 0}),
            "C2": defaultdict(lambda: {"total": 0, "leaked": 0}),
            "C5": defaultdict(lambda: {"total": 0, "leaked": 0}),
        },
        # Topology breakdown (2-agent vs 3+)
        "by_topology": {
            "2_agent": {"total": 0, "c1": 0, "c2": 0, "c5": 0, "h1_true": 0},
            "3plus_agent": {"total": 0, "c1": 0, "c2": 0, "c5": 0, "h1_true": 0},
        },
    }

    for t in traces:
        results = t.get("results", {})
        c1 = results.get("c1_leaked", False)
        c2 = results.get("c2_leaked", False)
        c5 = results.get("c5_leaked", False)

        if c1:
            stats["channels"]["C1"] += 1
        if c2:
            stats["channels"]["C2"] += 1
        if c5:
            stats["channels"]["C5"] += 1

        if results.get("single_leaked") or results.get("single_agent_leaked"):
            stats["single_leaked"] += 1
        if results.get("multi_leaked") or results.get("multi_agent_leaked"):
            stats["multi_leaked"] += 1

        # H1: C1 safe but internal leaked
        h1_true = not c1 and (c2 or c5)
        if h1_true:
            stats["h1_true"] += 1

        # By vertical
        v = t.get("vertical", "unknown")
        stats["by_vertical"][v]["total"] += 1
        if c1:
            stats["by_vertical"][v]["c1"] += 1
        if c2:
            stats["by_vertical"][v]["c2"] += 1
        if c5:
            stats["by_vertical"][v]["c5"] += 1

        # Attack analysis
        attack = t.get("attack_family")
        if attack:
            family = get_attack_family(attack)
            any_leak = c1 or c2 or c5

            stats["by_attack_class"][attack]["total"] += 1
            if any_leak:
                stats["by_attack_class"][attack]["success"] += 1

            stats["by_attack_family"][family]["total"] += 1
            if any_leak:
                stats["by_attack_family"][family]["success"] += 1

            # Per-channel attack family
            for ch, leaked in [("C1", c1), ("C2", c2), ("C5", c5)]:
                stats["channel_attack_family"][ch][family]["total"] += 1
                if leaked:
                    stats["channel_attack_family"][ch][family]["leaked"] += 1
        else:
            # Benign scenario (A0)
            family = "A0"
            for ch, leaked in [("C1", c1), ("C2", c2), ("C5", c5)]:
                stats["channel_attack_family"][ch][family]["total"] += 1
                if leaked:
                    stats["channel_attack_family"][ch][family]["leaked"] += 1

        # Topology (assume 2-agent for now, multi-agent scenarios)
        # In current benchmark, all are effectively 2-agent (coordinator + worker)
        topo = "2_agent"
        stats["by_topology"][topo]["total"] += 1
        if c1:
            stats["by_topology"][topo]["c1"] += 1
        if c2:
            stats["by_topology"][topo]["c2"] += 1
        if c5:
            stats["by_topology"][topo]["c5"] += 1
        if h1_true:
            stats["by_topology"][topo]["h1_true"] += 1

    # Build result
    result = {
        "metadata": {
            "total_traces": n,
            "source": str(traces_dir),
        },
        "channel_rates": {
            "C1": round(stats["channels"]["C1"] / n * 100, 1),
            "C2": round(stats["channels"]["C2"] / n * 100, 1),
            "C5": round(stats["channels"]["C5"] / n * 100, 1),
        },
        "leak_rates": {
            "single_agent": round(stats["single_leaked"] / n * 100, 1),
            "multi_agent": round(stats["multi_leaked"] / n * 100, 1),
        },
        "h1_validation": {
            "total": n,
            "h1_true": stats["h1_true"],
            "h1_rate": round(stats["h1_true"] / n * 100, 1),
            "ci_95": wilson_ci(stats["h1_true"], n),
            # For paper figure: detected vs missed
            "detected_rate": round(stats["channels"]["C1"] / n * 100, 1),
            "missed_rate": round(stats["h1_true"] / n * 100, 1),
        },
        "by_vertical": {},
        "by_topology": {},
        "by_attack_family": {},
        "channel_dominant_attack": {},
    }

    # Vertical breakdown
    for v, data in stats["by_vertical"].items():
        if data["total"] > 0:
            result["by_vertical"][v] = {
                "total": data["total"],
                "c1_rate": round(data["c1"] / data["total"] * 100, 1),
                "c2_rate": round(data["c2"] / data["total"] * 100, 1),
                "c5_rate": round(data["c5"] / data["total"] * 100, 1),
            }

    # Topology breakdown
    for topo, data in stats["by_topology"].items():
        if data["total"] > 0:
            result["by_topology"][topo] = {
                "total": data["total"],
                "c1_rate": round(data["c1"] / data["total"] * 100, 1),
                "c2_rate": round(data["c2"] / data["total"] * 100, 1),
                "c5_rate": round(data["c5"] / data["total"] * 100, 1),
                "h1_rate": round(data["h1_true"] / data["total"] * 100, 1),
            }

    # Attack family breakdown
    for family, data in stats["by_attack_family"].items():
        if data["total"] >= 3:
            rate = data["success"] / data["total"] * 100
            result["by_attack_family"][family] = {
                "total": data["total"],
                "success": data["success"],
                "rate": round(rate, 1),
                "ci_95": wilson_ci(data["success"], data["total"]),
            }

    # Determine dominant attack family per channel
    for ch in ["C1", "C2", "C5"]:
        ch_families = stats["channel_attack_family"][ch]
        if ch_families:
            # Find family with highest leak rate (excluding A0 if others exist)
            best_family = None
            best_rate = 0
            for family, data in ch_families.items():
                if data["total"] >= 3:  # Minimum sample
                    rate = data["leaked"] / data["total"]
                    # Prefer attack families over A0
                    if family != "A0" and rate >= best_rate:
                        best_family = family
                        best_rate = rate
                    elif best_family is None and rate > best_rate:
                        best_family = family
                        best_rate = rate

            result["channel_dominant_attack"][ch] = {
                "family": best_family or "A0",
                "rate": round(best_rate * 100, 1) if best_family else 0,
            }

    return result


def main():
    script_dir = Path(__file__).parent
    traces_dir = script_dir / "results" / "traces"
    output_file = script_dir / "results" / "paper_stats.json"

    print(f"Analyzing traces from: {traces_dir}")

    stats = analyze_traces(traces_dir)

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return 1

    # Save
    output_file.write_text(json.dumps(stats, indent=2))
    print(f"Saved: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRACE STATISTICS")
    print("=" * 60)
    print(f"Total traces: {stats['metadata']['total_traces']}")

    print(f"\nChannel Rates:")
    for ch, rate in stats["channel_rates"].items():
        dominant = stats.get("channel_dominant_attack", {}).get(ch, {})
        family = dominant.get("family", "?")
        print(f"  {ch}: {rate}% (dominant: {family})")

    print(f"\nLeak Rates:")
    print(f"  Single-agent: {stats['leak_rates']['single_agent']}%")
    print(f"  Multi-agent: {stats['leak_rates']['multi_agent']}%")

    print(f"\nH1 Validation (Audit Gap):")
    h1 = stats["h1_validation"]
    print(f"  Detected (C1 leaked): {h1['detected_rate']}%")
    print(f"  Missed (H1 true): {h1['missed_rate']}%")
    print(f"  95% CI: {h1['ci_95']}")

    print(f"\nBy Vertical:")
    for v, data in stats["by_vertical"].items():
        print(f"  {v}: C1={data['c1_rate']}%, C2={data['c2_rate']}%, C5={data['c5_rate']}%")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
