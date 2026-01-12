#!/usr/bin/env python3
"""
Analyze internal channel leakage (C2, C5, C6) to prove technical vulnerability.
"""
import json
import glob
from collections import defaultdict

def analyze_internal_channels():
    """Extract and analyze leakage rates for internal channels."""
    results_files = sorted(glob.glob('benchmark_results/benchmark_*.json'))
    
    if not results_files:
        print("No benchmark results found")
        return
    
    latest = results_files[-1]
    print(f"Analyzing: {latest}\n")
    
    with open(latest) as f:
        data = json.load(f)
    
    # Define internal channels
    internal_channels = {
        'C2_INTER_AGENT': 'Inter-agent messages',
        'C5_MEMORY_WRITE': 'Shared memory',
        'C6_LOGS': 'System logs',
    }
    
    external_channels = {
        'C1_OUTPUT': 'Final output',
        'C3_TOOL_INPUT': 'Tool inputs',
        'C4_TOOL_OUTPUT': 'Tool outputs',
    }
    
    print("=" * 70)
    print("INTERNAL CHANNELS (accessible to any agent in the system)")
    print("=" * 70)
    
    internal_stats = {}
    if 'channels' in data:
        for channel in internal_channels:
            if channel in data['channels']:
                stats = data['channels'][channel]
                leakage_rate = stats.get('leakage_rate', 0)
                samples_leaked = stats.get('samples_with_leakage', 0)
                total = stats.get('total_samples', 0)
                internal_stats[channel] = leakage_rate
                print(f"\n{internal_channels[channel]} ({channel}):")
                print(f"  Leakage rate: {leakage_rate:.1%}")
                print(f"  Samples: {samples_leaked}/{total}")
    
    print("\n" + "=" * 70)
    print("EXTERNAL CHANNELS (only accessible to end user)")
    print("=" * 70)
    
    external_stats = {}
    if 'channels' in data:
        for channel in external_channels:
            if channel in data['channels']:
                stats = data['channels'][channel]
                leakage_rate = stats.get('leakage_rate', 0)
                samples_leaked = stats.get('samples_with_leakage', 0)
                total = stats.get('total_samples', 0)
                external_stats[channel] = leakage_rate
                print(f"\n{external_channels[channel]} ({channel}):")
                print(f"  Leakage rate: {leakage_rate:.1%}")
                print(f"  Samples: {samples_leaked}/{total}")
    
    # Calculate impact
    internal_avg = sum(internal_stats.values()) / len(internal_stats) if internal_stats else 0
    external_avg = sum(external_stats.values()) / len(external_stats) if external_stats else 0
    
    print("\n" + "=" * 70)
    print("KEY FINDING: INTERNAL THREAT MODEL")
    print("=" * 70)
    print(f"\nAverage internal channel leakage:  {internal_avg:.1%}")
    print(f"Average external channel leakage: {external_avg:.1%}")
    print(f"\nRisk multiplier: {internal_avg / external_avg:.1f}x more PII exposed internally")
    
    # Framework breakdown
    print("\n" + "=" * 70)
    print("FRAMEWORK COMPARISON (CrewAI most studied)")
    print("=" * 70)
    
    if 'frameworks' in data:
        for fw in ['CrewAI', 'AutoGPT', 'LangChain', 'MetaGPT']:
            if fw in data['frameworks']:
                fw_data = data['frameworks'][fw]
                if 'channels' in fw_data:
                    # Calculate internal leakage for this framework
                    internal_leaks = []
                    for ch in ['C2_INTER_AGENT', 'C5_MEMORY_WRITE', 'C6_LOGS']:
                        if ch in fw_data['channels']:
                            internal_leaks.append(fw_data['channels'][ch].get('leakage_rate', 0))
                    
                    if internal_leaks:
                        avg = sum(internal_leaks) / len(internal_leaks)
                        print(f"\n{fw}: {avg:.1%} average internal leakage")

if __name__ == '__main__':
    analyze_internal_channels()
