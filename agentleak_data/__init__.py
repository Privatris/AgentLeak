"""
AgentLeak Data Module - Paths and Constants

Usage:
    from agentleak_data import SCENARIOS_FULL, SCENARIOS_DIFFICULT
    
    with jsonlines.open(SCENARIOS_FULL) as f:
        scenarios = list(f)
"""

from pathlib import Path

# Base directory
DATA_DIR = Path(__file__).parent

# ============================================================================
# DATASETS - Benchmark data
# ============================================================================

# Full scenarios (1000)
SCENARIOS_FULL = DATA_DIR / 'datasets' / 'scenarios_full_1000.jsonl'

# Difficult scenarios (100)
SCENARIOS_DIFFICULT = DATA_DIR / 'datasets' / 'scenarios_difficult_100.jsonl'

# Internal attack traces
TRACES_INTERNAL = DATA_DIR / 'datasets' / 'traces_internal_channels.jsonl'

# ============================================================================
# EXAMPLES - Example files
# ============================================================================

EXAMPLE_SCENARIO = DATA_DIR / 'examples' / 'scenario_example.jsonl'
EXAMPLE_TRACES = DATA_DIR / 'examples' / 'trace_sample.jsonl'

# ============================================================================
# PROMPTS - Configurations and prompts
# ============================================================================

PROMPT_LLM_JUDGE = DATA_DIR / 'prompts' / 'llm_judge_prompt.txt'

# ============================================================================
# UTILS
# ============================================================================

def get_dataset_stats():
    """Return dataset statistics."""
    import jsonlines
    
    stats = {}
    
    # Count full scenarios
    if SCENARIOS_FULL.exists():
        with jsonlines.open(SCENARIOS_FULL) as f:
            full = list(f)
            stats['scenarios_full'] = len(full)
            stats['verticals'] = len(set(s['vertical'] for s in full))
            stats['difficulties'] = list(set(s['difficulty'] for s in full))
    
    # Count difficult scenarios
    if SCENARIOS_DIFFICULT.exists():
        with jsonlines.open(SCENARIOS_DIFFICULT) as f:
            stats['scenarios_difficult'] = len(list(f))
    
    return stats

def print_info():
    """Display information about data."""
    print("\n" + "="*60)
    print("AgentLeak Benchmark Data")
    print("="*60)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Exists: {DATA_DIR.exists()}")
    
    print("\nDatasets:")
    print(f"  - Scenarios Full (1000): {SCENARIOS_FULL.exists()}")
    print(f"  - Scenarios Difficult (100): {SCENARIOS_DIFFICULT.exists()}")
    print(f"  - Traces Internal: {TRACES_INTERNAL.exists()}")
    
    print("\nExamples:")
    print(f"  - Example Scenario: {EXAMPLE_SCENARIO.exists()}")
    print(f"  - Example Traces: {EXAMPLE_TRACES.exists()}")
    
    print("\nPrompts:")
    print(f"  - LLM Judge Prompt: {PROMPT_LLM_JUDGE.exists()}")
    
    try:
        stats = get_dataset_stats()
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
    except Exception as e:
        print(f"\nStatistics: Error - {e}")
    
    print()

if __name__ == '__main__':
    print_info()
