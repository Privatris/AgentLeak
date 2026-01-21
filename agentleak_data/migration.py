"""
Migration helper - Backward compatibility with old structure

This file helps transition to the new agentleak_data/ structure.
"""

from pathlib import Path

def get_legacy_path(old_name: str) -> Path:
    """
    Return the new path for an old filename.
    
    Usage:
        old_path = 'agentleak_1000.jsonl'
        new_path = get_legacy_path(old_path)
        # Returns: agentleak_data/datasets/scenarios_full_1000.jsonl
    """
    
    mapping = {
        'agentleak_1000.jsonl': 'datasets/scenarios_full_1000.jsonl',
        'agentleak_1000_backup.jsonl': 'datasets/scenarios_full_1000.jsonl',
        'difficult_scenarios_100.jsonl': 'datasets/scenarios_difficult_100.jsonl',
        'difficult_scenarios_50.jsonl': 'datasets/scenarios_difficult_100.jsonl',  # Redirect to 100
        'internal_channel_attack_traces.jsonl': 'datasets/traces_internal_channels.jsonl',
        'scenario_example.jsonl': 'examples/scenario_example.jsonl',
        'trace_sample.jsonl': 'examples/trace_sample.jsonl',
        'llm_judge_prompt.txt': 'prompts/llm_judge_prompt.txt',
    }
    
    data_dir = Path(__file__).parent
    new_name = mapping.get(old_name)
    
    if new_name:
        return data_dir / new_name
    else:
        raise ValueError(f"Unknown legacy filename: {old_name}")

def load_with_compatibility(file_path):
    """
    Load a file with support for old structure.
    
    Usage:
        scenarios = load_with_compatibility('agentleak_1000.jsonl')
        # Automatically loads from the new path
    """
    import jsonlines
    
    if not file_path.is_absolute():
        file_path = Path(file_path)
        if not file_path.exists() and not file_path.is_absolute():
            # Try to find via legacy mapping
            try:
                file_path = get_legacy_path(file_path.name)
            except ValueError:
                pass
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with jsonlines.open(file_path) as f:
        return list(f)

# Display migration notes
MIGRATION_NOTES = """
╔══════════════════════════════════════════════════════════════════════╗
║         AgentLeak Data Migration - Old → New Structure                ║
╚══════════════════════════════════════════════════════════════════════╝

OLD NAMES → NEW NAMES:
┌────────────────────────────────────┬──────────────────────────────────┐
│ Old                                │ New                              │
├────────────────────────────────────┼──────────────────────────────────┤
│ agentleak_1000.jsonl               │ datasets/scenarios_full_1000    │
│ difficult_scenarios_100.jsonl       │ datasets/scenarios_difficult_100│
│ difficult_scenarios_50.jsonl        │ ❌ DELETED (empty)              │
│ internal_channel_attack_traces.jsonl│ datasets/traces_internal_...   │
│ scenario_example.jsonl              │ examples/scenario_example.jsonl │
│ trace_sample.jsonl                  │ examples/trace_sample.jsonl     │
│ llm_judge_prompt.txt                │ prompts/llm_judge_prompt.txt    │
│ agentleak_1000_summary.json         │ ❌ ARCHIVED (.archive/)         │
└────────────────────────────────────┴──────────────────────────────────┘

HOW TO UPDATE CODE:

Option 1 - Use constants (RECOMMENDED):
    from agentleak_data import SCENARIOS_FULL, SCENARIOS_DIFFICULT
    
    with jsonlines.open(SCENARIOS_FULL) as f:
        scenarios = list(f)

Option 2 - Use migration helper:
    from agentleak_data.migration import load_with_compatibility
    
    # Works with old names
    scenarios = load_with_compatibility('agentleak_1000.jsonl')

Option 3 - Direct paths:
    from agentleak_data import DATA_DIR
    
    path = DATA_DIR / 'datasets' / 'scenarios_full_1000.jsonl'
    with jsonlines.open(path) as f:
        scenarios = list(f)

CODE UPDATE STATUS:
- tests/              ⏳ To update
- scripts/            ⏳ To update
- agentleak/          ⏳ To update
"""

if __name__ == '__main__':
    print(MIGRATION_NOTES)
