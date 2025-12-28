#!/usr/bin/env python3
"""Generate cross-agent attack scenarios."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentleak.generators.cross_agent_scenarios import (
    generate_all_cross_agent_scenarios,
    generate_cross_agent_scenarios_jsonl,
    COLLUSION_ATTACKS,
    DELEGATION_ATTACKS,
    ROLE_BOUNDARY_ATTACKS,
    IPI_CROSSAGENT_ATTACKS,
)

print('=== Cross-Agent Attack Templates ===')
print(f'Collusion attacks: {len(COLLUSION_ATTACKS)}')
print(f'Delegation attacks: {len(DELEGATION_ATTACKS)}')
print(f'Role boundary attacks: {len(ROLE_BOUNDARY_ATTACKS)}')
print(f'IPI cross-agent attacks: {len(IPI_CROSSAGENT_ATTACKS)}')
total = len(COLLUSION_ATTACKS) + len(DELEGATION_ATTACKS) + len(ROLE_BOUNDARY_ATTACKS) + len(IPI_CROSSAGENT_ATTACKS)
print(f'Total templates: {total}')

print('\n=== Generating Scenarios ===')
scenarios = generate_all_cross_agent_scenarios(num_per_template=1)
print(f'Generated {len(scenarios)} scenarios')

# Show sample
scenario, gt = scenarios[0]
print(f'\nSample scenario: {scenario.scenario_id}')
print(f'  Vertical: {scenario.vertical}')
print(f'  Agents: {len(scenario.agents)}')
print(f'  Attack class: {scenario.attack.attack_class}')
print(f'  Forbidden flows: {len(gt.forbidden_flows)}')
print(f'  Protected patterns: {len(gt.protected_data_patterns)}')

# Save to file
output_path = 'data/scenarios/cross_agent_attacks.jsonl'
os.makedirs('data/scenarios', exist_ok=True)
count = generate_cross_agent_scenarios_jsonl(output_path, num_per_template=2)
print(f'\nSaved {count} scenarios to {output_path}')
