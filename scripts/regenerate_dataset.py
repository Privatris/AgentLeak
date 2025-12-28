#!/usr/bin/env python3
"""
Regenerate AgentLeak-1000 dataset with proper distribution.

Paper claims:
- 4 verticals: 250 each
- 40% single-agent, 60% multi-agent
- 50% benign (A0), 25% weak (A1), 25% strong (A2)
- All 32 attack classes represented (6 families)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentleak.generators.scenario_generator import ScenarioGenerator
from agentleak.schemas.scenario import Vertical, AdversaryLevel


def main():
    gen = ScenarioGenerator(seed=2024)
    
    os.makedirs('agentleak_data', exist_ok=True)
    
    scenarios = []
    for v in Vertical:
        for i in range(250):
            # Distribution adversaire: 50% benign, 25% A1, 25% A2
            r = i % 4
            if r < 2:
                adv = AdversaryLevel.A0_BENIGN
            elif r == 2:
                adv = AdversaryLevel.A1_WEAK
            else:
                adv = AdversaryLevel.A2_STRONG
            
            # 60% multi-agent
            multi = (i % 5) >= 2
            
            s = gen.generate(
                vertical=v,
                adversary_level=adv,
                multi_agent=multi,
            )
            scenarios.append(s)
    
    # Sauvegarder
    with open('agentleak_data/agentleak_1000_v2.jsonl', 'w') as f:
        for s in scenarios:
            f.write(s.model_dump_json() + '\n')
    
    print(f'Generated {len(scenarios)} scenarios')
    
    # Vérifier distribution
    verticals = {}
    attacks = {}
    single_agent = 0
    multi_agent = 0
    benign = 0
    
    for s in scenarios:
        verticals[s.vertical.value] = verticals.get(s.vertical.value, 0) + 1
        if len(s.agents) <= 1:
            single_agent += 1
        else:
            multi_agent += 1
        if s.attack.enabled:
            ac = s.attack.attack_class.value if s.attack.attack_class else "none"
            attacks[ac] = attacks.get(ac, 0) + 1
        else:
            benign += 1
    
    print(f'\n=== Distribution ===')
    print(f'Verticals: {verticals}')
    print(f'Single-agent: {single_agent} ({100*single_agent/len(scenarios):.0f}%)')
    print(f'Multi-agent: {multi_agent} ({100*multi_agent/len(scenarios):.0f}%)')
    print(f'Benign (A0): {benign} ({100*benign/len(scenarios):.0f}%)')
    print(f'Adversarial: {len(scenarios) - benign} ({100*(len(scenarios)-benign)/len(scenarios):.0f}%)')
    print(f'Attack classes: {attacks}')


if __name__ == "__main__":
    main()
