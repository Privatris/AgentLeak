#!/usr/bin/env python3
"""Diagnostic script to identify blocking issues in the benchmark."""

from master_benchmark import (
    ScenarioManager, MultiAgentSimulator, LLMClient, LeakDetector
)

def main():
    print("="*60)
    print("DIAGNOSTIC: AgentLeak Benchmark")
    print("="*60)
    
    # Test 1: Scenarios
    print("\n[1] Loading scenarios...")
    scenarios = ScenarioManager()
    s = scenarios.sample(1)[0]
    print(f"    ✓ Scenario: {s.scenario_id}")
    print(f"    ✓ Canaries: {s.canaries}")
    
    # Show vault content
    vault_samples = []
    for record in s.private_vault.get("records", []):
        for k, v in record.get("fields", {}).items():
            if v and len(str(v)) >= 5:
                vault_samples.append(f"{k}={str(v)[:25]}")
    print(f"    ✓ Vault samples: {vault_samples[:5]}")
    
    # Test 2: LLM Client
    print("\n[2] Testing LLM client...")
    client = LLMClient()
    result = client.query("openai/gpt-4o-mini", [{"role": "user", "content": "Say OK"}])
    print(f"    ✓ Success: {result['success']}")
    if not result["success"]:
        print(f"    ✗ Error: {result.get('error', 'Unknown')}")
        return
    print(f"    ✓ Response: {result.get('response', '')[:50]}")
    
    # Test 3: Simulator
    print("\n[3] Running multi-agent simulation...")
    simulator = MultiAgentSimulator(client, model="openai/gpt-4o-mini")
    traces = simulator.run_multi(s)
    
    print(f"    ✓ C1 length: {len(traces.get('C1', ''))}")
    print(f"    ✓ C2 messages: {len(traces.get('C2', []))}")
    print(f"    ✓ C3 tool calls: {len(traces.get('C3', []))}")
    print(f"    ✓ C4 logs: {len(traces.get('C4', []))}")
    print(f"    ✓ C5 memory: {len(traces.get('C5', []))}")
    
    if traces.get("C1"):
        print(f"\n    C1 preview:\n    {traces['C1'][:300]}...")
    
    if traces.get("C2"):
        c2_content = traces["C2"][0].get("content", "")[:200]
        print(f"\n    C2[0] preview:\n    {c2_content}...")
    
    # Test 4: Detection
    print("\n[4] Running leak detection...")
    detector = LeakDetector()
    result = detector.detect(s, traces)
    
    print(f"    ✓ Any leak: {result['any_leak']}")
    print(f"    ✓ C1 leaked: {result['C1']['leaked']} - matches: {result['C1'].get('matches', [])}")
    print(f"    ✓ C2 leaked: {result['C2']['leaked']} - matches: {result['C2'].get('matches', [])}")
    print(f"    ✓ Internal leak: {result['internal_leak']}")
    print(f"    ✓ External leak: {result['external_leak']}")
    
    # Test 5: Manual check
    print("\n[5] Manual vault value check...")
    c2_text = ""
    for msg in traces.get("C2", []):
        c2_text += str(msg.get("content", "")) + "\n"
    
    found_in_c2 = []
    for record in s.private_vault.get("records", []):
        for k, v in record.get("fields", {}).items():
            if v and len(str(v)) >= 5:
                v_str = str(v)
                if v_str in c2_text:
                    found_in_c2.append(f"{k}={v_str[:20]}")
    
    print(f"    Vault values found in C2: {found_in_c2[:10]}")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    
    if not traces.get("C1") and not traces.get("C2"):
        print("❌ PROBLEM: No traces generated - LLM calls may be failing silently")
    elif not result["any_leak"] and found_in_c2:
        print("❌ PROBLEM: Vault values in traces but not detected - detector logic issue")
    elif result["any_leak"]:
        print("✅ Detection working - leaks found")
    else:
        print("⚠️  No leaks found - may be normal or detection threshold too high")


if __name__ == "__main__":
    main()
