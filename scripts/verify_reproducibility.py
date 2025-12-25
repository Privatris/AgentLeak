#!/usr/bin/env python3
"""
AgentLeak Reproducibility Verifier
==================================

This script proves that AgentLeak is a deterministic and reliable benchmark.
It runs the evaluation twice on the same subset of scenarios and asserts
that the results are identical (ignoring timestamps).

Usage:
    python scripts/verify_reproducibility.py
"""

import sys
import json
import subprocess
from pathlib import Path
import shutil

def run_benchmark(output_dir):
    """Runs the benchmark script in quick mode."""
    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--quick",
        "--output-format", "json",
        "--output-dir", output_dir
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def load_results(output_dir):
    """Loads the JSON results."""
    path = Path(output_dir) / "benchmark_results.json"
    with open(path) as f:
        return json.load(f)

def clean_results(data):
    """Removes non-deterministic fields (timestamps, durations)."""
    if "timestamp" in data:
        del data["timestamp"]
    
    for res in data.get("framework_results", []):
        if "total_time" in res:
            del res["total_time"]
            
    return data

def run_verification():
    print("🔍 Starting AgentLeak Reproducibility Verification...")
    
    dir1 = "verify_run_1"
    dir2 = "verify_run_2"
    
    # Cleanup
    shutil.rmtree(dir1, ignore_errors=True)
    shutil.rmtree(dir2, ignore_errors=True)
    
    try:
        # Run 1
        print(f"\n[Run 1] Executing benchmark...")
        run_benchmark(dir1)
        results1 = load_results(dir1)
        
        # Run 2
        print(f"\n[Run 2] Executing benchmark (Repeat)...")
        run_benchmark(dir2)
        results2 = load_results(dir2)
        
        # Compare
        clean1 = clean_results(results1)
        clean2 = clean_results(results2)
        
        print("\n[Comparison]")
        # Convert to string for easy diffing if needed, or just equality check
        json1 = json.dumps(clean1, sort_keys=True)
        json2 = json.dumps(clean2, sort_keys=True)
        
        if json1 == json2:
            print("✅ SUCCESS: Results are identical.")
            print("   AgentLeak is deterministic and verifiable.")
            return True
        else:
            print("❌ FAILURE: Results differ.")
            print("   The benchmark is not reproducible.")
            return False
            
    finally:
        # Cleanup
        shutil.rmtree(dir1, ignore_errors=True)
        shutil.rmtree(dir2, ignore_errors=True)

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
