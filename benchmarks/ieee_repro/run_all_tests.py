#!/usr/bin/env python3
"""
AgentLeak Full IEEE Benchmark Suite
===================================

Orchestrates all benchmark tests for the IEEE paper reproduction.

Tests Included:
    1. Proof-of-Concept Attack (Section VII-B)
    2. Framework Comparison (Experiment IV)
    3. Defense Evaluation (Table V)
    4. Tier Analysis (Section VI-H)
    5. Pareto Frontier (Figure 5)
    6. Multi-Model Evaluation (Table VII) [optional]

Usage:
    # Quick run (essential tests only)
    python benchmarks/ieee_repro/run_all_tests.py --quick
    
    # Full run (all tests)
    python benchmarks/ieee_repro/run_all_tests.py --full
    
    # Specific tests
    python benchmarks/ieee_repro/run_all_tests.py --tests poc,frameworks,defenses
"""

import os
import sys
import json
import argparse
import subprocess
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).parent.parent.parent
BENCHMARK_DIR = ROOT / "benchmarks/ieee_repro"
RESULTS_DIR = BENCHMARK_DIR / "results"
VENV_PYTHON = BENCHMARK_DIR / ".venv_new/bin/python"

# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

TESTS = {
    "benchmark": {
        "name": "Core Benchmark (14 Claims)",
        "script": "benchmarks/ieee_repro/benchmark.py",
        "args": ["--n", "20"],
        "description": "Main benchmark - all 14 paper claims",
        "essential": True,
    },
    "frameworks": {
        "name": "Framework Comparison",
        "script": "benchmarks/ieee_repro/run_frameworks.py",
        "args": [],
        "description": "Table IV - LangChain/CrewAI/MetaGPT/AutoGPT",
        "essential": True,
    },
    "defenses": {
        "name": "Defense Evaluation",
        "script": "benchmarks/ieee_repro/test_defenses.py",
        "args": ["--n", "10"],
        "description": "Table V - 7 defense mechanisms",
        "essential": True,
    },
    "tiers": {
        "name": "Tier Analysis",
        "script": "benchmarks/ieee_repro/test_tier_analysis.py",
        "args": ["--n", "20"],
        "description": "Section VI-H - Canary/PII/Semantic breakdown",
        "essential": False,
    },
    "pareto": {
        "name": "Pareto Frontier",
        "script": "benchmarks/ieee_repro/test_pareto.py",
        "args": ["--n", "10"],
        "description": "Figure 5 - Security-utility tradeoff",
        "essential": False,
    },
    "multimodel": {
        "name": "Multi-Model Evaluation",
        "script": "benchmarks/ieee_repro/test_multi_model.py",
        "args": ["--models", "gpt-4o-mini,llama-3.3-70b", "--n", "10"],
        "description": "Table VII - Model comparison",
        "essential": False,
    },
}


# =============================================================================
# RUNNER
# =============================================================================

def run_test(test_key: str, config: Dict) -> Dict:
    """Run a single test."""
    print(f"\n{'='*70}")
    print(f"🧪 {config['name']}")
    print(f"   {config['description']}")
    print(f"{'='*70}\n")
    
    script_path = ROOT / config["script"]
    
    if not script_path.exists():
        print(f"   ⚠️ Script not found: {script_path}")
        return {"status": "skipped", "reason": "script_not_found"}
    
    # Build command
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    cmd = [python, str(script_path)] + config["args"]
    
    print(f"   Running: {' '.join(cmd[-3:])}")
    
    start = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=False,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        duration = (datetime.now() - start).total_seconds()
        
        if result.returncode == 0:
            print(f"\n   ✅ {config['name']} completed in {duration:.1f}s")
            return {"status": "success", "duration": duration}
        else:
            print(f"\n   ❌ {config['name']} failed (exit code {result.returncode})")
            return {"status": "failed", "exit_code": result.returncode}
            
    except subprocess.TimeoutExpired:
        print(f"\n   ⏱️ {config['name']} timed out")
        return {"status": "timeout"}
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        return {"status": "error", "message": str(e)}


def run_suite(tests_to_run: List[str], quick_mode: bool = False):
    """Run the test suite."""
    print("="*70)
    print("AGENTLEAK IEEE BENCHMARK SUITE")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Quick (essential only)' if quick_mode else 'Full'}")
    
    if quick_mode:
        tests_to_run = [k for k, v in TESTS.items() if v["essential"]]
    
    print(f"Tests: {', '.join(tests_to_run)}\n")
    
    results = {}
    
    for test_key in tests_to_run:
        if test_key not in TESTS:
            print(f"⚠️ Unknown test: {test_key}")
            continue
        
        config = TESTS[test_key]
        results[test_key] = run_test(test_key, config)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    skipped = sum(1 for r in results.values() if r["status"] in ["skipped", "timeout", "error"])
    
    print(f"\n   ✅ Passed:  {success}")
    print(f"   ❌ Failed:  {failed}")
    print(f"   ⚠️ Skipped: {skipped}")
    print(f"   ─────────────────")
    print(f"   Total:    {len(results)}")
    
    # List results
    print("\n   Details:")
    for test_key, result in results.items():
        status_icon = {"success": "✅", "failed": "❌", "skipped": "⚠️", "timeout": "⏱️", "error": "💥"}.get(result["status"], "?")
        duration = f" ({result.get('duration', 0):.1f}s)" if "duration" in result else ""
        print(f"   {status_icon} {TESTS[test_key]['name']}{duration}")
    
    # Save summary
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "quick" if quick_mode else "full",
        "results": results,
        "counts": {
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "total": len(results)
        }
    }
    
    with open(RESULTS_DIR / "suite_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {RESULTS_DIR}/suite_summary.json")
    
    # List generated artifacts
    print("\n📁 Generated Artifacts:")
    for f in RESULTS_DIR.glob("*.json"):
        print(f"   - {f.name}")
    for f in RESULTS_DIR.glob("*.tex"):
        print(f"   - {f.name}")
    for f in RESULTS_DIR.glob("*.pdf"):
        print(f"   - {f.name}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AgentLeak IEEE Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run essential tests only")
    parser.add_argument("--full", action="store_true", help="Run all tests")
    parser.add_argument("--tests", type=str, help="Comma-separated list of tests")
    parser.add_argument("--list", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Tests:")
        print("-"*60)
        for key, config in TESTS.items():
            essential = "★" if config["essential"] else " "
            print(f"  {essential} {key:<15} {config['name']}")
            print(f"                    {config['description']}")
        print("\n★ = Essential (included in --quick mode)")
        return
    
    if args.tests:
        tests_to_run = [t.strip() for t in args.tests.split(",")]
        run_suite(tests_to_run)
    elif args.quick:
        run_suite([], quick_mode=True)
    elif args.full:
        run_suite(list(TESTS.keys()))
    else:
        # Default: essential tests
        run_suite([], quick_mode=True)


if __name__ == "__main__":
    main()
