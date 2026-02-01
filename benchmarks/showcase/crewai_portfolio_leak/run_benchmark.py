#!/usr/bin/env python3
"""
AgentLeak Showcase Benchmark Runner
===================================

Runs the CrewAI Portfolio Leak scenario across multiple LLMs to compare
privacy leakage rates for the IEEE paper.

Usage:
  python run_benchmark.py [--fresh]
  
Options:
  --fresh    Clear previous results and re-run all models

Models tested (via OpenRouter):
  - GPT-4o
  - GPT-4o-mini  
  - Mistral Large
  - Llama 3.3 70B
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.showcase.crewai_portfolio_leak.crew import StockAnalysisCrew
from benchmarks.showcase.crewai_portfolio_leak.utils.db_manager import DBManager
from benchmarks.showcase.crewai_portfolio_leak.utils.monitor import SDKChannelMonitor, DetectionMode
from benchmarks.showcase.crewai_portfolio_leak.utils.logger import setup_logging

logger = setup_logging("BenchmarkRunner")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"
TRACES_DIR = RESULTS_DIR / "showcase_traces"

# Models to benchmark via OpenRouter
MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "mistralai/mistral-large-2411",  # OpenRouter model name
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-3.5-sonnet",
]


def load_results() -> Dict[str, Any]:
    """Load existing results from file."""
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_results(results: Dict[str, Any]):
    """Save results to file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))


def run_single_model(model: str, user_id: str = "user_001", stock: str = "AAPL") -> Dict[str, Any]:
    """Run benchmark for a single model."""
    logger.info(f"🚀 Starting benchmark: {model}")
    
    db = DBManager()
    user_data = db.get_user(user_id)
    
    monitor = SDKChannelMonitor(vault=user_data, mode=DetectionMode.HYBRID)
    
    start = time.time()
    try:
        crew = StockAnalysisCrew(
            stock_symbol=stock,
            user_id=user_id,
            monitor=monitor,
            model_name=model
        )
        crew.run()
        elapsed = time.time() - start
        
        stats = monitor.get_stats()
        
        # Save individual trace for verification
        TRACES_DIR.mkdir(exist_ok=True, parents=True)
        trace_file = TRACES_DIR / f"trace_{model.split('/')[-1]}.json"
        
        trace_data = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "execution_time_s": round(elapsed, 1),
            "stats": stats
        }
        trace_file.write_text(json.dumps(trace_data, indent=2))
        logger.info(f"📄 Trace saved to {trace_file}")
        
        # Extract key leak info
        c3_leaks = [d for d in stats["detections"] if d["channel"] == "C3"]
        leaked_pii = set()
        for d in stats["detections"]:
            for item in d.get("leaked_items", []):
                if "CREDIT_CARD" in item or "IBAN" in item or "VAULT_MATCH" in item:
                    leaked_pii.add(item.split(":")[0])
        
        return {
            "model": model,
            "status": "success",
            "execution_time_s": round(elapsed, 1),
            "total_leaks": stats["total_leaks"],
            "c3_tool_leaks": len(c3_leaks),
            "leaked_pii_types": list(leaked_pii),
            "leaks_by_channel": stats["by_channel"],
            "detection_tier": stats["detections"][0]["tier"] if stats["detections"] else "none",
        }
        
    except Exception as e:
        logger.error(f"❌ Failed: {model} - {e}")
        return {
            "model": model,
            "status": "failed",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="AgentLeak Showcase Benchmark")
    parser.add_argument("--fresh", action="store_true", help="Clear and re-run all")
    args = parser.parse_args()
    
    results = {} if args.fresh else load_results()
    
    print("\n" + "="*60)
    print("🔬 AGENTLEAK SHOWCASE BENCHMARK")
    print("="*60)
    print(f"Models: {len(MODELS)}")
    print(f"Scenario: CrewAI Portfolio Analysis (C3 Tool Leak)")
    print("="*60 + "\n")
    
    for model in MODELS:
        if model in results and results[model].get("status") == "success":
            print(f"✓ {model} - Cached (Leaks: {results[model]['total_leaks']})")
            continue
            
        print(f"► Testing {model}...")
        result = run_single_model(model)
        results[model] = result
        save_results(results)
        
        if result["status"] == "success":
            print(f"  ✅ Done in {result['execution_time_s']}s | Leaks: {result['total_leaks']} | C3: {result['c3_tool_leaks']}")
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown')[:50]}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    successful = [r for r in results.values() if r.get("status") == "success"]
    if successful:
        print(f"\n{'Model':<35} {'Leaks':>6} {'C3':>4} {'Time':>8}")
        print("-"*55)
        for r in successful:
            print(f"{r['model']:<35} {r['total_leaks']:>6} {r['c3_tool_leaks']:>4} {r['execution_time_s']:>7}s")
        
        avg_leaks = sum(r['total_leaks'] for r in successful) / len(successful)
        c3_rate = sum(1 for r in successful if r['c3_tool_leaks'] > 0) / len(successful) * 100
        print("-"*55)
        print(f"{'Average':<35} {avg_leaks:>6.1f}")
        print(f"{'C3 Leak Rate':<35} {c3_rate:>5.0f}%")
    
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
