#!/usr/bin/env python3
"""
AGENTLEAK SHOWCASE - Single Run Demo
====================================

Run a single portfolio analysis to demonstrate the AgentLeak SDK integration.

Usage:
  python showcase.py --stock AAPL --user user_001
"""

import sys
import argparse
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from benchmarks.showcase.crewai_portfolio_leak.crew import StockAnalysisCrew
from benchmarks.showcase.crewai_portfolio_leak.utils.db_manager import DBManager
from benchmarks.showcase.crewai_portfolio_leak.utils.monitor import SDKChannelMonitor, DetectionMode
from benchmarks.showcase.crewai_portfolio_leak.utils.logger import setup_logging

logger = setup_logging("ShowcaseCLI")

def run_cli(stock: str, user_id: str):
    logger.info(f"Starting CLI showcase for {stock} (User: {user_id})")
    
    db = DBManager()
    user_data = db.get_user(user_id)
    if not user_data:
        logger.error(f"User {user_id} not found")
        return
        
    monitor = SDKChannelMonitor(vault=user_data, mode=DetectionMode.HYBRID)
    
    try:
        crew = StockAnalysisCrew(stock_symbol=stock, user_id=user_id, monitor=monitor)
        result = crew.run()
        
        print("\n" + "="*50)
        print("✅ ANALYSIS RESULT")
        print("="*50)
        print(result)
        
        print("\n" + "="*50)
        print("⚠️ LEAK DETECTOR REPORT")
        print("="*50)
        stats = monitor.get_stats()
        print(f"Total Leaks: {stats['total_leaks']}")
        for ch, count in stats['by_channel'].items():
            print(f"  - {ch}: {count} leaks")
            
        if stats['detections']:
            print("\nLatest detection:")
            print(stats['detections'][-1])
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentLeak CrewAI Showcase")
    parser.add_argument("--stock", default="AAPL", help="Stock symbol to analyze")
    parser.add_argument("--user", default="user_001", help="User ID from vault")
    
    args = parser.parse_args()
    run_cli(args.stock, args.user)
