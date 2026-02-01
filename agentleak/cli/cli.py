#!/usr/bin/env python3
"""
AgentLeak CLI - Command-line interface for privacy benchmarking.

Usage:
    agentleak list models         List OpenRouter models
    agentleak list attacks        Attack families F1-F6
    agentleak run -n 10           Quick benchmark
    agentleak check "text"        Check for PII in text
"""

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Logo ASCII
LOGO = r"""
    ___                    __  __               __
   /   | ____ ____  ____  / /_/ /   ___  ____ _/ /__
  / /| |/ __ `/ _ \/ __ \/ __/ /   / _ \/ __ `/ //_/
 / ___ / /_/ /  __/ / / / /_/ /___/  __/ /_/ / ,<
/_/  |_\__, /\___/_/ /_/\__/_____/\___/\__,_/_/|_|
      /____/
"""

# OpenRouter models
MODELS = {
    # Frontier
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "gemini-2.0-flash": "google/gemini-2.0-flash-exp",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    # Production
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "mistral-large": "mistralai/mistral-large-2411",
    # Open Source
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "deepseek-v3": "deepseek/deepseek-chat",
}

ATTACKS = {
    "F1": "Prompt Injection",
    "F2": "Context Manipulation",
    "F3": "Tool Misuse",
    "F4": "Multi-Agent Coordination",
    "F5": "Memory Exploitation",
    "F6": "Inference Attacks",
}

CHANNELS = {
    "C1": "Final Output (external)",
    "C2": "Inter-Agent Messages",
    "C3": "Tool Inputs",
    "C4": "Tool Outputs",
    "C5": "Memory/Context",
    "C6": "Logs/Debug",
    "C7": "Artifacts/Files",
}

ADVERSARIES = {"A0": "Benign", "A1": "Weak", "A2": "Strong"}
FRAMEWORKS = ["langchain", "crewai", "autogpt", "metagpt"]


def banner(minimal=False):
    """Display the AgentLeak logo."""
    if minimal:
        print("\n🔒 AgentLeak - Privacy Benchmark for Multi-Agent LLM Systems\n")
    else:
        print(LOGO)
        print("  Privacy Benchmark for Multi-Agent LLM Systems")
        print("  " + "─" * 50 + "\n")


def cmd_list(args):
    """List available options."""
    banner(minimal=True)
    
    what = args.what
    
    if what in ("models", "all"):
        print("📦 OPENROUTER MODELS")
        print("─" * 40)
        tiers = {
            "Frontier": ["gpt-4o", "gpt-4-turbo", "claude-3.5-sonnet", "claude-3-opus", "gemini-2.0-flash", "gemini-1.5-pro"],
            "Production": ["gpt-4o-mini", "claude-3-haiku", "mistral-large"],
            "Open Source": ["llama-3.3-70b", "qwen-2.5-72b", "deepseek-v3"],
        }
        for tier, names in tiers.items():
            print(f"\n  {tier}:")
            for name in names:
                print(f"    {name:20} → {MODELS[name]}")
        print()
    
    if what in ("attacks", "all"):
        print("⚔️  ATTACK FAMILIES")
        print("─" * 40)
        for code, desc in ATTACKS.items():
            print(f"  {code}: {desc}")
        print("\n  💡 F4 (Multi-Agent) = 80% success rate\n")
    
    if what in ("channels", "all"):
        print("📡 LEAKAGE CHANNELS")
        print("─" * 40)
        for code, desc in CHANNELS.items():
            print(f"  {code}: {desc}")
        print("\n  💡 C2/C5 (internal) = 8.3× more leaks than C1\n")
    
    if what in ("adversaries", "all"):
        print("🎭 ADVERSARY LEVELS")
        print("─" * 40)
        for code, desc in ADVERSARIES.items():
            print(f"  {code}: {desc}")
        print()


def cmd_run(args):
    """Launch the benchmark."""
    banner()
    
    # Validate models
    models = [m.strip() for m in args.models.split(",")]
    if "all" in models:
        models = list(MODELS.keys())
    else:
        for m in models:
            if m not in MODELS:
                print(f"❌ Unknown model: {m}")
                print(f"   → agentleak list models")
                return 1
    
    # Validate attacks
    attacks = [a.strip() for a in args.attacks.split(",")]
    if "all" in attacks:
        attacks = list(ATTACKS.keys())
    
    # Validate channels
    channels = [c.strip() for c in args.channels.split(",")]
    if "all" in channels:
        channels = list(CHANNELS.keys())
    
    print("🎯 CONFIGURATION")
    print("─" * 40)
    print(f"  Scenarios:   {args.scenarios}")
    print(f"  Models:      {', '.join(models)}")
    print(f"  Attacks:     {', '.join(attacks)}")
    print(f"  Channels:    {', '.join(channels)}")
    print(f"  Adversary:   {args.adversary}")
    print(f"  LLM Judge:   {'✓' if args.llm_judge else '✗'}")
    print(f"  API Delay:   {args.delay}s")
    print()
    
    if args.dry_run:
        print("🔍 Dry-run mode - no execution\n")
        return 0
    
    # Import and execute
    try:
        from agentleak.runner import TestRunner
        from agentleak.config import Config
        
        # Build configuration
        config = Config()
        config.benchmark.n_scenarios = args.scenarios
        config.benchmark.scenarios_per_vertical = args.scenarios // 4
        config.benchmark.attack_distribution = {"A0": 0.5, "A1": 0.25, "A2": 0.25} if args.adversary == "A2" else {"A0": 1.0}
        config.benchmark.api_delay = args.delay
        config.detection.enable_llm_judge = args.llm_judge
        
        # Initialize runner
        runner = TestRunner(config)
        
        print("\n🚀 STARTING BENCHMARK via agentleak.runner")
        if len(models) > 1:
            runner.run_comparison(models=models, n_scenarios=args.scenarios)
        else:
            # Update single model config
            config.model.name = models[0]
            # runner is already init with config, but we need to update adapter if model changed?
            # run_comparison handles model switching. For run(), we probably need to set it.
            # But TestRunner init creates adapter based on config.model.name. 
            # So better to recreate runner or modify config before init 
            # (which I did, but models[0] was set after init!).
            
            # Correct approach:
            config.model.name = models[0]
            runner = TestRunner(config) # Re-init to pick up model
            
            runner.run(n_scenarios=args.scenarios)
            
    except ImportError as e:
        print(f"❌ Error: {e}")
        # Fallback to experiments if available (legacy)
        try:
            experiments_dir = PROJECT_ROOT / "experiments" / "all_to_all"
            if experiments_dir.exists():
                sys.path.insert(0, str(experiments_dir))
                from run import BenchmarkRunner
                print("⚠️  Fallback to legacy experiments runner")
                # ... legacy code ...
        except ImportError:
            return 1
        return 1
    except Exception as e:
         print(f"❌ Execution failed: {e}")
         return 1
    
    return 0
    
    return 0


def cmd_check(args):
    """Check text for PII leaks."""
    banner(minimal=True)
    
    text = args.text
    print(f"Scanning: \"{text[:60]}{'...' if len(text) > 60 else ''}\"\n")
    
    patterns = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "Credit Card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        "Email": r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b',
        "Phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    }
    
    found = []
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found.append((name, matches))
            print(f"  ⚠️  {name}: {matches}")
    
    if found:
        print("\n🔴 LEAKS DETECTED")
        return 1
    else:
        print("  ✅ No leaks detected")
        return 0


def cmd_info(args):
    """Display project information."""
    banner()
    
    print("📊 IEEE PAPER CLAIMS")
    print("─" * 40)
    print("  1. Multi-agent = 2.3× more leaks")
    print("  2. F4 (coordination) = 80% success")
    print("  3. C2/C5 (internal) = 8.3× vs C1")
    print()
    
    print("🔧 COMMANDS")
    print("─" * 40)
    print("  agentleak list models     OpenRouter models")
    print("  agentleak list attacks    Attacks F1-F6")
    print("  agentleak list channels   Channels C1-C7")
    print("  agentleak run             Benchmark")
    print("  agentleak check           Check for PII")
    print()
    
    print("📖 EXEMPLES")
    print("─" * 40)
    print("  agentleak run -n 10 -m gpt-4o")
    print("  agentleak run -m gpt-4o,claude-3.5-sonnet -n 50 --llm-judge")
    print("  agentleak check \"SSN: 123-45-6789\"")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agentleak",
        description="Privacy Benchmark for Multi-Agent LLM Systems",
    )
    subs = parser.add_subparsers(dest="cmd")
    
    # list
    p_list = subs.add_parser("list", help="List options")
    p_list.add_argument("what", choices=["models", "attacks", "channels", "adversaries", "all"])
    
    # run
    p_run = subs.add_parser("run", help="Run benchmark")
    p_run.add_argument("-n", "--scenarios", type=int, default=10)
    p_run.add_argument("-m", "--models", default="gpt-4o-mini")
    p_run.add_argument("-a", "--attacks", default="F1,F4")
    p_run.add_argument("-c", "--channels", default="C1,C2")
    p_run.add_argument("--adversary", default="A1", choices=["A0", "A1", "A2"])
    p_run.add_argument("--llm-judge", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--delay", type=float, default=0.5)
    
    # check
    p_check = subs.add_parser("check", help="Check for PII")
    p_check.add_argument("text")
    
    # info
    subs.add_parser("info", help="Information")
    
    args = parser.parse_args()
    
    if args.cmd == "list":
        return cmd_list(args)
    elif args.cmd == "run":
        return cmd_run(args)
    elif args.cmd == "check":
        return cmd_check(args)
    elif args.cmd == "info":
        return cmd_info(args)
    else:
        return cmd_info(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
