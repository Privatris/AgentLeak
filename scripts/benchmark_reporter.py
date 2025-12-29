#!/usr/bin/env python3
"""
AgentLeak Benchmark Reporter

Generates professional reports, tables, and visualizations from benchmark results.

Outputs:
- LaTeX tables for paper inclusion
- Markdown reports for documentation
- CSV exports for analysis
- Comparison charts (if matplotlib available)

Usage:
    python scripts/benchmark_reporter.py benchmark_results/benchmark_20251226.json
    python scripts/benchmark_reporter.py --compare run1.json run2.json
    python scripts/benchmark_reporter.py --latex-only results.json

Author: AgentLeak Research Team
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_config import ALL_MODELS, Channel, AttackLevel


# =============================================================================
# LATEX TABLE GENERATORS
# =============================================================================


class LaTeXGenerator:
    """Generate LaTeX tables from benchmark results."""

    @staticmethod
    def model_comparison_table(results: Dict) -> str:
        """Generate main model comparison table (Table 5 style)."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Model Privacy Leakage Comparison across AgentLeak Benchmark}",
            r"\label{tab:model-comparison}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{TSR} $\uparrow$ & \textbf{ELR} $\downarrow$ & \textbf{WLS} $\downarrow$ & \textbf{Tokens} & \textbf{Cost} \\",
            r"\midrule",
        ]

        models = results.get("models", {})

        # Sort by ELR (ascending - lower is better)
        sorted_models = sorted(models.items(), key=lambda x: x[1].get("elr", 0))

        for model_name, model_data in sorted_models:
            tsr = model_data.get("tsr", 0) * 100
            elr = model_data.get("elr", 0) * 100
            wls = model_data.get("wls", 0)
            tokens = model_data.get("total_tokens", 0)
            cost = model_data.get("total_cost", 0)

            # Format model name for LaTeX
            display_name = model_name.replace("_", r"\_").replace("-", r"-")

            lines.append(
                f"{display_name} & {tsr:.1f}\\% & {elr:.1f}\\% & {wls:.2f} & "
                f"{tokens:,} & \\${cost:.2f} \\\\"
            )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def channel_breakdown_table(results: Dict) -> str:
        """Generate per-channel leakage table (Table 7 style)."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Leakage Rate by Channel (CLR)}",
            r"\label{tab:channel-breakdown}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{C1} & \textbf{C2} & \textbf{C3} & \textbf{C4} & \textbf{C5} & \textbf{C6} & \textbf{C7} \\",
            r" & \small{Output} & \small{Inter} & \small{Tool-In} & \small{Tool-Out} & \small{Memory} & \small{Logs} & \small{Artifacts} \\",
            r"\midrule",
        ]

        models = results.get("models", {})

        for model_name, model_data in models.items():
            clr = model_data.get("clr", {})
            display_name = model_name.replace("_", r"\_")

            values = [
                clr.get("C1_final_output", 0) * 100,
                clr.get("C2_inter_agent", 0) * 100,
                clr.get("C3_tool_input", 0) * 100,
                clr.get("C4_tool_output", 0) * 100,
                clr.get("C5_memory", 0) * 100,
                clr.get("C6_logs", 0) * 100,
                clr.get("C7_artifacts", 0) * 100,
            ]

            values_str = " & ".join(f"{v:.1f}\\%" for v in values)
            lines.append(f"{display_name} & {values_str} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def attack_comparison_table(results: Dict) -> str:
        """Generate attack success rate table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Attack Success Rate by Level (ASR)}",
            r"\label{tab:attack-comparison}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{A0 (Benign)} & \textbf{A1 (Indirect)} & \textbf{A2 (Adversarial)} \\",
            r"\midrule",
        ]

        models = results.get("models", {})

        for model_name, model_data in models.items():
            asr = model_data.get("asr", {})
            display_name = model_name.replace("_", r"\_")

            a0 = asr.get("A0", 0) * 100
            a1 = asr.get("A1", 0) * 100
            a2 = asr.get("A2", 0) * 100

            lines.append(f"{display_name} & {a0:.1f}\\% & {a1:.1f}\\% & {a2:.1f}\\% \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def pareto_table(results: Dict) -> str:
        """Generate Pareto frontier analysis table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Privacy-Utility Pareto Analysis}",
            r"\label{tab:pareto}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Model} & \textbf{TSR} & \textbf{1-ELR} & \textbf{Pareto Score} & \textbf{Pareto Optimal} \\",
            r"\midrule",
        ]

        models = results.get("models", {})

        # Calculate Pareto scores
        pareto_data = []
        for model_name, model_data in models.items():
            tsr = model_data.get("tsr", 0)
            elr = model_data.get("elr", 0)
            privacy = 1 - elr  # Higher is better
            pareto_score = (tsr + privacy) / 2  # Simple average
            pareto_data.append((model_name, tsr, privacy, pareto_score))

        # Determine Pareto optimal points
        pareto_optimal = set()
        for i, (name, tsr, priv, _) in enumerate(pareto_data):
            is_dominated = False
            for j, (_, other_tsr, other_priv, _) in enumerate(pareto_data):
                if (
                    i != j
                    and other_tsr >= tsr
                    and other_priv >= priv
                    and (other_tsr > tsr or other_priv > priv)
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_optimal.add(name)

        # Sort by Pareto score
        pareto_data.sort(key=lambda x: x[3], reverse=True)

        for model_name, tsr, privacy, score in pareto_data:
            display_name = model_name.replace("_", r"\_")
            is_optimal = r"\checkmark" if model_name in pareto_optimal else ""

            lines.append(
                f"{display_name} & {tsr*100:.1f}\\% & {privacy*100:.1f}\\% & "
                f"{score:.3f} & {is_optimal} \\\\"
            )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# MARKDOWN REPORT GENERATOR
# =============================================================================


class MarkdownGenerator:
    """Generate Markdown reports from benchmark results."""

    @staticmethod
    def full_report(results: Dict) -> str:
        """Generate comprehensive Markdown report."""
        timestamp = results.get("timestamp", "Unknown")
        name = results.get("name", "AgentLeak Benchmark")
        config = results.get("config", {})
        models = results.get("models", {})
        overall = results.get("overall_metrics", {})
        cost = results.get("cost_summary", {})
        runtime = results.get("runtime_seconds", 0)

        lines = [
            f"# {name}",
            "",
            f"**Timestamp:** {timestamp}  ",
            f"**Runtime:** {runtime:.1f} seconds  ",
            f"**Total Tokens:** {cost.get('total_tokens', 0):,}  ",
            "",
            "---",
            "",
            "## Configuration",
            "",
            f"- **Models:** {len(config.get('models', []))}",
            f"- **Frameworks:** {', '.join(config.get('frameworks', []))}",
            f"- **Attack Levels:** {', '.join(config.get('attack_levels', []))}",
            f"- **Channels:** {len(config.get('channels', []))}",
            "",
            "---",
            "",
            "## Overall Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Scenarios | {overall.get('n_scenarios', 0)} |",
            f"| Overall TSR | {overall.get('overall_tsr', 0)*100:.1f}% |",
            f"| Overall ELR | {overall.get('overall_elr', 0)*100:.1f}% |",
            f"| Average WLS | {overall.get('avg_wls', 0):.2f} |",
            "",
            "---",
            "",
            "## Model Comparison",
            "",
            "| Model | TSR ↑ | ELR ↓ | WLS ↓ | Tokens |",
            "|-------|-------|-------|-------|--------|",
        ]

        # Sort by ELR
        sorted_models = sorted(models.items(), key=lambda x: x[1].get("elr", 0))

        for model_name, model_data in sorted_models:
            tsr = model_data.get("tsr", 0) * 100
            elr = model_data.get("elr", 0) * 100
            wls = model_data.get("wls", 0)
            tokens = model_data.get("total_tokens", 0)

            lines.append(f"| {model_name} | {tsr:.1f}% | {elr:.1f}% | {wls:.2f} | {tokens:,} |")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Channel Leakage Breakdown",
                "",
                "| Model | C1 | C2 | C3 | C4 | C5 | C6 | C7 |",
                "|-------|-----|-----|-----|-----|-----|-----|-----|",
            ]
        )

        for model_name, model_data in sorted_models:
            clr = model_data.get("clr", {})
            values = [
                f"{clr.get(f'C{i}_' + ['final_output', 'inter_agent', 'tool_input', 'tool_output', 'memory', 'logs', 'artifacts'][i-1], 0)*100:.0f}%"
                for i in range(1, 8)
            ]
            lines.append(f"| {model_name} | {' | '.join(values)} |")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Attack Success Rate",
                "",
                "| Model | A0 (Benign) | A1 (Indirect) | A2 (Adversarial) |",
                "|-------|-------------|---------------|------------------|",
            ]
        )

        for model_name, model_data in sorted_models:
            asr = model_data.get("asr", {})
            a0 = asr.get("A0", 0) * 100
            a1 = asr.get("A1", 0) * 100
            a2 = asr.get("A2", 0) * 100
            lines.append(f"| {model_name} | {a0:.1f}% | {a1:.1f}% | {a2:.1f}% |")

        lines.extend(
            [
                "",
                "---",
                "",
                "## Key Findings",
                "",
                "### Best Privacy (Lowest ELR)",
                "",
            ]
        )

        if sorted_models:
            best_privacy = sorted_models[0]
            lines.append(
                f"**{best_privacy[0]}** with ELR = {best_privacy[1].get('elr', 0)*100:.1f}%"
            )

        lines.extend(
            [
                "",
                "### Best Utility (Highest TSR)",
                "",
            ]
        )

        best_utility = max(sorted_models, key=lambda x: x[1].get("tsr", 0))
        lines.append(f"**{best_utility[0]}** with TSR = {best_utility[1].get('tsr', 0)*100:.1f}%")

        lines.extend(
            [
                "",
                "### Most Vulnerable Channel",
                "",
            ]
        )

        # Find most vulnerable channel across all models
        channel_totals = {}
        for model_name, model_data in models.items():
            clr = model_data.get("clr", {})
            for ch, val in clr.items():
                channel_totals[ch] = channel_totals.get(ch, 0) + val

        if channel_totals:
            most_vuln = max(channel_totals.items(), key=lambda x: x[1])
            avg_rate = most_vuln[1] / len(models) * 100
            lines.append(f"**{most_vuln[0]}** with average CLR = {avg_rate:.1f}%")

        lines.extend(
            [
                "",
                "---",
                "",
                f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            ]
        )

        return "\n".join(lines)


# =============================================================================
# CSV EXPORTER
# =============================================================================


class CSVExporter:
    """Export benchmark results to CSV."""

    @staticmethod
    def export_model_summary(results: Dict, output_path: Path):
        """Export model summary to CSV."""
        models = results.get("models", {})

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "model",
                    "tsr",
                    "elr",
                    "wls",
                    "n_scenarios",
                    "total_tokens",
                    "total_cost",
                    "avg_latency_ms",
                ]
            )

            for model_name, model_data in models.items():
                writer.writerow(
                    [
                        model_name,
                        model_data.get("tsr", 0),
                        model_data.get("elr", 0),
                        model_data.get("wls", 0),
                        model_data.get("n_scenarios", 0),
                        model_data.get("total_tokens", 0),
                        model_data.get("total_cost", 0),
                        model_data.get("avg_latency_ms", 0),
                    ]
                )

    @staticmethod
    def export_channel_breakdown(results: Dict, output_path: Path):
        """Export channel breakdown to CSV."""
        models = results.get("models", {})

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            channels = [
                "C1_final_output",
                "C2_inter_agent",
                "C3_tool_input",
                "C4_tool_output",
                "C5_memory",
                "C6_logs",
                "C7_artifacts",
            ]
            writer.writerow(["model"] + channels)

            for model_name, model_data in models.items():
                clr = model_data.get("clr", {})
                row = [model_name] + [clr.get(ch, 0) for ch in channels]
                writer.writerow(row)

    @staticmethod
    def export_attack_breakdown(results: Dict, output_path: Path):
        """Export attack breakdown to CSV."""
        models = results.get("models", {})

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["model", "A0", "A1", "A2"])

            for model_name, model_data in models.items():
                asr = model_data.get("asr", {})
                writer.writerow(
                    [
                        model_name,
                        asr.get("A0", 0),
                        asr.get("A1", 0),
                        asr.get("A2", 0),
                    ]
                )


# =============================================================================
# COMPARISON REPORTER
# =============================================================================


class ComparisonReporter:
    """Compare multiple benchmark runs."""

    @staticmethod
    def compare(results_list: List[Dict]) -> str:
        """Generate comparison report between runs."""
        lines = [
            "# Benchmark Comparison Report",
            "",
            f"Comparing {len(results_list)} benchmark runs",
            "",
            "---",
            "",
            "## Run Summary",
            "",
            "| Run | Timestamp | Models | Scenarios | Overall ELR | Overall TSR |",
            "|-----|-----------|--------|-----------|-------------|-------------|",
        ]

        for i, results in enumerate(results_list, 1):
            timestamp = results.get("timestamp", "Unknown")
            n_models = len(results.get("models", {}))
            overall = results.get("overall_metrics", {})
            n_scenarios = overall.get("n_scenarios", 0)
            elr = overall.get("overall_elr", 0) * 100
            tsr = overall.get("overall_tsr", 0) * 100

            lines.append(
                f"| Run {i} | {timestamp} | {n_models} | {n_scenarios} | {elr:.1f}% | {tsr:.1f}% |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## Model-by-Model Comparison",
                "",
            ]
        )

        # Get all unique models
        all_models = set()
        for results in results_list:
            all_models.update(results.get("models", {}).keys())

        for model in sorted(all_models):
            lines.append(f"### {model}")
            lines.append("")
            lines.append("| Run | TSR | ELR | WLS |")
            lines.append("|-----|-----|-----|-----|")

            for i, results in enumerate(results_list, 1):
                model_data = results.get("models", {}).get(model, {})
                if model_data:
                    tsr = model_data.get("tsr", 0) * 100
                    elr = model_data.get("elr", 0) * 100
                    wls = model_data.get("wls", 0)
                    lines.append(f"| Run {i} | {tsr:.1f}% | {elr:.1f}% | {wls:.2f} |")
                else:
                    lines.append(f"| Run {i} | N/A | N/A | N/A |")

            lines.append("")

        return "\n".join(lines)


# =============================================================================
# MAIN REPORTER CLASS
# =============================================================================


class BenchmarkReporter:
    """Main reporter class."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "benchmark_results"
        self.latex = LaTeXGenerator()
        self.markdown = MarkdownGenerator()
        self.csv_export = CSVExporter()

    def generate_all_reports(self, results_path: Path):
        """Generate all report types from a results file."""
        with open(results_path) as f:
            results = json.load(f)

        timestamp = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        base_name = f"report_{timestamp}"

        # Generate LaTeX tables
        latex_dir = self.output_dir / "latex"
        latex_dir.mkdir(parents=True, exist_ok=True)

        tables = {
            "model_comparison": self.latex.model_comparison_table(results),
            "channel_breakdown": self.latex.channel_breakdown_table(results),
            "attack_comparison": self.latex.attack_comparison_table(results),
            "pareto_analysis": self.latex.pareto_table(results),
        }

        for name, content in tables.items():
            table_path = latex_dir / f"{base_name}_{name}.tex"
            with open(table_path, "w") as f:
                f.write(content)
            print(f"Generated: {table_path}")

        # Generate combined LaTeX file
        combined_latex = latex_dir / f"{base_name}_all_tables.tex"
        with open(combined_latex, "w") as f:
            f.write("% AgentLeak Benchmark Tables\n")
            f.write(f"% Generated: {datetime.now().isoformat()}\n\n")
            for name, content in tables.items():
                f.write(f"% {name}\n")
                f.write(content)
                f.write("\n\n")
        print(f"Generated: {combined_latex}")

        # Generate Markdown report
        md_path = self.output_dir / f"{base_name}.md"
        with open(md_path, "w") as f:
            f.write(self.markdown.full_report(results))
        print(f"Generated: {md_path}")

        # Generate CSV exports
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        self.csv_export.export_model_summary(results, csv_dir / f"{base_name}_models.csv")
        self.csv_export.export_channel_breakdown(results, csv_dir / f"{base_name}_channels.csv")
        self.csv_export.export_attack_breakdown(results, csv_dir / f"{base_name}_attacks.csv")
        print(f"Generated CSV exports in: {csv_dir}")

        return {
            "latex": latex_dir,
            "markdown": md_path,
            "csv": csv_dir,
        }

    def compare_runs(self, results_paths: List[Path], output_path: Optional[Path] = None):
        """Compare multiple benchmark runs."""
        results_list = []
        for path in results_paths:
            with open(path) as f:
                results_list.append(json.load(f))

        comparison = ComparisonReporter.compare(results_list)

        if output_path is None:
            output_path = self.output_dir / "comparison_report.md"

        with open(output_path, "w") as f:
            f.write(comparison)

        print(f"Generated comparison report: {output_path}")
        return output_path


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports from AgentLeak benchmark results"
    )

    parser.add_argument(
        "results_file", type=str, nargs="?", help="Path to benchmark results JSON file"
    )

    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple result files")

    parser.add_argument("--latex-only", action="store_true", help="Only generate LaTeX tables")

    parser.add_argument(
        "--markdown-only", action="store_true", help="Only generate Markdown report"
    )

    parser.add_argument("--csv-only", action="store_true", help="Only generate CSV exports")

    parser.add_argument("--output-dir", type=str, help="Output directory for reports")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    reporter = BenchmarkReporter(output_dir)

    if args.compare:
        paths = [Path(p) for p in args.compare]
        reporter.compare_runs(paths)
    elif args.results_file:
        results_path = Path(args.results_file)

        if not results_path.exists():
            print(f"Error: Results file not found: {results_path}")
            sys.exit(1)

        reporter.generate_all_reports(results_path)
    else:
        # Find latest results file
        results_dir = Path(__file__).parent.parent / "benchmark_results"
        json_files = list(results_dir.glob("benchmark_*.json"))

        if not json_files:
            print("No benchmark results found. Run full_benchmark.py first.")
            sys.exit(1)

        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest results: {latest}")
        reporter.generate_all_reports(latest)


if __name__ == "__main__":
    main()
