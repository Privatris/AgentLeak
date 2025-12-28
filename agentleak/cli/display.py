"""
AgentLeak CLI Display Components.

Rich-based display components for the interactive CLI,
including logo, panels, and styled output.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import ROUNDED, HEAVY, DOUBLE
from rich.style import Style
from rich.align import Align
from rich import box
from typing import Optional, Dict, Any, List


# ASCII Art Logo for AgentLeak
LOGO_ASCII = r"""
    ___                    __  __               __  
   /   | ____ ____  ____  / /_/ /   ___  ____ _/ /__
  / /| |/ __ `/ _ \/ __ \/ __/ /   / _ \/ __ `/ //_/
 / ___ / /_/ /  __/ / / / /_/ /___/  __/ /_/ / ,<   
/_/  |_\__, /\___/_/ /_/\__/_____/\___/\__,_/_/|_|  
      /____/                                        
"""

LOGO_SMALL = r"""
╔═══════════════════════════════════════╗
║          🔒 AgentLeak v1.0            ║
║   Privacy Benchmark for LLM Agents    ║
╚═══════════════════════════════════════╝
"""

TAGLINE = "A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems"


class Logo:
    """Logo display utilities."""
    
    @staticmethod
    def render(console: Console, style: str = "full") -> None:
        """Render the AgentLeak logo."""
        if style == "full":
            logo_text = Text(LOGO_ASCII)
            logo_text.stylize("bold cyan")
            console.print(logo_text)
            console.print(f"[dim]{TAGLINE}[/dim]", justify="center")
            console.print()
        elif style == "small":
            console.print(LOGO_SMALL, style="cyan")
        elif style == "minimal":
            console.print("[bold cyan]🔒 AgentLeak v1.0[/bold cyan]")
    
    @staticmethod
    def get_header_panel(subtitle: str = None) -> Panel:
        """Get a header panel with logo."""
        title = "[bold cyan]🔒 AgentLeak v1.0[/bold cyan]"
        if subtitle:
            title += f"\n[dim]{subtitle}[/dim]"
        
        return Panel(
            Align.center(title),
            box=box.DOUBLE,
            border_style="cyan",
            padding=(0, 2),
        )


class Display:
    """
    Display manager for AgentLeak CLI.
    
    Provides styled output for:
    - Tables
    - Panels
    - Status messages
    - Progress information
    """
    
    def __init__(self, console: Console = None, theme: str = "dark"):
        self.console = console or Console()
        self.theme = theme
        
        # Theme colors
        self.colors = {
            "primary": "cyan",
            "success": "green",
            "error": "red",
            "warning": "yellow",
            "info": "blue",
            "muted": "dim white",
        }
    
    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()
    
    def print_logo(self, style: str = "full") -> None:
        """Print the logo."""
        Logo.render(self.console, style)
    
    def print_header(self, title: str, subtitle: str = None) -> None:
        """Print a styled header."""
        header = f"[bold {self.colors['primary']}]{title}[/bold {self.colors['primary']}]"
        if subtitle:
            header += f"\n[{self.colors['muted']}]{subtitle}[/{self.colors['muted']}]"
        
        self.console.print(Panel(
            Align.center(header),
            box=box.ROUNDED,
            border_style=self.colors['primary'],
        ))
    
    def print_section(self, title: str) -> None:
        """Print a section header."""
        self.console.print()
        self.console.print(f"[bold {self.colors['primary']}]━━━ {title} ━━━[/bold {self.colors['primary']}]")
        self.console.print()
    
    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[{self.colors['success']}]✓[/{self.colors['success']}] {message}")
    
    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[{self.colors['error']}]✗[/{self.colors['error']}] {message}")
    
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[{self.colors['warning']}]⚠[/{self.colors['warning']}] {message}")
    
    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[{self.colors['info']}]ℹ[/{self.colors['info']}] {message}")
    
    def print_status(self, label: str, value: str, style: str = None) -> None:
        """Print a status line."""
        style = style or self.colors['primary']
        self.console.print(f"  [{self.colors['muted']}]{label}:[/{self.colors['muted']}] [{style}]{value}[/{style}]")
    
    def create_table(
        self,
        title: str = None,
        columns: List[Dict[str, Any]] = None,
        rows: List[List[Any]] = None,
        show_header: bool = True,
        box_style: box.Box = box.ROUNDED,
    ) -> Table:
        """Create a styled table."""
        table = Table(
            title=title,
            box=box_style,
            show_header=show_header,
            header_style=f"bold {self.colors['primary']}",
            border_style=self.colors['muted'],
        )
        
        if columns:
            for col in columns:
                table.add_column(
                    col.get("header", ""),
                    justify=col.get("justify", "left"),
                    style=col.get("style", ""),
                    width=col.get("width"),
                )
        
        if rows:
            for row in rows:
                table.add_row(*[str(cell) for cell in row])
        
        return table
    
    def print_config_summary(self, config: Dict[str, Any]) -> None:
        """Print a configuration summary."""
        table = self.create_table(
            title="Configuration",
            columns=[
                {"header": "Setting", "style": "cyan"},
                {"header": "Value", "style": "white"},
            ],
        )
        
        # Flatten config for display
        def add_rows(d: dict, prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict) and len(value) <= 5:
                    add_rows(value, full_key)
                else:
                    display_value = str(value)
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    table.add_row(full_key, display_value)
        
        add_rows(config)
        self.console.print(table)
    
    def print_results_table(
        self,
        results: List[Dict[str, Any]],
        title: str = "Results",
    ) -> None:
        """Print a results table."""
        if not results:
            self.print_warning("No results to display")
            return
        
        # Get columns from first result
        columns = list(results[0].keys())
        
        table = self.create_table(
            title=title,
            columns=[{"header": col, "style": "white"} for col in columns],
        )
        
        for result in results:
            row = []
            for col in columns:
                value = result.get(col, "")
                # Style based on column type
                if col.lower() in ("status", "result"):
                    if str(value).lower() in ("pass", "success", "ok"):
                        value = f"[green]{value}[/green]"
                    elif str(value).lower() in ("fail", "error", "leaked"):
                        value = f"[red]{value}[/red]"
                elif col.lower() in ("leak", "leakage", "leaked"):
                    if value:
                        value = f"[red]Yes[/red]"
                    else:
                        value = f"[green]No[/green]"
                row.append(str(value))
            table.add_row(*row)
        
        self.console.print(table)
    
    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Print a metrics summary with visual indicators."""
        table = self.create_table(
            title="📊 Metrics Summary",
            columns=[
                {"header": "Metric", "style": "cyan", "width": 25},
                {"header": "Value", "justify": "right", "width": 15},
                {"header": "Indicator", "width": 30},
            ],
        )
        
        for name, value in metrics.items():
            # Create bar indicator
            if 0 <= value <= 1:
                bar_length = int(value * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                # Color based on metric type
                if "leak" in name.lower() or "error" in name.lower():
                    # Lower is better
                    if value < 0.2:
                        color = "green"
                    elif value < 0.5:
                        color = "yellow"
                    else:
                        color = "red"
                else:
                    # Higher is better (success, utility)
                    if value > 0.8:
                        color = "green"
                    elif value > 0.5:
                        color = "yellow"
                    else:
                        color = "red"
                
                indicator = f"[{color}]{bar}[/{color}]"
            else:
                indicator = ""
            
            table.add_row(name, f"{value:.2%}" if value <= 1 else f"{value:.2f}", indicator)
        
        self.console.print(table)
    
    def print_channel_summary(self, channel_stats: Dict[str, Dict]) -> None:
        """Print channel-wise statistics."""
        table = self.create_table(
            title="📡 Channel Analysis (C1-C7)",
            columns=[
                {"header": "Channel", "style": "cyan", "width": 8},
                {"header": "Name", "width": 20},
                {"header": "Events", "justify": "right", "width": 10},
                {"header": "Leaks", "justify": "right", "width": 10},
                {"header": "Rate", "justify": "right", "width": 10},
            ],
        )
        
        for channel_id, stats in channel_stats.items():
            events = stats.get("event_count", 0)
            leaks = stats.get("leak_count", 0)
            rate = leaks / events if events > 0 else 0
            
            rate_str = f"[red]{rate:.1%}[/red]" if rate > 0.1 else f"[green]{rate:.1%}[/green]"
            
            table.add_row(
                channel_id,
                stats.get("name", ""),
                str(events),
                str(leaks),
                rate_str,
            )
        
        self.console.print(table)
    
    def print_attack_summary(self, attack_stats: Dict[str, Dict]) -> None:
        """Print attack family statistics."""
        table = self.create_table(
            title="⚔️ Attack Analysis (F1-F5)",
            columns=[
                {"header": "Family", "style": "cyan", "width": 8},
                {"header": "Name", "width": 25},
                {"header": "Attempts", "justify": "right", "width": 10},
                {"header": "Success", "justify": "right", "width": 10},
                {"header": "ASR", "justify": "right", "width": 10},
            ],
        )
        
        for family_id, stats in attack_stats.items():
            attempts = stats.get("attempts", 0)
            success = stats.get("success", 0)
            asr = success / attempts if attempts > 0 else 0
            
            asr_str = f"[red]{asr:.1%}[/red]" if asr > 0.3 else f"[yellow]{asr:.1%}[/yellow]"
            
            table.add_row(
                family_id,
                stats.get("name", ""),
                str(attempts),
                str(success),
                asr_str,
            )
        
        self.console.print(table)
    
    def print_comparison_table(
        self,
        models: List[str],
        metrics: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
    ) -> None:
        """Print a comparison table across models."""
        table = self.create_table(
            title=title,
            columns=[
                {"header": "Metric", "style": "cyan", "width": 20},
                *[{"header": model, "justify": "center", "width": 15} for model in models],
            ],
        )
        
        for metric_name, model_values in metrics.items():
            row = [metric_name]
            
            # Find best value
            values = list(model_values.values())
            if "leak" in metric_name.lower():
                best = min(values)
            else:
                best = max(values)
            
            for model in models:
                value = model_values.get(model, 0)
                formatted = f"{value:.2%}" if value <= 1 else f"{value:.2f}"
                
                if value == best:
                    formatted = f"[bold green]{formatted}[/bold green]"
                
                row.append(formatted)
            
            table.add_row(*row)
        
        self.console.print(table)
