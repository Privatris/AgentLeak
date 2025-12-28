"""
AgentLeak Interactive Menu System.

Provides an interactive menu interface similar to fuzzy-finder tools,
allowing users to navigate options with keyboard.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box
import sys


@dataclass
class MenuItem:
    """A single menu item."""
    
    key: str  # Keyboard shortcut
    label: str  # Display label
    description: str = ""  # Optional description
    action: Optional[Callable] = None  # Action to execute
    submenu: Optional["Menu"] = None  # Nested menu
    enabled: bool = True
    
    @property
    def display(self) -> str:
        """Get display string for this item."""
        if self.enabled:
            return f"[cyan][{self.key}][/cyan] {self.label}"
        else:
            return f"[dim][{self.key}] {self.label}[/dim]"


class Menu:
    """
    Interactive menu with keyboard navigation.
    
    Example:
        menu = Menu("Main Menu")
        menu.add_item("1", "Run Benchmark", action=run_benchmark)
        menu.add_item("2", "Configure", submenu=config_menu)
        menu.add_item("q", "Quit", action=lambda: sys.exit(0))
        
        menu.show()
    """
    
    def __init__(
        self,
        title: str,
        subtitle: str = None,
        console: Console = None,
        show_header: bool = True,
    ):
        self.title = title
        self.subtitle = subtitle
        self.console = console or Console()
        self.show_header = show_header
        self.items: List[MenuItem] = []
        self._parent: Optional["Menu"] = None
    
    def add_item(
        self,
        key: str,
        label: str,
        description: str = "",
        action: Callable = None,
        submenu: "Menu" = None,
        enabled: bool = True,
    ) -> "Menu":
        """Add a menu item. Returns self for chaining."""
        item = MenuItem(
            key=key,
            label=label,
            description=description,
            action=action,
            submenu=submenu,
            enabled=enabled,
        )
        
        if submenu:
            submenu._parent = self
        
        self.items.append(item)
        return self
    
    def add_separator(self) -> "Menu":
        """Add a visual separator."""
        self.items.append(MenuItem(key="", label="─" * 40, enabled=False))
        return self
    
    def render(self) -> Panel:
        """Render the menu as a Rich Panel."""
        # Build menu content
        lines = []
        
        for item in self.items:
            if not item.key:  # Separator
                lines.append(f"[dim]{item.label}[/dim]")
            else:
                line = item.display
                if item.description:
                    line += f"  [dim]{item.description}[/dim]"
                if item.submenu:
                    line += " [dim]→[/dim]"
                lines.append(line)
        
        content = "\n".join(lines)
        
        # Create panel
        title = f"[bold cyan]{self.title}[/bold cyan]"
        if self.subtitle:
            title += f"\n[dim]{self.subtitle}[/dim]"
        
        return Panel(
            content,
            title=title,
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2),
        )
    
    def show(self) -> Optional[Any]:
        """Display the menu and wait for selection."""
        while True:
            # Clear and show menu
            self.console.clear()
            self.console.print(self.render())
            self.console.print()
            
            # Get input
            try:
                choice = Prompt.ask(
                    "[cyan]Select option[/cyan]",
                    console=self.console,
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                return None
            
            # Find matching item
            for item in self.items:
                if item.key.lower() == choice and item.enabled:
                    if item.submenu:
                        result = item.submenu.show()
                        if result is not None:
                            return result
                        # Continue showing this menu after submenu exits
                        break
                    elif item.action:
                        result = item.action()
                        if result is not None:
                            return result
                        return choice
            else:
                # No match, show error
                self.console.print("[red]Invalid option. Please try again.[/red]")
                self.console.input("[dim]Press Enter to continue...[/dim]")
    
    def show_once(self) -> Optional[str]:
        """Display the menu once and return the selection (no loop)."""
        self.console.print(self.render())
        self.console.print()
        
        try:
            choice = Prompt.ask(
                "[cyan]Select option[/cyan]",
                console=self.console,
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None
        
        return choice


class MenuBuilder:
    """
    Builder for creating complex menus.
    
    Example:
        menu = (MenuBuilder("AgentLeak")
            .add("1", "Quick Test", "Run 10 scenarios")
            .add("2", "Full Benchmark", "Run 1000 scenarios")
            .separator()
            .submenu("c", "Configure", config_builder)
            .separator()
            .add("q", "Quit")
            .build())
    """
    
    def __init__(self, title: str, subtitle: str = None):
        self.title = title
        self.subtitle = subtitle
        self._items: List[Dict[str, Any]] = []
    
    def add(
        self,
        key: str,
        label: str,
        description: str = "",
        action: Callable = None,
        enabled: bool = True,
    ) -> "MenuBuilder":
        """Add a menu item."""
        self._items.append({
            "type": "item",
            "key": key,
            "label": label,
            "description": description,
            "action": action,
            "enabled": enabled,
        })
        return self
    
    def separator(self) -> "MenuBuilder":
        """Add a separator."""
        self._items.append({"type": "separator"})
        return self
    
    def submenu(
        self,
        key: str,
        label: str,
        builder: "MenuBuilder",
        description: str = "",
    ) -> "MenuBuilder":
        """Add a submenu."""
        self._items.append({
            "type": "submenu",
            "key": key,
            "label": label,
            "description": description,
            "builder": builder,
        })
        return self
    
    def build(self, console: Console = None) -> Menu:
        """Build the menu."""
        menu = Menu(self.title, self.subtitle, console)
        
        for item in self._items:
            if item["type"] == "separator":
                menu.add_separator()
            elif item["type"] == "submenu":
                submenu = item["builder"].build(console)
                menu.add_item(
                    item["key"],
                    item["label"],
                    item.get("description", ""),
                    submenu=submenu,
                )
            else:
                menu.add_item(
                    item["key"],
                    item["label"],
                    item.get("description", ""),
                    action=item.get("action"),
                    enabled=item.get("enabled", True),
                )
        
        return menu


class ConfigWizard:
    """
    Interactive configuration wizard.
    
    Guides the user through setting up test parameters.
    """
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def run(self) -> Dict[str, Any]:
        """Run the configuration wizard."""
        config = {}
        
        self.console.clear()
        self.console.print(Panel(
            "[bold cyan]🔧 Configuration Wizard[/bold cyan]\n"
            "[dim]Configure your benchmark run[/dim]",
            box=box.DOUBLE,
            border_style="cyan",
        ))
        self.console.print()
        
        # Model selection
        self.console.print("[bold]1. Model Selection[/bold]")
        models = [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2.5-72b-instruct",
        ]
        for i, model in enumerate(models, 1):
            self.console.print(f"  [{i}] {model}")
        
        model_choice = IntPrompt.ask(
            "Select model",
            default=1,
            console=self.console,
        )
        config["model"] = models[min(model_choice - 1, len(models) - 1)]
        self.console.print()
        
        # Number of scenarios
        self.console.print("[bold]2. Test Scale[/bold]")
        config["n_scenarios"] = IntPrompt.ask(
            "Number of scenarios",
            default=100,
            console=self.console,
        )
        self.console.print()
        
        # Verticals
        self.console.print("[bold]3. Domains[/bold]")
        all_verticals = ["healthcare", "finance", "legal", "corporate"]
        self.console.print("  Available: " + ", ".join(all_verticals))
        
        if Confirm.ask("Test all domains?", default=True, console=self.console):
            config["verticals"] = all_verticals
        else:
            selected = Prompt.ask(
                "Enter domains (comma-separated)",
                default="healthcare,finance",
                console=self.console,
            )
            config["verticals"] = [v.strip() for v in selected.split(",")]
        self.console.print()
        
        # Attacks
        self.console.print("[bold]4. Attack Configuration[/bold]")
        config["enable_attacks"] = Confirm.ask(
            "Enable adversarial attacks?",
            default=True,
            console=self.console,
        )
        
        if config["enable_attacks"]:
            config["attack_probability"] = IntPrompt.ask(
                "Attack probability (%)",
                default=50,
                console=self.console,
            ) / 100.0
        self.console.print()
        
        # Defense
        self.console.print("[bold]5. Defense Configuration[/bold]")
        defenses = ["none", "sanitizer", "filter"]
        for i, defense in enumerate(defenses):
            self.console.print(f"  [{i}] {defense}")
        
        defense_choice = IntPrompt.ask(
            "Select defense",
            default=0,
            console=self.console,
        )
        defense = defenses[min(defense_choice, len(defenses) - 1)]
        config["defense"] = None if defense == "none" else defense
        self.console.print()
        
        # Summary
        self.console.print(Panel(
            "[bold green]✓ Configuration Complete[/bold green]",
            box=box.ROUNDED,
            border_style="green",
        ))
        
        return config


class ModelSelector:
    """Interactive model selector with multi-select support."""
    
    AVAILABLE_MODELS = [
        {"name": "gpt-4o-mini", "provider": "OpenAI", "cost": "$"},
        {"name": "gpt-4o", "provider": "OpenAI", "cost": "$$$"},
        {"name": "gpt-4-turbo", "provider": "OpenAI", "cost": "$$$"},
        {"name": "claude-3-haiku-20240307", "provider": "Anthropic", "cost": "$"},
        {"name": "claude-3-5-sonnet-20241022", "provider": "Anthropic", "cost": "$$"},
        {"name": "google/gemini-2.0-flash-exp:free", "provider": "Google", "cost": "Free"},
        {"name": "google/gemini-pro", "provider": "Google", "cost": "$"},
        {"name": "qwen/qwen-2.5-72b-instruct", "provider": "Alibaba", "cost": "$$"},
        {"name": "meta-llama/llama-3.3-70b-instruct", "provider": "Meta", "cost": "$$"},
    ]
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
    
    def select_single(self) -> Optional[str]:
        """Select a single model."""
        table = Table(title="Available Models", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Model", style="white")
        table.add_column("Provider", style="dim")
        table.add_column("Cost", style="yellow")
        
        for i, model in enumerate(self.AVAILABLE_MODELS, 1):
            table.add_row(str(i), model["name"], model["provider"], model["cost"])
        
        self.console.print(table)
        self.console.print()
        
        choice = IntPrompt.ask(
            "Select model",
            default=1,
            console=self.console,
        )
        
        idx = min(choice - 1, len(self.AVAILABLE_MODELS) - 1)
        return self.AVAILABLE_MODELS[idx]["name"]
    
    def select_multiple(self) -> List[str]:
        """Select multiple models for comparison."""
        table = Table(title="Available Models (Select Multiple)", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Model", style="white")
        table.add_column("Provider", style="dim")
        table.add_column("Cost", style="yellow")
        
        for i, model in enumerate(self.AVAILABLE_MODELS, 1):
            table.add_row(str(i), model["name"], model["provider"], model["cost"])
        
        self.console.print(table)
        self.console.print()
        
        selection = Prompt.ask(
            "Select models (comma-separated numbers, e.g., 1,3,5)",
            default="1,4",
            console=self.console,
        )
        
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        return [
            self.AVAILABLE_MODELS[i]["name"]
            for i in indices
            if 0 <= i < len(self.AVAILABLE_MODELS)
        ]
