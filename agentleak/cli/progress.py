"""
AgentLeak Progress and Live Display.

Real-time progress tracking with Rich Live display,
showing test execution status, results, and metrics.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class TestResult:
    """Result from a single test."""

    scenario_id: str
    vertical: str
    attack_class: Optional[str] = None

    # Outcomes
    task_success: bool = False
    leaked: bool = False
    leak_channels: List[str] = field(default_factory=list)

    # Timing
    duration: float = 0.0

    # Details
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestProgress:
    """Tracks progress of test execution."""

    total: int
    completed: int = 0
    passed: int = 0
    leaked: int = 0
    failed: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.now)

    # Results
    results: List[TestResult] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def rate(self) -> float:
        """Tests per second."""
        if self.elapsed > 0:
            return self.completed / self.elapsed
        return 0.0

    @property
    def eta(self) -> float:
        """Estimated time remaining in seconds."""
        if self.rate > 0:
            return (self.total - self.completed) / self.rate
        return 0.0

    @property
    def leak_rate(self) -> float:
        """Current leak rate."""
        if self.completed > 0:
            return self.leaked / self.completed
        return 0.0

    @property
    def success_rate(self) -> float:
        """Task success rate."""
        if self.completed > 0:
            return self.passed / self.completed
        return 0.0

    def add_result(self, result: TestResult) -> None:
        """Add a test result."""
        self.results.append(result)
        self.completed += 1

        if result.task_success:
            self.passed += 1
        if result.leaked:
            self.leaked += 1
        if result.error:
            self.failed += 1


class ProgressBar:
    """Simple progress bar wrapper."""

    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def __enter__(self):
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        self.progress.__exit__(*args)

    def add_task(self, description: str, total: int) -> int:
        """Add a new task."""
        return self.progress.add_task(description, total=total)

    def update(self, task_id: int, advance: int = 1) -> None:
        """Update task progress."""
        self.progress.update(task_id, advance=advance)


class LiveDashboard:
    """
    Live dashboard for real-time test monitoring.

    Displays:
    - Overall progress bar
    - Current test status
    - Running metrics
    - Recent results table

    Similar to fuzzy-finder style interfaces.
    """

    def __init__(
        self,
        console: Console = None,
        title: str = "AgentLeak Benchmark",
        show_recent: int = 10,
    ):
        self.console = console or Console()
        self.title = title
        self.show_recent = show_recent

        self._progress: Optional[TestProgress] = None
        self._current_test: Optional[str] = None
        self._live: Optional[Live] = None
        self._lock = threading.Lock()

    def _make_header(self) -> Panel:
        """Create the header panel."""
        return Panel(
            Text("🔒 AgentLeak v1.0", style="bold cyan", justify="center"),
            box=box.DOUBLE,
            border_style="cyan",
            padding=(0, 2),
        )

    def _make_progress_panel(self) -> Panel:
        """Create the progress panel."""
        if not self._progress:
            return Panel("Waiting...", title="Progress")

        p = self._progress

        # Progress bar
        completed_pct = p.completed / p.total if p.total > 0 else 0
        bar_width = 40
        filled = int(completed_pct * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Stats
        content = Text()
        content.append(f"\n  Progress: [{bar}] {completed_pct:.1%}\n", style="cyan")
        content.append(f"\n  Completed: {p.completed}/{p.total}\n")
        content.append(f"  Passed:    {p.passed} ", style="green")
        content.append(f"({p.success_rate:.1%})\n", style="dim")
        content.append(f"  Leaked:    {p.leaked} ", style="red")
        content.append(f"({p.leak_rate:.1%})\n", style="dim")
        content.append(f"  Failed:    {p.failed}\n", style="yellow")
        content.append(f"\n  Elapsed:   {p.elapsed:.1f}s\n", style="dim")
        content.append(f"  Rate:      {p.rate:.2f} tests/s\n", style="dim")
        content.append(f"  ETA:       {p.eta:.1f}s\n", style="dim")

        if self._current_test:
            content.append(f"\n  Current:   {self._current_test}\n", style="cyan")

        return Panel(
            content,
            title="[bold]📊 Progress[/bold]",
            box=box.ROUNDED,
            border_style="cyan",
        )

    def _make_metrics_panel(self) -> Panel:
        """Create the live metrics panel."""
        if not self._progress:
            return Panel("Waiting...", title="Metrics")

        p = self._progress

        # Calculate metrics
        content = Text()

        # Task Success Rate (TSR)
        tsr = p.success_rate
        tsr_bar = "█" * int(tsr * 10) + "░" * (10 - int(tsr * 10))
        content.append(f"  TSR:  [{tsr_bar}] {tsr:.1%}\n", style="green" if tsr > 0.8 else "yellow")

        # Exact Leak Rate (ELR)
        elr = p.leak_rate
        elr_bar = "█" * int(elr * 10) + "░" * (10 - int(elr * 10))
        content.append(f"  ELR:  [{elr_bar}] {elr:.1%}\n", style="red" if elr > 0.3 else "green")

        # Channel breakdown (if we have results)
        if p.results:
            content.append("\n  Channel Leaks:\n", style="dim")
            channel_counts = {}
            for r in p.results:
                for ch in r.leak_channels:
                    channel_counts[ch] = channel_counts.get(ch, 0) + 1

            for ch in sorted(channel_counts.keys()):
                count = channel_counts[ch]
                content.append(f"    {ch}: {count}\n", style="dim")

        return Panel(
            content,
            title="[bold]📈 Live Metrics[/bold]",
            box=box.ROUNDED,
            border_style="cyan",
        )

    def _make_results_table(self) -> Panel:
        """Create the recent results table."""
        if not self._progress or not self._progress.results:
            return Panel("No results yet...", title="Recent Results")

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )

        table.add_column("ID", style="dim", width=15)
        table.add_column("Vertical", width=12)
        table.add_column("Attack", width=15)
        table.add_column("Task", justify="center", width=8)
        table.add_column("Leak", justify="center", width=8)
        table.add_column("Channels", width=15)
        table.add_column("Time", justify="right", width=8)

        # Get recent results
        recent = self._progress.results[-self.show_recent :]

        for r in reversed(recent):
            task_status = "[green]✓[/green]" if r.task_success else "[red]✗[/red]"
            leak_status = "[red]YES[/red]" if r.leaked else "[green]NO[/green]"
            channels = ",".join(r.leak_channels) if r.leak_channels else "-"

            table.add_row(
                r.scenario_id[:15],
                r.vertical,
                r.attack_class or "-",
                task_status,
                leak_status,
                channels,
                f"{r.duration:.2f}s",
            )

        return Panel(
            table,
            title="[bold]📋 Recent Results[/bold]",
            box=box.ROUNDED,
            border_style="cyan",
        )

    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="results", size=15),
        )

        layout["body"].split_row(
            Layout(name="progress"),
            Layout(name="metrics"),
        )

        layout["header"].update(self._make_header())
        layout["progress"].update(self._make_progress_panel())
        layout["metrics"].update(self._make_metrics_panel())
        layout["results"].update(self._make_results_table())

        return layout

    def start(self, total: int) -> None:
        """Start the live dashboard."""
        self._progress = TestProgress(total=total)
        self._live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.__enter__()

    def update(
        self,
        result: TestResult = None,
        current_test: str = None,
    ) -> None:
        """Update the dashboard."""
        with self._lock:
            if result:
                self._progress.add_result(result)
            if current_test:
                self._current_test = current_test

            if self._live:
                self._live.update(self._make_layout())

    def stop(self) -> TestProgress:
        """Stop the dashboard and return final progress."""
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

        return self._progress


class ProgressManager:
    """
    Manages progress display for benchmark runs.

    Provides both simple progress bar and full dashboard modes.

    Example:
        # Simple mode
        with ProgressManager(total=100, mode="simple") as pm:
            for scenario in scenarios:
                result = run_test(scenario)
                pm.update(result)

        # Dashboard mode
        with ProgressManager(total=100, mode="dashboard") as pm:
            for scenario in scenarios:
                pm.set_current(scenario.id)
                result = run_test(scenario)
                pm.update(result)
    """

    def __init__(
        self,
        total: int,
        mode: str = "dashboard",  # "simple", "dashboard", "quiet"
        console: Console = None,
    ):
        self.total = total
        self.mode = mode
        self.console = console or Console()

        self._progress: Optional[TestProgress] = None
        self._dashboard: Optional[LiveDashboard] = None
        self._simple_progress: Optional[Progress] = None
        self._task_id: Optional[int] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self) -> None:
        """Start progress tracking."""
        self._progress = TestProgress(total=self.total)

        if self.mode == "dashboard":
            self._dashboard = LiveDashboard(console=self.console)
            self._dashboard.start(self.total)
        elif self.mode == "simple":
            self._simple_progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
            )
            self._simple_progress.__enter__()
            self._task_id = self._simple_progress.add_task(
                "Running tests...",
                total=self.total,
            )

    def update(self, result: TestResult = None) -> None:
        """Update progress with a test result."""
        if result:
            self._progress.add_result(result)

        if self.mode == "dashboard" and self._dashboard:
            self._dashboard.update(result=result)
        elif self.mode == "simple" and self._simple_progress:
            self._simple_progress.update(self._task_id, advance=1)

    def set_current(self, test_id: str) -> None:
        """Set the current test being executed."""
        if self.mode == "dashboard" and self._dashboard:
            self._dashboard.update(current_test=test_id)

    def stop(self) -> TestProgress:
        """Stop progress tracking and return results."""
        if self.mode == "dashboard" and self._dashboard:
            return self._dashboard.stop()
        elif self.mode == "simple" and self._simple_progress:
            self._simple_progress.__exit__(None, None, None)

        return self._progress

    def get_progress(self) -> TestProgress:
        """Get current progress state."""
        return self._progress
