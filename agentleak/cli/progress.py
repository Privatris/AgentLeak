"""Progress tracking utilities for AgentLeak CLI."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class TestResult:
    """Result of a single test."""
    scenario_id: str
    task_success: bool = False  # Did the agent complete the task?
    leaked: bool = False        # Did a privacy leak occur?
    leak_channels: List[str] = field(default_factory=list)
    vertical: str = "unknown"
    attack_class: Optional[str] = None
    duration: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Test passed if task succeeded AND no leak occurred (for Benign)."""
        # Interpretation depends on goal, but generally "Safe & Functional"
        return self.task_success and not self.leaked

@dataclass 
class TestProgress:
    """Track progress of test execution."""
    total: int = 0
    completed: int = 0
    passed: int = 0  # Not used directly by add_result currently
    results: List[TestResult] = field(default_factory=list)
    
    def add_result(self, result: TestResult):
        """Update progress with a new result."""
        self.completed += 1
        self.results.append(result)
        # We can track specific metrics if needed
    
    @property
    def progress_pct(self) -> float:
        """Return completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0
