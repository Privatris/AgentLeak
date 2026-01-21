"""Progress tracking utilities for AgentLeak CLI."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class TestResult:
    """Result of a single test."""
    scenario_id: str
    passed: bool
    leaked: bool = False
    channel: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class TestProgress:
    """Track progress of test execution."""
    total: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    results: List[TestResult] = field(default_factory=list)
    
    def update(self, result: TestResult):
        """Update progress with a new result."""
        self.completed += 1
        self.results.append(result)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1
    
    @property
    def progress_pct(self) -> float:
        """Return completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0
