"""
AgentLeak Pareto Calculator - Privacy-utility tradeoff analysis.

Calculates Pareto frontiers and area under the curve (AUC)
for comparing agents/defenses on privacy-utility tradeoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParetoPoint:
    """
    A single point in the privacy-utility space.

    Convention:
    - privacy_cost: Lower is better (0 = no leakage)
    - utility: Higher is better (1 = perfect task success)
    """

    name: str  # Agent/defense identifier
    privacy_cost: float  # ELR or WLS
    utility: float  # TSR

    # Optional metadata
    config: Optional[dict] = None
    run_id: Optional[str] = None

    def dominates(self, other: "ParetoPoint") -> bool:
        """
        Check if this point Pareto-dominates another.

        A dominates B if A is better or equal in all dimensions,
        and strictly better in at least one.
        """
        better_privacy = self.privacy_cost <= other.privacy_cost
        better_utility = self.utility >= other.utility
        strictly_better = self.privacy_cost < other.privacy_cost or self.utility > other.utility
        return better_privacy and better_utility and strictly_better

    def distance_to(self, other: "ParetoPoint") -> float:
        """Euclidean distance to another point."""
        return math.sqrt(
            (self.privacy_cost - other.privacy_cost) ** 2 + (self.utility - other.utility) ** 2
        )


@dataclass
class ParetoFrontier:
    """
    Pareto frontier of optimal points.

    Contains points that are not dominated by any other point.
    """

    points: list[ParetoPoint] = field(default_factory=list)
    all_points: list[ParetoPoint] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0

    def get_optimal_point(
        self,
        max_privacy_cost: Optional[float] = None,
        min_utility: Optional[float] = None,
    ) -> Optional[ParetoPoint]:
        """
        Get the optimal point satisfying constraints.

        Args:
            max_privacy_cost: Maximum acceptable privacy cost
            min_utility: Minimum acceptable utility

        Returns:
            Best point satisfying constraints, or None
        """
        candidates = self.points

        if max_privacy_cost is not None:
            candidates = [p for p in candidates if p.privacy_cost <= max_privacy_cost]

        if min_utility is not None:
            candidates = [p for p in candidates if p.utility >= min_utility]

        if not candidates:
            return None

        # Return point with best balance (closest to ideal corner)
        ideal = ParetoPoint(name="ideal", privacy_cost=0.0, utility=1.0)
        return min(candidates, key=lambda p: p.distance_to(ideal))


class ParetoCalculator:
    """
    Calculator for Pareto frontiers and AUC.

    Example:
        calc = ParetoCalculator()

        # Add points from different agents
        calc.add_point("gpt-4", privacy_cost=0.15, utility=0.92)
        calc.add_point("sanitizer", privacy_cost=0.05, utility=0.88)
        calc.add_point("claude-3", privacy_cost=0.12, utility=0.89)

        # Get frontier
        frontier = calc.compute_frontier()

        # Calculate AUC
        auc = calc.compute_auc()
        print(f"Pareto AUC: {auc:.3f}")
    """

    def __init__(self):
        """Initialize calculator."""
        self.points: list[ParetoPoint] = []

    def add_point(
        self,
        name: str,
        privacy_cost: float,
        utility: float,
        **metadata,
    ) -> ParetoPoint:
        """
        Add a point to the analysis.

        Args:
            name: Identifier for this configuration
            privacy_cost: ELR or WLS (lower is better)
            utility: TSR (higher is better)
            **metadata: Additional metadata

        Returns:
            The created ParetoPoint
        """
        point = ParetoPoint(
            name=name,
            privacy_cost=privacy_cost,
            utility=utility,
            config=metadata if metadata else None,
        )
        self.points.append(point)
        return point

    def clear(self) -> None:
        """Clear all points."""
        self.points.clear()

    def compute_frontier(self) -> ParetoFrontier:
        """
        Compute the Pareto frontier.

        Returns:
            ParetoFrontier with optimal points
        """
        if not self.points:
            return ParetoFrontier()

        # Find non-dominated points
        optimal = []
        for p in self.points:
            is_dominated = any(other.dominates(p) for other in self.points if other != p)
            if not is_dominated:
                optimal.append(p)

        # Sort by privacy cost for proper frontier ordering
        optimal.sort(key=lambda p: p.privacy_cost)

        return ParetoFrontier(
            points=optimal,
            all_points=self.points.copy(),
        )

    def compute_auc(
        self,
        normalize: bool = True,
        max_privacy_cost: float = 1.0,
    ) -> float:
        """
        Compute area under the Pareto frontier.

        Uses trapezoidal integration. Higher AUC = better tradeoff.

        Args:
            normalize: If True, normalize to [0, 1] range
            max_privacy_cost: Maximum privacy cost for normalization

        Returns:
            Area under curve (higher is better)
        """
        frontier = self.compute_frontier()

        if len(frontier.points) < 2:
            if len(frontier.points) == 1:
                # Single point: area is rectangle from origin
                p = frontier.points[0]
                area = (max_privacy_cost - p.privacy_cost) * p.utility
            else:
                return 0.0
        else:
            # Sort points by privacy cost
            sorted_points = sorted(frontier.points, key=lambda p: p.privacy_cost)

            # Trapezoidal integration
            area = 0.0

            # Area from x=0 to first point (assume utility drops to 0 at x=0)
            first = sorted_points[0]
            area += first.privacy_cost * first.utility / 2

            # Area between consecutive points
            for i in range(len(sorted_points) - 1):
                p1 = sorted_points[i]
                p2 = sorted_points[i + 1]
                width = p2.privacy_cost - p1.privacy_cost
                avg_height = (p1.utility + p2.utility) / 2
                area += width * avg_height

            # Area from last point to max_privacy_cost
            last = sorted_points[-1]
            if last.privacy_cost < max_privacy_cost:
                area += (max_privacy_cost - last.privacy_cost) * last.utility

        if normalize:
            # Max possible area is max_privacy_cost * 1.0 (utility)
            area = area / max_privacy_cost

        return area

    def compute_hypervolume(
        self,
        reference_point: Optional[tuple[float, float]] = None,
    ) -> float:
        """
        Compute hypervolume indicator.

        The hypervolume is the area dominated by the frontier
        relative to a reference point (worst case).

        Args:
            reference_point: (max_privacy, min_utility) reference

        Returns:
            Hypervolume indicator (higher is better)
        """
        if reference_point is None:
            reference_point = (1.0, 0.0)  # Worst: 100% leakage, 0% utility

        ref_x, ref_y = reference_point

        frontier = self.compute_frontier()
        if frontier.is_empty:
            return 0.0

        # Sort by privacy cost (x-axis)
        sorted_points = sorted(frontier.points, key=lambda p: p.privacy_cost)

        # Calculate dominated area
        area = 0.0
        prev_x = ref_x  # Start from reference x (worst privacy)

        for point in reversed(sorted_points):  # Process right to left
            if point.utility > ref_y and point.privacy_cost < prev_x:
                # Width from this point to previous x
                width = prev_x - point.privacy_cost
                # Height from reference y to this utility
                height = point.utility - ref_y
                area += width * height
                prev_x = point.privacy_cost

        return area

    def rank_points(self) -> list[tuple[int, ParetoPoint]]:
        """
        Rank all points by Pareto dominance layers.

        Returns:
            List of (rank, point) tuples, rank 1 = frontier
        """
        remaining = self.points.copy()
        ranked = []
        current_rank = 1

        while remaining:
            # Find non-dominated points in remaining
            layer = []
            for p in remaining:
                is_dominated = any(other.dominates(p) for other in remaining if other != p)
                if not is_dominated:
                    layer.append(p)

            # Add to ranked
            for p in layer:
                ranked.append((current_rank, p))
                remaining.remove(p)

            current_rank += 1

        return ranked

    def summary(self) -> str:
        """Generate human-readable summary."""
        frontier = self.compute_frontier()
        auc = self.compute_auc()
        hypervolume = self.compute_hypervolume()

        lines = [
            "=" * 50,
            "Pareto Analysis Summary",
            "=" * 50,
            "",
            f"Total Points: {len(self.points)}",
            f"Frontier Points: {len(frontier.points)}",
            f"Pareto AUC: {auc:.3f}",
            f"Hypervolume: {hypervolume:.3f}",
            "",
            "Frontier Points:",
        ]

        for p in frontier.points:
            lines.append(f"  {p.name}: privacy={p.privacy_cost:.3f}, utility={p.utility:.3f}")

        lines.append("")
        lines.append("Rankings:")
        for rank, point in self.rank_points():
            lines.append(f"  Rank {rank}: {point.name}")

        lines.append("=" * 50)
        return "\n".join(lines)
