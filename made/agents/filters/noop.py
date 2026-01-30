from typing import Any

from pymatgen.core.structure import Structure

from made.agents.base import Filter, FilterResult


class NoOpFilter(Filter):
    """No-op filter that returns all candidates unchanged.

    This filter does nothing - it's a pass-through filter for cases where
    no filtering is desired. All candidate structures are returned as-is.
    """

    filter_name = "NoOp"

    def __init__(self) -> None:
        """Initialize the no-op filter."""

    def _filter_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Return all candidates with passing results."""
        results = [
            FilterResult(passed=True, filter_name=self.filter_name) for _ in candidates
        ]
        return candidates, results

    def get_state(self) -> dict[str, Any]:
        """Return the current state of the filter."""
        return {}

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the filter (no-op)."""
