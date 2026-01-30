from typing import Any

from pymatgen.core.structure import Structure

from made.agents.base import Filter, FilterResult
from made.evaluation.metrics import is_min_distance_valid


class MinDistanceFilter(Filter):
    """Filter that keeps only structures with minimum interatomic distance above threshold.

    This filter removes structures where atoms are too close together, which is
    typically a sign of invalid or unrealistic structures.
    """

    filter_name = "MinDistance"

    def __init__(self, min_distance_threshold: float = 0.5) -> None:
        """Initialize the min distance filter.

        Args:
            min_distance_threshold: Minimum allowed interatomic distance in Angstroms.
                Structures with minimum distance below this threshold are filtered out.
                Default is 0.5 Angstroms.
        """
        self.min_distance_threshold = min_distance_threshold

    def _filter_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Filter with detailed min distance results."""
        passed = []
        results = []

        for cand in candidates:
            # Use existing function with return_distance=True
            is_valid, min_dist = is_min_distance_valid(
                cand, self.min_distance_threshold, return_distance=True
            )

            if is_valid:
                passed.append(cand)
                results.append(
                    FilterResult(
                        passed=True,
                        filter_name=self.filter_name,
                        details={
                            "min_distance": min_dist,
                            "threshold": self.min_distance_threshold,
                        },
                    )
                )
            else:
                results.append(
                    FilterResult(
                        passed=False,
                        filter_name=self.filter_name,
                        rejection_reason=f"Min distance {min_dist:.3f} Å < threshold {self.min_distance_threshold:.3f} Å",
                        details={
                            "min_distance": min_dist,
                            "threshold": self.min_distance_threshold,
                        },
                    )
                )

        return passed, results

    def get_state(self) -> dict[str, Any]:
        """Return the current state of the filter."""
        return {"min_distance_threshold": self.min_distance_threshold}

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the filter (no-op for min distance filter)."""
