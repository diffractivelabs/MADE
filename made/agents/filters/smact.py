from typing import Any

from pymatgen.core.structure import Structure

from made.agents.base import Filter, FilterResult
from made.evaluation.metrics import is_smact_valid


class SMACTValidityFilter(Filter):
    """Filter that keeps only structures valid according to SMACT validity checks.

    This filter removes structures that fail chemical sanity checks such as
    charge balance and electronegativity constraints.
    """

    filter_name = "SMACTValidity"

    def __init__(self) -> None:
        """Initialize the SMACT validity filter."""

    def _filter_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Filter with detailed SMACT validity results."""
        passed = []
        results = []

        for cand in candidates:
            is_valid = is_smact_valid(cand)
            comp = cand.composition.reduced_formula

            if is_valid:
                passed.append(cand)
                results.append(
                    FilterResult(
                        passed=True,
                        filter_name=self.filter_name,
                        details={"composition": comp},
                    )
                )
            else:
                results.append(
                    FilterResult(
                        passed=False,
                        filter_name=self.filter_name,
                        rejection_reason=f"SMACT invalid: {comp} fails charge balance or electronegativity checks",
                        details={"composition": comp},
                    )
                )

        return passed, results

    def get_state(self) -> dict[str, Any]:
        """Return the current state of the filter."""
        return {}

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the filter (no-op for SMACT filter)."""
