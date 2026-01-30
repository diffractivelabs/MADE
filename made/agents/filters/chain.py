from typing import Any

from pymatgen.core.structure import Structure

from made.agents.base import Filter, FilterResult


class FilterChain(Filter):
    """Chain multiple filters together, applying them sequentially.

    Filters are applied in the order they are provided. Each filter receives
    the output of the previous filter as its input.
    """

    filter_name = "FilterChain"

    def __init__(self, filters: list[Filter]) -> None:
        """Initialize the filter chain.

        Args:
            filters: List of filters to apply in sequence.
        """
        if not filters:
            raise ValueError("FilterChain requires at least one filter")
        self.filters = filters

    def _filter_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Apply all filters and collect detailed results from each.

        Returns aggregated results showing which filter(s) rejected each structure.
        A structure passes only if it passes ALL filters in the chain.
        """
        # Track results per structure index across all filters
        all_results_by_idx: dict[int, list[FilterResult]] = {
            i: [] for i in range(len(candidates))
        }

        current_candidates = candidates
        current_indices = list(range(len(candidates)))  # Track original indices

        for filter_obj in self.filters:
            # Get results from this filter
            passed, filter_results = filter_obj.filter(
                current_candidates, state, return_results=True
            )

            # Record results for structures that were tested
            for idx, result in zip(current_indices, filter_results, strict=True):
                all_results_by_idx[idx].append(result)

            # Track which structures pass to next filter
            passed_set = {id(s) for s in passed}
            new_candidates = []
            new_indices = []
            for cand, idx in zip(current_candidates, current_indices, strict=True):
                if id(cand) in passed_set:
                    new_candidates.append(cand)
                    new_indices.append(idx)

            current_candidates = new_candidates
            current_indices = new_indices

        # Build final results
        final_passed = []
        final_results = []
        passed_indices = set(current_indices)

        for i, cand in enumerate(candidates):
            filter_results = all_results_by_idx[i]

            if i in passed_indices:
                final_passed.append(cand)
                final_results.append(
                    FilterResult(
                        passed=True,
                        filter_name=self.filter_name,
                        details={
                            "filter_results": [
                                {"filter": r.filter_name, "passed": r.passed}
                                for r in filter_results
                            ]
                        },
                    )
                )
            else:
                rejection_reasons = [
                    f"{r.filter_name}: {r.rejection_reason}"
                    for r in filter_results
                    if not r.passed and r.rejection_reason
                ]
                final_results.append(
                    FilterResult(
                        passed=False,
                        filter_name=self.filter_name,
                        rejection_reason=" | ".join(rejection_reasons)
                        if rejection_reasons
                        else "Rejected",
                        details={
                            "filter_results": [
                                {
                                    "filter": r.filter_name,
                                    "passed": r.passed,
                                    "reason": r.rejection_reason,
                                }
                                for r in filter_results
                            ]
                        },
                    )
                )

        return final_passed, final_results

    def get_state(self) -> dict[str, Any]:
        """Return the current state of all filters in the chain."""
        return {"filters": [f.get_state() for f in self.filters]}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load the state of all filters in the chain."""
        if "filters" in state:
            for filter_obj, filter_state in zip(
                self.filters, state["filters"], strict=True
            ):
                if hasattr(filter_obj, "load_state"):
                    filter_obj.load_state(filter_state)

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of all filters in the chain."""
        for filter_obj in self.filters:
            filter_obj.update_state(state)
