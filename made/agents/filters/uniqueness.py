import logging
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

from made.agents.base import Filter, FilterResult
from made.utils.convex_hull_utils import extract_structure_from_entry

logger = logging.getLogger(__name__)


class UniquenessFilter(Filter):
    """Filter that keeps only structures that are unique compared to already attempted structures.

    This filter removes structures that match (via StructureMatcher) any structure that has
    already been attempted. All attempted structures are found in the observed phase diagram
    entries in the state.

    Args:
        ltol: Fractional length tolerance for structure matching (default: 0.2).
        stol: Site tolerance for structure matching (default: 0.3).
        angle_tol: Angle tolerance for structure matching in degrees (default: 5.0).
        primitive_cell: If True, use primitive cell for matching (default: True).
    """

    filter_name = "Uniqueness"

    def __init__(
        self,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        primitive_cell: bool = True,
    ) -> None:
        """Initialize the uniqueness filter."""
        self.structure_matcher = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
        )

    def _filter_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Filter with detailed uniqueness results."""
        reference_structures = self._get_reference_structures(state)

        if not reference_structures:
            results = [
                FilterResult(
                    passed=True,
                    filter_name=self.filter_name,
                    details={"num_references": 0},
                )
                for _ in candidates
            ]
            return candidates, results

        passed = []
        results = []

        for candidate in candidates:
            matching_ref = None
            for ref_struct in reference_structures:
                try:
                    if self.structure_matcher.fit(candidate, ref_struct):
                        matching_ref = ref_struct
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {candidate.composition} and {ref_struct.composition}: {e}"
                    )

            if matching_ref is None:
                passed.append(candidate)
                results.append(
                    FilterResult(
                        passed=True,
                        filter_name=self.filter_name,
                        details={"num_references": len(reference_structures)},
                    )
                )
            else:
                ref_comp = matching_ref.composition.reduced_formula
                cand_comp = candidate.composition.reduced_formula
                results.append(
                    FilterResult(
                        passed=False,
                        filter_name=self.filter_name,
                        rejection_reason=f"Matches existing structure ({cand_comp} ~ {ref_comp})",
                        details={
                            "num_references": len(reference_structures),
                            "matching_composition": ref_comp,
                        },
                    )
                )

        return passed, results

    def _get_reference_structures(self, state: dict[str, Any]) -> list[Structure]:
        """Extract reference structures from state."""
        reference_structures = []

        phase_diagram_entries = state.get("phase_diagram_all_entries", [])
        if not phase_diagram_entries:
            return reference_structures

        entries = [PDEntry.from_dict(e) for e in phase_diagram_entries]
        for entry in entries:
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                reference_structures.append(structure)

        return reference_structures

    def get_state(self) -> dict[str, Any]:
        """Return the current state of the filter."""
        return {}

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the filter (no-op for uniqueness filter)."""
