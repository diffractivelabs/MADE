"""
Data utilities for generating and serving Phase Diagram ground-truth datasets.
"""

import logging
import os

from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Structure

from made.utils.convex_hull_utils import (
    extract_structure_from_entry,
    safe_e_above_hull,
)

logger = logging.getLogger(__name__)


class PhaseDiagramDataset:
    """
    Ground-truth phase diagram for a chemical system pulled from Materials Project.

    Provides the Materials Project `PhaseDiagram` and convenience accessors for
    elemental references and stable entries.

    IMPORTANT: This dataset is intentionally unfiltered (except for thermo_types).
    It provides the raw Materials Project data. Task-specific filtering (SMACT,
    max_stoichiometry, energy-above-hull) is applied at the Environment level to
    define the "effective ground truth" for discovery tasks. This separation allows
    the same dataset to be reused for different task difficulties.
    """

    def __init__(
        self,
        elements: list[str],
        thermo_types: list[str] | None = None,
    ):
        self.elements = list(elements)
        self.thermo_types = thermo_types or ["GGA_GGA+U"]
        self.dataset = self._load_dataset()
        self._ground_truth_pd: PhaseDiagram = self.dataset

    def _load_dataset(self) -> PhaseDiagram:
        api_key = os.environ.get("MATERIALS_PROJECT_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Environment variable MATERIALS_PROJECT_API_KEY must be set with a valid Materials Project API key."
            )

        with MPRester(api_key) as mpr:
            entries = mpr.get_entries_in_chemsys(
                elements=self.elements,
                additional_criteria={"thermo_types": self.thermo_types},
            )

            # Ensure all entries have structures properly attached in attributes
            entries_with_structures = []
            for entry in entries:
                if hasattr(entry, "structure") and entry.structure is not None:
                    # Create new PDEntry with structure in attribute for consistency with oracle entries
                    new_entry = PDEntry(
                        composition=entry.composition,
                        energy=entry.energy,
                        attribute={
                            "structure": entry.structure.as_dict(),
                        },
                    )
                    entries_with_structures.append(new_entry)
                else:
                    logger.warning(f"Entry {entry} does not have a structure, skipping")

            logger.info(
                f"Loaded {len(entries_with_structures)} entries with structures from Materials Project"
            )
            return PhaseDiagram(entries_with_structures)

    def get_ground_truth_pd(self) -> PhaseDiagram:
        return self._ground_truth_pd

    def get_all_entries(self):
        return self._ground_truth_pd.all_entries

    def get_elemental_reference_entries(self) -> list[PDEntry]:
        # Pymatgen exposes elemental references for the PD
        return list(self._ground_truth_pd.el_refs.values())

    def get_stable_entries(self, epsilon: float | None = None) -> list[PDEntry]:
        """Return entries that are on or within epsilon of the convex hull.

        Args:
            epsilon: Energy-above-hull threshold (eV/atom). If None or <= 0,
                returns strictly stable entries on the hull.
        """
        if epsilon is None or epsilon <= 0:
            return list(self._ground_truth_pd.stable_entries)

        near_stable: list[PDEntry] = []
        for entry in self._ground_truth_pd.all_entries:
            try:
                if float(safe_e_above_hull(self._ground_truth_pd, entry)) <= float(
                    epsilon
                ):
                    near_stable.append(entry)
            except Exception:
                # If e_above_hull fails for any entry, skip it for robustness
                continue
        return near_stable

    def get_stable_structures(self, epsilon: float | None = None) -> list[Structure]:
        """Get structures from (near-)stable entries for oracle evaluation.

        Args:
            epsilon: Energy-above-hull threshold (eV/atom). If None or <= 0,
                uses strictly stable entries.
        """
        structures = []
        for entry in self.get_stable_entries(epsilon=epsilon):
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                structures.append(structure)
            else:
                raise ValueError(f"Entry {entry} does not have an accessible structure")
        return structures

    def get_elemental_structures(self) -> list[Structure]:
        """Get structures from elemental reference entries for oracle evaluation."""
        structures = []
        for entry in self.get_elemental_reference_entries():
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                structures.append(structure)
            else:
                raise ValueError(
                    f"Elemental entry {entry} does not have an accessible structure"
                )
        return structures

    def get_stable_entries_with_structures(
        self, epsilon: float | None = None
    ) -> list[tuple[PDEntry, Structure]]:
        """Get both PDEntry and Structure objects for (near-)stable entries.

        Args:
            epsilon: Energy-above-hull threshold (eV/atom). If None or <= 0,
                uses strictly stable entries.
        """
        entries_with_structures = []
        for entry in self.get_stable_entries(epsilon=epsilon):
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                entries_with_structures.append((entry, structure))
            else:
                raise ValueError(f"Entry {entry} does not have an accessible structure")
        return entries_with_structures

    def get_elemental_entries_with_structures(self) -> list[tuple[PDEntry, Structure]]:
        """Get both PDEntry and Structure objects for elemental reference entries."""
        entries_with_structures = []
        for entry in self.get_elemental_reference_entries():
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                entries_with_structures.append((entry, structure))
            else:
                raise ValueError(
                    f"Elemental entry {entry} does not have an accessible structure"
                )
        return entries_with_structures
