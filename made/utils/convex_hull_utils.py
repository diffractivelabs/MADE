"""Utility functions for convex hull calculations and phase diagram operations."""

import logging
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


def safe_e_above_hull(phase_diagram: PhaseDiagram, entry: PDEntry) -> float:
    """Return e_above_hull with robust handling for elemental duplicates.

    If a valid decomposition is not found (commonly when the proposal is an
    elemental entry conflicting with the current elemental reference), warn
    and fall back to a manual computation for elemental compositions.

    This function is side-effect-free and does not modify the phase diagram.

    Args:
        phase_diagram: The phase diagram to compute e_above_hull against.
        entry: The PDEntry to compute e_above_hull for.

    Returns:
        Energy above hull in eV/atom. Returns float("inf") if computation fails
        for non-elemental compositions, or if no elemental reference is available.
    """
    try:
        return phase_diagram.get_e_above_hull(entry)
    except ValueError as exc:  # pragma: no cover - resilience path
        comp = entry.composition
        logger.warning(
            f"Phase diagram decomposition failed for {comp}: {exc}. "
            "Falling back to safe handling."
        )

        # Only special-case single-element compositions
        if len(comp.elements) == 1:
            elem = list(comp.elements)[0]
            # Try to obtain the current elemental reference from the PD
            try:
                ref_entry = phase_diagram.el_refs[elem]
            except Exception:
                # If we cannot obtain a reference, treat as undefined
                return float("inf")

            entry_e_pa = entry.energy_per_atom
            ref_e_pa = ref_entry.energy_per_atom

            # Compute a non-negative difference relative to the reference
            return max(0.0, entry_e_pa - ref_e_pa)

        # For non-elemental cases where decomposition failed, return +inf to mark unstable
        return float("inf")


def formulas_within_epsilon(phase_diagram: PhaseDiagram, epsilon: float) -> set[str]:
    """Get formulas of entries within epsilon of the convex hull for a given PD.

    Args:
        phase_diagram: The phase diagram to analyze.
        epsilon: Energy tolerance in eV/atom.

    Returns:
        Set of reduced alphabetical formulas for entries within epsilon of the hull.
    """
    formulas: set[str] = set()
    for e in phase_diagram.all_entries:
        try:
            if float(safe_e_above_hull(phase_diagram, e)) <= float(epsilon):
                formulas.add(e.composition.reduced_composition.alphabetical_formula)
        except Exception:
            continue
    return formulas


def structure_result_to_entry(
    structure: Structure, oracle_result: dict[str, Any]
) -> PDEntry:
    """Create a PDEntry from a structure and oracle result.

    Expected oracle outputs:
    - energy_per_atom (float) OR energy (float): If both present, energy takes precedence.

    Args:
        structure: The structure to create an entry for.
        oracle_result: Dictionary containing energy information from oracle.

    Returns:
        A PDEntry with the structure attached as an attribute.

    Raises:
        ValueError: If oracle result doesn't contain required energy information.
    """
    composition: Composition = structure.composition
    num_atoms = structure.num_sites

    if "energy" in oracle_result and isinstance(oracle_result["energy"], (int | float)):
        total_energy = float(oracle_result["energy"])
    elif "energy_per_atom" in oracle_result:
        total_energy = float(oracle_result["energy_per_atom"]) * float(num_atoms)
    else:
        raise ValueError(
            "Oracle result must contain 'energy' or 'energy_per_atom' to build a PDEntry."
        )

    # Attach structure as attribute for later access/serialization
    return PDEntry(
        composition=composition,
        energy=total_energy,
        attribute={"structure": structure.as_dict()},
    )


def extract_structure_from_entry(entry: PDEntry) -> Structure | None:
    """Extract structure from PDEntry.

    Args:
        entry: The PDEntry to extract structure from.

    Returns:
        The Structure if found in entry attributes, None otherwise.
    """
    if (
        hasattr(entry, "attribute")
        and entry.attribute is not None
        and "structure" in entry.attribute
    ):
        structure_dict = entry.attribute["structure"]
        return Structure.from_dict(structure_dict)

    return None
