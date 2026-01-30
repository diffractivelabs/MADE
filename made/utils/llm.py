"""
Shared utilities for LLM-based components
"""

import random
from collections import Counter
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Structure

from made.utils.convex_hull_utils import safe_e_above_hull


def pd_from_state(state: dict[str, Any]) -> PhaseDiagram | None:
    entries_raw = state.get("phase_diagram_all_entries", [])
    if not entries_raw:
        return None
    entries = [PDEntry.from_dict(e) for e in entries_raw]
    return PhaseDiagram(entries)


def summarize_context_for_llm(
    state: dict[str, Any],
    stability_tolerance: float | None = None,
    include_structures: bool = True,
    include_composition_counter: bool = True,
    include_recent_trial: bool = True,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Summarize experiment context for LLM consumption.

    Args:
        state: Current experiment state
        stability_tolerance: e_above_hull threshold for stability
        include_structures: Include full structure info (pymatgen string representation)
        include_composition_counter: Include composition trial counts
        include_recent_trial: Include most recent trial info
        max_entries: Maximum number of entries to include. If exceeded, randomly
            samples entries while always including stable/metastable ones first.
    """
    pd = pd_from_state(state)
    elements = state.get("elements", [])

    # Get stability tolerance from parameter or state
    if stability_tolerance is None:
        stability_tolerance = state.get("stability_tolerance", 1e-8)

    if pd is None:
        return {
            "query_count": state.get("query_count", 0),
            "chemical_space": elements,
            "stability_tolerance": stability_tolerance,
        }

    entries = sorted(pd.all_entries, key=lambda e: float(safe_e_above_hull(pd, e)))

    # Apply max_entries limit with random sampling if needed
    total_entries = len(entries)
    sampled = False
    if max_entries is not None and len(entries) > max_entries:
        # Prioritize stable/metastable entries, then randomly sample the rest
        stable_entries = [
            e for e in entries if float(safe_e_above_hull(pd, e)) <= stability_tolerance
        ]
        unstable_entries = [
            e for e in entries if float(safe_e_above_hull(pd, e)) > stability_tolerance
        ]

        # Always include all stable entries (up to max_entries)
        if len(stable_entries) >= max_entries:
            entries = random.sample(stable_entries, max_entries)
        else:
            # Fill remaining slots with random unstable entries
            remaining_slots = max_entries - len(stable_entries)
            sampled_unstable = random.sample(
                unstable_entries, min(remaining_slots, len(unstable_entries))
            )
            entries = stable_entries + sampled_unstable

        # Re-sort by e_above_hull
        entries = sorted(entries, key=lambda e: float(safe_e_above_hull(pd, e)))
        sampled = True

    entries_summary = []
    for e in entries:
        e_hull = float(safe_e_above_hull(pd, e))
        is_stable_metastable = e_hull <= stability_tolerance

        entry_summary: dict[str, Any] = {
            "full_formula": e.composition.formula,  # Full formula (e.g., Li4O2)
            "reduced_formula": e.composition.reduced_formula,  # Reduced (e.g., Li2O)
            "energy_per_atom": f"{e.energy_per_atom:.4f}",
            "formation_energy_per_atom": f"{pd.get_form_energy_per_atom(e):.4f}",
            "e_above_hull": f"{e_hull:.4f}",
            "is_stable_or_metastable": is_stable_metastable,
        }
        if include_structures:
            # Use pymatgen's string representation for complete structural info
            entry_summary["structure"] = str(
                Structure.from_dict(e.attribute["structure"])
            )
        entries_summary.append(entry_summary)

    context: dict[str, Any] = {
        "query_count": state.get("query_count", 0),
        "chemical_space": elements,
        "stability_tolerance": stability_tolerance,
        "state": {
            "evaluated_entries_summary": entries_summary,
            "total_entries": total_entries,
        },
    }

    # Note if we sampled
    if sampled:
        context["state"]["entries_sampled"] = True
        context["state"]["entries_shown"] = len(entries)

    if include_composition_counter:
        # Use full entries list for accurate counts (not sampled)
        all_entries = pd.all_entries
        # Count by reduced formula (phase diagram points)
        reduced_counts = Counter(
            e.composition.reduced_formula for e in all_entries
        ).most_common()
        context["state"]["reduced_composition_trial_counts"] = dict(reduced_counts)

        # Also count by full formula (different structure sizes)
        full_counts = Counter(e.composition.formula for e in all_entries).most_common()
        context["state"]["full_composition_trial_counts"] = dict(full_counts)

    last = state.get("last_observation")
    if last and include_recent_trial:
        structure = Structure.from_dict(state["last_observation"]["proposal"])
        e_hull_last = last.get("e_above_hull", float("inf"))
        context["recent_trial"] = {
            "full_formula": structure.composition.formula,
            "reduced_formula": structure.composition.reduced_formula,
            "formation_energy_per_atom": f"{last.get('formation_energy_per_atom'):.4f}",
            "energy_per_atom": f"{last.get('energy_per_atom'):.4f}",
            "e_above_hull": f"{e_hull_last:.4f}",
            "is_stable_or_metastable": e_hull_last <= stability_tolerance,
            "is_newly_discovered": bool(last.get("is_newly_discovered", False)),
        }

    return context


def summarize_candidates_for_llm(
    candidates: list[Structure],
    include_structures: bool = True,
    max_candidates: int | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Summarize candidate structures for LLM consumption.

    Args:
        candidates: List of candidate structures
        include_structures: Include full structure info (pymatgen string representation)
        max_candidates: Maximum number of candidates to include. If exceeded,
            randomly samples candidates.

    Returns:
        Tuple of (summaries, original_indices) where:
        - summaries: List of candidate summaries for LLM
        - original_indices: List mapping summary index to original candidate index
            (needed to map LLM selection back to original candidates)
    """
    # Build list of (original_index, structure) pairs
    indexed_candidates = list(enumerate(candidates))

    # Apply max_candidates limit with random sampling if needed
    total_candidates = len(candidates)
    sampled = False
    if max_candidates is not None and len(indexed_candidates) > max_candidates:
        indexed_candidates = random.sample(indexed_candidates, max_candidates)
        sampled = True

    summaries = []
    original_indices = []
    for display_idx, (orig_idx, s) in enumerate(indexed_candidates):
        item: dict[str, Any] = {
            "index": display_idx,  # Index in the displayed list (for LLM selection)
            "full_formula": s.composition.formula,
            "reduced_formula": s.composition.reduced_formula,
            "num_sites": len(s),
        }
        if include_structures:
            # Use pymatgen's string representation for complete structural info
            item["structure"] = str(s)
        summaries.append(item)
        original_indices.append(orig_idx)

    # Add metadata about sampling
    if sampled and summaries:
        summaries[0]["_note"] = (
            f"Showing {len(summaries)} of {total_candidates} candidates (randomly sampled)"
        )

    return summaries, original_indices
