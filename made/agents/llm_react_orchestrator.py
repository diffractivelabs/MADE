"""
LLM ReAct Orchestrator Agent - Flexible LLM agent that orchestrates generators, filters, and scorers.

This agent uses DSPy's ReAct to decide what actions to take at each iteration:
- Generate structures using various generators (with filtering)
- Score structures using various scorers
- Query buffer for candidates (topk, diverse, by composition)
- Select structures for oracle evaluation

Key features:
- Maintains a buffer of pre-validated structures from generators
- All generated structures are filtered before adding to buffer
- Uniqueness filter is always re-run on new generations
- Full history of oracle evaluations available to LLM
- Caching of structure hashes to avoid duplicate processing
- Flexible buffer queries (topk, bottomk, diverse compositions)
"""

import logging
from typing import Any

import dspy
import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from made.agents.base import Agent, Filter, Generator, Plan, Scorer
from made.utils.structure_hash import structure_hash

logger = logging.getLogger(__name__)


# ============================================================================
# DSPy Signature for ReAct Orchestration (default, can be overridden via config)
# ============================================================================


class OrchestratorReActSignature(dspy.Signature):
    """
    You are an autonomous materials discovery agent.

    OBJECTIVE: Find as many NOVEL, UNIQUE, STABLE (or metastable) structures as possible. Use the available tools, then select ONE composition + structure for oracle evaluation.

    - Structures with e_above_hull <= stability_tolerance are stable/metastable (SUCCESS!)
    - Entries marked [STABLE, NOVEL] in evaluation_history are successful discoveries
    - We want to MAXIMIZE the number of novel stable structures found

    IMPORTANT: Different structures for the SAME composition can have DIFFERENT stabilities.
    - One unstable structure for a composition does NOT mean all structures for that composition are unstable.
    - Generators produce many different structures for the same composition

    UNIT CELL SIZE MATTERS:
    - Compositions are stored by REDUCED formula (e.g., Li2O)
    - But you can generate different UNIT CELL SIZES: Li2O, Li4O2, Li6O3, etc.
    - These occupy the same position on the phase diagram but are different structures
    - Different unit cell sizes often have different stabilities

    BUFFER ORGANIZATION:
    - Buffer is organized by reduced formula: {composition: [structures]}
    - Each structure shows its full formula (unit cell size) and index
    - Selection is two-step: pick composition, then pick structure index

    WORKFLOW:
    1. Decide which composition(s) to explore based on evaluation history
    2. Generate/create candidate structures for those compositions
    3. Score candidates if needed to prioritize within each composition
    4. List compositions and query structures to decide what to evaluate
    5. Select ONE composition + structure for oracle evaluation

    STRATEGY GUIDANCE:
    - If buffer empty/small: Generate more structures (diverse compositions OR more for promising ones)
    - If buffer has candidates: Score them, query compositions, select best
    - Balance exploration (new compositions) vs exploitation (more structures for successful compositions)
    """

    chemical_system: str = dspy.InputField(desc="Allowed elements (e.g., 'Li, O')")
    stability_tolerance: float = dspy.InputField(
        desc="e_above_hull threshold for stability"
    )
    max_stoichiometry: int = dspy.InputField(desc="Maximum atoms per structure")
    buffer_summary: str = dspy.InputField(
        desc="Summary of unevaluated candidate buffer"
    )
    evaluation_history: str = dspy.InputField(
        desc="History of oracle evaluations with results"
    )
    known_stable_materials: str = dspy.InputField(
        desc="Known stable materials from phase diagram"
    )

    answer: str = dspy.OutputField(
        desc="Summary of actions taken and final selection rationale"
    )


# ============================================================================
# Tool Classes for the Orchestrator
# ============================================================================


class OrchestratorTools:
    """Tools for the LLM ReAct Orchestrator to use.

    Buffer is organized by composition: {composition: [structure_entries]}
    This enables natural two-step selection: pick composition, then pick structure.
    """

    def __init__(
        self,
        generators: dict[str, Generator],
        static_filter: Filter | None,
        uniqueness_filter: Filter | None,
        scorers: dict[str, Scorer],
        chemical_system_elements: list[str],
        max_stoichiometry: int,
        buffer: dict[
            str, list[dict[str, Any]]
        ],  # composition -> list of structure entries
        structure_cache: dict[str, dict[str, Any]],
        state: dict[str, Any],
    ):
        self.generators = generators
        self.static_filter = static_filter
        self.uniqueness_filter = uniqueness_filter
        self.scorers = scorers
        self.chemical_system_elements = chemical_system_elements
        self.max_stoichiometry = max_stoichiometry
        self.buffer = buffer  # Now a dict: composition -> list of entries
        self.structure_cache = structure_cache
        self.state = state
        self._selected_structure: Structure | None = None
        self._selection_reason: str = ""

    def generate_structures(
        self,
        generator_name: str,
        compositions: str = "",
        num_candidates: int = 10,
    ) -> str:
        """Generate new candidate structures using a specified generator.

        Structures are automatically filtered (distance, SMACT, etc.) before adding to buffer.
        Uniqueness filter is always applied to remove duplicates of existing structures.

        Args:
            generator_name: Name of generator to use - one of 'chemeleon' (a generative model trained on stable structures), 'random' (a random structure generator)
            compositions: Comma-separated compositions with unit cell size (e.g., 'Li2O, Li4O2, LiO2').
                         Different unit cell sizes (Li2O vs Li4O2) represent different structure sizes
                         but occupy the same position on the phase diagram (both reduce to Li2O).
                         Generators create different structures for different unit cell sizes.
            num_candidates: Number of candidates to generate per composition

        Returns:
            Summary of generated and filtered structures
        """
        if generator_name not in self.generators:
            available = list(self.generators.keys())
            return (
                f"Error: Generator '{generator_name}' not found. Available: {available}"
            )

        generator = self.generators[generator_name]

        # Parse compositions
        comp_list = []
        if compositions.strip():
            for comp_str in compositions.split(","):
                comp_str = comp_str.strip()
                if comp_str:
                    try:
                        comp = Composition(comp_str)
                        for el in comp.elements:
                            if str(el) not in self.chemical_system_elements:
                                return f"Error: Element {el} not in allowed elements {self.chemical_system_elements}"
                        comp_list.append(comp)
                    except Exception as e:
                        return f"Error parsing composition '{comp_str}': {e}"

        if not comp_list:
            logger.warning(f"No compositions provided for generator {generator_name}")

            return f"No compositions provided for generator {generator_name}"

        plan = Plan(
            compositions=comp_list,
            num_candidates=num_candidates,
            constraints={
                "elements": self.chemical_system_elements,
                "max_stoichiometry": self.max_stoichiometry,
            },
        )

        try:
            structures = generator.generate(plan, self.state)

            if not structures:
                return f"Generator {generator_name} produced no structures."

            # Filter and add to buffer
            added_count = 0
            filtered_out = {"duplicate": 0, "cached": 0, "static_filter": 0}

            for structure in structures:
                struct_hash = structure_hash(structure)

                # Check cache
                if struct_hash in self.structure_cache:
                    filtered_out["cached"] += 1
                    continue

                # Run static filter chain
                if self.static_filter is not None:
                    try:
                        passed, _ = self.static_filter.filter(
                            [structure], self.state, return_results=True
                        )
                        if not passed:
                            filtered_out["static_filter"] += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Static filter failed: {e}")
                        filtered_out["static_filter"] += 1
                        continue

                # Run uniqueness filter (always re-run)
                if self.uniqueness_filter is not None:
                    try:
                        passed, _ = self.uniqueness_filter.filter(
                            [structure], self.state, return_results=True
                        )
                        if not passed:
                            filtered_out["duplicate"] += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Uniqueness filter failed: {e}")

                # Add to buffer and cache
                comp = structure.composition.reduced_formula
                full_formula = structure.composition.formula.replace(" ", "")
                entry = {
                    "structure": structure,
                    "hash": struct_hash,
                    "composition": comp,  # Reduced formula (e.g., Li2O)
                    "full_formula": full_formula,  # Full formula (e.g., Li4O2)
                    "source": generator_name,
                    "scores": {},
                    "num_sites": len(structure),
                }
                # Add to composition-based buffer (keyed by reduced formula)
                if comp not in self.buffer:
                    self.buffer[comp] = []
                self.buffer[comp].append(entry)
                self.structure_cache[struct_hash] = entry
                added_count += 1

            # Build summary
            total_count = sum(len(entries) for entries in self.buffer.values())
            msg_parts = [f"Generated {len(structures)} using {generator_name}."]
            msg_parts.append(
                f"Added {added_count} to buffer ({total_count} total, {len(self.buffer)} compositions)."
            )

            if filtered_out["cached"]:
                msg_parts.append(f"Cached: {filtered_out['cached']}.")
            if filtered_out["duplicate"]:
                msg_parts.append(f"Duplicates: {filtered_out['duplicate']}.")
            if filtered_out["static_filter"]:
                msg_parts.append(f"Filter failures: {filtered_out['static_filter']}.")

            msg = " ".join(msg_parts)
            logger.info(f"[Tool] {msg}")
            return msg

        except Exception as e:
            logger.error(f"[Tool] Generation failed: {e}")
            return f"Error during generation: {e}"

    def score_buffer(self, scorer_name: str, composition: str = "") -> str:
        """Score candidates in the buffer using a specified scorer.

        Args:
            scorer_name: Name of scorer to use (e.g., 'diversity', 'oracle')
            composition: Optional composition to score (if empty, scores all)

        Returns:
            Summary of scoring results with top candidates per composition
        """
        if scorer_name not in self.scorers:
            available = list(self.scorers.keys())
            return f"Error: Scorer '{scorer_name}' not found. Available: {available}"

        if not self.buffer:
            return "Buffer is empty, nothing to score."

        scorer = self.scorers[scorer_name]

        try:
            # Collect structures to score
            to_score_comps = [composition] if composition else list(self.buffer.keys())
            total_scored = 0

            for comp in to_score_comps:
                if comp not in self.buffer:
                    continue

                entries = self.buffer[comp]
                structures = [e["structure"] for e in entries]

                scores, results = scorer.score_candidates(
                    structures, self.state, return_results=True
                )

                # Store scores in entries
                for entry, score in zip(entries, scores, strict=True):
                    entry["scores"][scorer_name] = score

                # Sort entries by score (descending) within composition
                self.buffer[comp].sort(
                    key=lambda e: e["scores"].get(scorer_name, float("-inf")),
                    reverse=True,
                )
                total_scored += len(entries)

            # Build summary showing top structures per composition
            lines = [f"Scored {total_scored} structures with '{scorer_name}'."]
            lines.append("Top structures per composition:")
            for comp in sorted(to_score_comps):
                if comp not in self.buffer:
                    continue
                entries = self.buffer[comp]
                if entries and scorer_name in entries[0].get("scores", {}):
                    top_score = entries[0]["scores"][scorer_name]
                    lines.append(
                        f"  {comp}: {len(entries)} structures, best score={top_score:.4f}"
                    )

            msg = "\n".join(lines)
            logger.info(f"[Tool] {msg}")
            return msg

        except Exception as e:
            logger.error(f"[Tool] Scoring failed: {e}")
            return f"Error during scoring: {e}"

    def list_compositions(
        self,
        k: int = 10,
        mode: str = "top",
        scorer_name: str = "",
    ) -> str:
        """List compositions in the buffer with structure counts and optional scoring.

        Use this to see what compositions are available and their characteristics.

        Args:
            k: Number of compositions to show
            mode: Query mode:
                - 'top': Highest scoring compositions (requires scorer_name, default)
                - 'bottom': Lowest scoring compositions (requires scorer_name)
                - 'random': Random sample (no scorer needed)
                - 'count': By structure count (most structures first)
            scorer_name: Scorer to rank compositions by best structure score (required for top/bottom)

        Returns:
            List of compositions with counts and optional best scores

        Examples:
            - list_compositions(k=5, mode='count')
              → Top 5 compositions by structure count
            - list_compositions(k=10, mode='top', scorer_name='oracle')
              → Top 10 compositions by best oracle score
            - list_compositions(k=3, mode='bottom', scorer_name='diversity')
              → Bottom 3 (least diverse) compositions
            - list_compositions(k=5, mode='random')
              → Random 5 compositions
        """
        if not self.buffer:
            return "Buffer is empty."

        comp_info = []
        for comp, entries in self.buffer.items():
            info = {
                "composition": comp,
                "count": len(entries),
                "best_score": None,
            }

            if scorer_name and entries:
                # Get best score for this composition
                scores = [
                    e["scores"].get(scorer_name)
                    for e in entries
                    if scorer_name in e.get("scores", {})
                ]
                if scores:
                    info["best_score"] = max(scores)

            comp_info.append(info)

        # Sort or sample based on mode
        if mode in ["top", "bottom"]:
            if not scorer_name:
                return f"Error: mode '{mode}' requires scorer_name."
            # Check if any composition has scores
            if not any(info["best_score"] is not None for info in comp_info):
                return f"No compositions scored with '{scorer_name}'. Run score_buffer first."
            reverse = mode == "top"
            comp_info.sort(
                key=lambda x: x["best_score"]
                if x["best_score"] is not None
                else float("-inf"),
                reverse=reverse,
            )
        elif mode == "random":
            np.random.shuffle(comp_info)
        else:  # mode == "count" or any other value
            comp_info.sort(key=lambda x: x["count"], reverse=True)

        # Format output
        total_structures = sum(info["count"] for info in comp_info)
        mode_desc = f"by {mode}" if mode != "count" else "by count"
        if mode in ["top", "bottom"] and scorer_name:
            mode_desc = f"{mode} by {scorer_name}"

        lines = [
            f"Buffer: {total_structures} structures across {len(self.buffer)} compositions"
        ]
        lines.append(f"Showing {min(k, len(comp_info))} compositions ({mode_desc}):")

        for i, info in enumerate(comp_info[:k]):
            score_str = (
                f", best_{scorer_name}={info['best_score']:.4f}"
                if info["best_score"] is not None
                else ""
            )
            lines.append(
                f"  {i + 1}. {info['composition']}: {info['count']} structures{score_str}"
            )

        msg = "\n".join(lines)
        logger.info(f"[Tool] {msg}")
        return msg

    def query_structures(
        self,
        composition: str,
        k: int = 5,
        mode: str = "top",
        scorer_name: str = "",
        include_structure_details: bool = False,
    ) -> str:
        """Query structures within a specific composition.

        Retrieve structures to help decide which one to select for evaluation.
        The index shown can be used directly in select_for_evaluation.

        Args:
            composition: Composition to query (e.g., 'Li2O')
            k: Number of structures to return
            mode: Query mode:
                - 'top': Highest scores (requires scorer_name, default)
                - 'bottom': Lowest scores (requires scorer_name)
                - 'random': Random sample (no scorer needed)
                - 'all': All structures in order (no sorting)
            scorer_name: Scorer to use for ranking (required for top/bottom modes)
            include_structure_details: If True, includes full structure (lattice, species, positions)

        Returns:
            List of structures with indices, scores, and optional structural details

        Examples:
            - query_structures('Li2O', k=3, mode='top', scorer_name='oracle')
              → Top 3 structures for Li2O by oracle score
            - query_structures('Li2O', k=5, mode='bottom', scorer_name='diversity')
              → Bottom 5 (least diverse) Li2O structures
            - query_structures('LiO2', k=2, mode='random')
              → Random 2 structures from LiO2
            - query_structures('Li2O', k=10, mode='all', include_structure_details=True)
              → All Li2O structures with lattice info
        """
        if not self.buffer:
            return "Buffer is empty."

        comp = composition.strip()
        if comp not in self.buffer:
            available = list(self.buffer.keys())[:10]
            return f"No structures for composition '{comp}'. Available: {available}"

        entries = self.buffer[comp].copy()

        # Sort or sample based on mode
        if mode in ["top", "bottom"]:
            if not scorer_name:
                return f"Error: mode '{mode}' requires scorer_name."
            if not entries[0].get("scores", {}).get(scorer_name):
                return f"Structures not scored with '{scorer_name}'. Run score_buffer first."
            # Sort by score
            reverse = mode == "top"
            sorted_entries = sorted(
                entries,
                key=lambda e: e["scores"].get(scorer_name, float("-inf")),
                reverse=reverse,
            )
        elif mode == "random":
            sorted_entries = entries.copy()
            np.random.shuffle(sorted_entries)
        else:  # mode == "all" or any other value
            sorted_entries = entries

        # Format output - show full formula to indicate unit cell size
        lines = [f"Structures for {comp} (reduced formula): {len(entries)} total"]
        for i, entry in enumerate(sorted_entries[:k]):
            full_formula = entry.get("full_formula", comp)
            score_str = ", ".join(
                f"{s}={v:.4f}" for s, v in entry.get("scores", {}).items()
            )

            structure_info = f"  {i}. {full_formula} [{entry['num_sites']} sites, {entry['source']}]{': ' + score_str if score_str else ''}"

            # Add full structural details if requested
            if include_structure_details:
                structure = entry["structure"]
                # Use pymatgen's string representation for complete structural info
                struct_str = str(structure)
                # Indent each line for better formatting
                indented_struct = "\n      ".join(struct_str.split("\n"))
                structure_info += f"\n      {indented_struct}"

            lines.append(structure_info)

        msg = "\n".join(lines)
        logger.info(f"[Tool] {msg}")
        return msg

    def get_buffer_stats(self) -> str:
        """Get detailed statistics about the current buffer.

        Returns:
            Summary: total count, compositions, sources, score ranges
        """
        if not self.buffer:
            return "Buffer is empty. Use generate_structures or create_structure to add candidates."

        # Count totals
        total_structures = sum(len(entries) for entries in self.buffer.values())
        num_compositions = len(self.buffer)

        # Sources
        sources = {}
        for entries in self.buffer.values():
            for entry in entries:
                src = entry.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

        # Score stats
        score_stats = {}
        for entries in self.buffer.values():
            for entry in entries:
                for scorer_name, score in entry.get("scores", {}).items():
                    if scorer_name not in score_stats:
                        score_stats[scorer_name] = []
                    score_stats[scorer_name].append(score)

        lines = [
            f"Buffer: {total_structures} structures across {num_compositions} compositions",
            f"  Sources: {sources}",
            "  Top compositions by count:",
        ]

        # Sort compositions by count
        sorted_comps = sorted(
            self.buffer.items(), key=lambda x: len(x[1]), reverse=True
        )[:5]
        for comp, entries in sorted_comps:
            lines.append(f"    {comp}: {len(entries)} structures")
        if len(self.buffer) > 5:
            lines.append(f"    ... +{len(self.buffer) - 5} more compositions")

        if score_stats:
            lines.append("  Score ranges:")
            for scorer_name, scores in score_stats.items():
                lines.append(
                    f"    {scorer_name}: min={min(scores):.4f}, max={max(scores):.4f}, mean={np.mean(scores):.4f}"
                )

        msg = "\n".join(lines)
        logger.info(f"[Tool] {msg}")
        return msg

    def select_for_evaluation(
        self,
        composition: str,
        structure_index: int = 0,
        scorer_name: str = "",
        reason: str = "",
    ) -> str:
        """Select a structure for oracle evaluation (two-step: composition then structure).

        This is the final action - selects ONE structure for ground-truth evaluation.

        Args:
            composition: Composition to select from (e.g., 'Li2O'). Required.
            structure_index: Index within the composition's structures (0-based).
                           If scorer_name provided, 0 = best scored structure.
            scorer_name: Optional scorer to use for ranking structures within composition.
                        If provided, structure_index refers to rank (0=best).
            reason: Reason for selection (for logging)

        Returns:
            Confirmation of selection with composition and scores
        """
        if not self.buffer:
            return "Error: Buffer is empty. Use generate_structures or create_structure first."

        comp = composition.strip()
        if not comp:
            return "Error: Must provide composition. Use list_compositions to see available."

        if comp not in self.buffer:
            available = list(self.buffer.keys())[:10]
            return (
                f"Error: No structures for composition '{comp}'. Available: {available}"
            )

        entries = self.buffer[comp]

        # Sort by scorer if specified
        if scorer_name:
            # Check if structures are scored
            if not entries[0].get("scores", {}).get(scorer_name):
                return f"Error: Structures for '{comp}' not scored with '{scorer_name}'. Run score_buffer first."
            # Already sorted by score_buffer, but ensure correct order
            entries = sorted(
                entries,
                key=lambda e: e["scores"].get(scorer_name, float("-inf")),
                reverse=True,
            )

        # Check index bounds
        if structure_index < 0 or structure_index >= len(entries):
            return f"Error: Index {structure_index} out of range for {comp} (has {len(entries)} structures)."

        # Select structure
        entry = entries[structure_index]
        self._selected_structure = entry["structure"]
        rank_str = (
            f"rank {structure_index} by {scorer_name}"
            if scorer_name
            else f"index {structure_index}"
        )
        self._selection_reason = reason or f"Selected {comp} ({rank_str})"

        # Remove from buffer
        self.buffer[comp].pop(structure_index)
        if not self.buffer[comp]:  # Remove composition if no more structures
            del self.buffer[comp]

        # Remove from cache
        if entry["hash"] in self.structure_cache:
            del self.structure_cache[entry["hash"]]

        # Format message
        full_formula = entry.get("full_formula", entry["composition"])
        score_str = ", ".join(
            f"{s}={v:.4f}" for s, v in entry.get("scores", {}).items()
        )
        total_remaining = sum(len(e) for e in self.buffer.values())
        msg = (
            f"SELECTED: {full_formula} (reduced: {entry['composition']}, {entry['num_sites']} sites, {entry['source']})"
            f"{': ' + score_str if score_str else ''}. "
            f"{self._selection_reason}. "
            f"Buffer: {total_remaining} structures remaining."
        )
        logger.info(f"[Tool] {msg}")
        return msg

    def create_structure(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        species: str = "",
        frac_coords: str = "",
    ) -> str:
        """Create a new structure from scratch with specified lattice and sites.

        This allows direct structure creation without using a generator.
        The structure is validated and added to the buffer if it passes filters.

        Args:
            a: Lattice parameter a (Angstroms)
            b: Lattice parameter b (Angstroms)
            c: Lattice parameter c (Angstroms)
            alpha: Lattice angle alpha (degrees, default 90)
            beta: Lattice angle beta (degrees, default 90)
            gamma: Lattice angle gamma (degrees, default 90)
            species: Comma-separated element symbols for each site (e.g., "Li, Li, O")
            frac_coords: Semicolon-separated fractional coordinates, each as "x,y,z"
                        (e.g., "0,0,0; 0.5,0.5,0.5; 0.25,0.25,0.25")

        Returns:
            Result message indicating success or failure
        """
        from pymatgen.core.lattice import Lattice

        # Parse species
        if not species.strip():
            return "Error: Must provide species (comma-separated element symbols)"
        species_list = [s.strip() for s in species.split(",") if s.strip()]

        # Parse fractional coordinates
        if not frac_coords.strip():
            return (
                "Error: Must provide frac_coords (semicolon-separated, each as 'x,y,z')"
            )

        coords_list = []
        for coord_str in frac_coords.split(";"):
            coord_str = coord_str.strip()
            if not coord_str:
                continue
            parts = [float(x.strip()) for x in coord_str.split(",")]
            if len(parts) != 3:
                return f"Error: Each coordinate must be 'x,y,z', got '{coord_str}'"
            coords_list.append(parts)

        # Validate lengths match
        if len(species_list) != len(coords_list):
            return f"Error: species count ({len(species_list)}) != coords count ({len(coords_list)})"

        # Validate elements
        for element in species_list:
            if element not in self.chemical_system_elements:
                return f"Error: Element {element} not in allowed elements: {self.chemical_system_elements}"

        # Check stoichiometry
        if len(species_list) > self.max_stoichiometry:
            return f"Error: Too many atoms ({len(species_list)}), max is {self.max_stoichiometry}"

        try:
            lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
            structure = Structure(lattice, species_list, coords_list)

            # Check for duplicates
            struct_hash = structure_hash(structure)
            if struct_hash in self.structure_cache:
                return f"Structure already in buffer (composition: {structure.composition.reduced_formula})"

            # Run static filter
            if self.static_filter is not None:
                try:
                    passed, results = self.static_filter.filter(
                        [structure], self.state, return_results=True
                    )
                    if not passed:
                        reasons = [
                            r.rejection_reason
                            for r in results
                            if not r.passed and r.rejection_reason
                        ]
                        return f"Structure failed filters: {'; '.join(reasons) if reasons else 'failed validation'}"
                except Exception as e:
                    return f"Filter error: {e}"

            # Run uniqueness filter
            if self.uniqueness_filter is not None:
                try:
                    passed, results = self.uniqueness_filter.filter(
                        [structure], self.state, return_results=True
                    )
                    if not passed:
                        return (
                            "Structure is not unique (duplicate of existing structure)"
                        )
                except Exception as e:
                    logger.warning(f"Uniqueness filter error: {e}")

            # Add to buffer
            comp = structure.composition.reduced_formula
            full_formula = structure.composition.formula.replace(" ", "")
            entry = {
                "structure": structure,
                "hash": struct_hash,
                "composition": comp,  # Reduced formula
                "full_formula": full_formula,  # Full formula
                "source": "llm_created",
                "scores": {},
                "num_sites": len(structure),
            }
            # Add to composition-based buffer (keyed by reduced formula)
            if comp not in self.buffer:
                self.buffer[comp] = []
            comp_idx = len(self.buffer[comp])
            self.buffer[comp].append(entry)
            self.structure_cache[struct_hash] = entry

            total_count = sum(len(entries) for entries in self.buffer.values())
            msg = f"Created and added: {full_formula} (reduced: {comp}, {len(structure)} sites). Buffer: {total_count} structures, {len(self.buffer)} compositions. {comp}: {comp_idx + 1} structures."
            logger.info(f"[Tool] {msg}")
            return msg

        except Exception as e:
            return f"Error creating structure: {e}"

    def get_selected_structure(self) -> Structure | None:
        """Get the selected structure (internal use)."""
        return self._selected_structure


# ============================================================================
# Main Agent Class
# ============================================================================


class LLMReActOrchestratorAgent(Agent):
    """
    LLM ReAct Orchestrator Agent that uses DSPy ReAct to decide actions.

    Maintains a buffer of pre-validated structures and uses an LLM
    to decide: generate, score, query, or select.
    """

    def __init__(
        self,
        # LLM configuration
        llm_config: dict[str, Any] | None = None,
        # Tool components
        generators: dict[str, Generator] | None = None,
        static_filters: Filter | None = None,  # FilterChain or single filter
        uniqueness_filter: Filter | None = None,  # Separate uniqueness filter
        scorers: dict[str, Scorer] | None = None,
        # Context and prompts (from config)
        context_config: dict[str, Any] | None = None,
        # Tool access
        enabled_tools: list[str] | None = None,
        # ReAct config
        max_iters: int = 10,
        # Constraints
        max_stoichiometry: int = 20,
        # History
        max_history_length: int = 20,
        **kwargs: Any,
    ):
        """
        Initialize the LLM ReAct Orchestrator Agent.

        Args:
            llm_config: DSPy LM configuration (model, cache, etc.)
            generators: Dict of named generators
            static_filters: Filter or FilterChain for static filtering (cached)
            uniqueness_filter: Uniqueness filter (always re-run)
            scorers: Dict of named scorers
            context_config: Prompts and context configuration
                - orchestration_prompt: Custom prompt (overrides default)
                - include_structure_in_history: Include full structure (lattice, species, positions) in evaluation history
                - include_structure_in_known_materials: Include full structure in known stable materials
            enabled_tools: List of tool names to enable
            max_iters: Maximum ReAct iterations per step
            max_stoichiometry: Maximum atoms per structure
            max_history_length: Maximum evaluation history entries
        """
        # Initialize base Agent (sets self.last_step = 0)
        super().__init__()

        # Store components
        self.generators = generators or {}
        self.static_filter = static_filters
        self.uniqueness_filter = uniqueness_filter
        self.scorers_dict = scorers or {}

        # Context and prompts
        self.context_config = context_config or {}

        # Configuration
        self.max_iters = max_iters
        self.max_stoichiometry = max_stoichiometry
        self.max_history_length = max_history_length

        # Enabled tools
        all_tools = [
            "generate_structures",
            "create_structure",
            "score_buffer",
            "list_compositions",
            "query_structures",
            "get_buffer_stats",
            "select_for_evaluation",
        ]
        self.enabled_tools = enabled_tools or all_tools

        # State - buffer is now composition-based: {composition: [entries]}
        self.buffer: dict[str, list[dict[str, Any]]] = {}
        self.structure_cache: dict[str, dict[str, Any]] = {}
        self.evaluation_history: list[dict[str, Any]] = []
        self.chemical_system_elements: list[str] = []

        # Initialize DSPy LM
        self.llm_config = llm_config or {}
        self._setup_dspy()

        # Prepare signature with optional prompt override (mirrors LLMScorer/LLMPlanner approach)
        signature = OrchestratorReActSignature
        if self.context_config.get("orchestration_prompt"):
            # Override the signature instructions with custom prompt from config
            signature.instructions = self.context_config["orchestration_prompt"]
        self.signature_class = signature

        logger.info(
            f"[LLMReActOrchestrator] Initialized with generators={list(self.generators.keys())}, "
            f"static_filter={self.static_filter is not None}, uniqueness_filter={self.uniqueness_filter is not None}, "
            f"scorers={list(self.scorers_dict.keys())}"
        )

    def _setup_dspy(self):
        """Setup DSPy LM."""
        try:
            model = self.llm_config.get("model", "anthropic/claude-sonnet-4-20250514")
            cache = self.llm_config.get("cache", True)
            max_tokens = self.llm_config.get("max_output_tokens")
            temperature = self.llm_config.get("temperature")

            kwargs = {"cache": cache}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            self.lm = dspy.LM(model, **kwargs)
            logger.info(f"[LLMReActOrchestrator] DSPy LM: {model}")
        except Exception as e:
            logger.error(f"[LLMReActOrchestrator] Failed to initialize DSPy LM: {e}")
            raise

    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Composition, Structure]:
        """
        Propose a structure using the ReAct loop.
        """
        self.chemical_system_elements = state.get("elements", [])
        stability_tolerance = state.get("stability_tolerance", 1e-8)

        logger.info(
            f"[LLMReActOrchestrator] Starting proposal (buffer={len(self.buffer)}, history={len(self.evaluation_history)})"
        )

        # Build context
        buffer_summary = self._format_buffer_summary()
        history_str = self._format_evaluation_history(stability_tolerance)
        stable_materials_str = self._format_known_stable_materials(state)

        # Create tools
        tools = OrchestratorTools(
            generators=self.generators,
            static_filter=self.static_filter,
            uniqueness_filter=self.uniqueness_filter,
            scorers=self.scorers_dict,
            chemical_system_elements=self.chemical_system_elements,
            max_stoichiometry=self.max_stoichiometry,
            buffer=self.buffer,
            structure_cache=self.structure_cache,
            state=state,
        )

        # Build tool list
        tool_map = {
            "generate_structures": tools.generate_structures,
            "create_structure": tools.create_structure,
            "score_buffer": tools.score_buffer,
            "list_compositions": tools.list_compositions,
            "query_structures": tools.query_structures,
            "get_buffer_stats": tools.get_buffer_stats,
            "select_for_evaluation": tools.select_for_evaluation,
        }
        enabled_tool_functions = [
            tool_map[name] for name in self.enabled_tools if name in tool_map
        ]

        # Create ReAct module
        react_module = dspy.ReAct(
            self.signature_class,
            tools=enabled_tool_functions,
            max_iters=self.max_iters,
        )

        logger.info(
            f"[LLMReActOrchestrator] ReAct: elements={self.chemical_system_elements}, tools={self.enabled_tools}"
        )

        # Run ReAct
        try:
            with dspy.context(lm=self.lm):
                result = react_module(
                    chemical_system=", ".join(self.chemical_system_elements),
                    stability_tolerance=stability_tolerance,
                    max_stoichiometry=self.max_stoichiometry,
                    buffer_summary=buffer_summary,
                    evaluation_history=history_str,
                    known_stable_materials=stable_materials_str,
                )

            logger.info(f"[LLMReActOrchestrator] Result: {result.answer}")

            selected = tools.get_selected_structure()

            if selected is None:
                logger.warning(
                    "[LLMReActOrchestrator] No structure selected. Falling back."
                )
                selected = self._fallback_selection(state)

            return selected.composition, selected

        except Exception as e:
            logger.error(f"[LLMReActOrchestrator] ReAct failed: {e}")
            selected = self._fallback_selection(state)
            return selected.composition, selected

    def _fallback_selection(self, state: dict[str, Any]) -> Structure:
        """Fallback selection if ReAct fails."""
        if self.buffer:
            # Pick first composition and first structure from it
            comp = next(iter(self.buffer.keys()))
            entry = self.buffer[comp].pop(0)
            if not self.buffer[comp]:  # Remove composition if empty
                del self.buffer[comp]
            logger.info(f"[LLMReActOrchestrator] Fallback: {entry['composition']}")
            return entry["structure"]

        if self.generators:
            gen_name, generator = next(iter(self.generators.items()))
            plan = Plan(
                compositions=[
                    Composition(dict.fromkeys(self.chemical_system_elements, 1))
                ],
                num_candidates=1,
                constraints={"elements": self.chemical_system_elements},
            )
            try:
                structures = generator.generate(plan, state)
                if structures:
                    logger.info(
                        f"[LLMReActOrchestrator] Fallback: generated via {gen_name}"
                    )
                    return structures[0]
            except Exception as e:
                logger.error(f"[LLMReActOrchestrator] Fallback generation failed: {e}")

        raise RuntimeError("No structure available")

    def _format_buffer_summary(self) -> str:
        """Format buffer summary for LLM context."""
        if not self.buffer:
            return "Buffer is empty. Use generate_structures or create_structure to add candidates."

        total_structures = sum(len(entries) for entries in self.buffer.values())
        num_compositions = len(self.buffer)

        sources = {}
        for entries in self.buffer.values():
            for entry in entries:
                src = entry.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

        lines = [
            f"Buffer: {total_structures} structures across {num_compositions} compositions",
            f"  Sources: {sources}",
            "  Top compositions:",
        ]

        # Sort compositions by count
        sorted_comps = sorted(
            self.buffer.items(), key=lambda x: len(x[1]), reverse=True
        )[:3]
        for comp, entries in sorted_comps:
            lines.append(f"    {comp}: {len(entries)} structures")

        return "\n".join(lines)

    def _format_evaluation_history(self, stability_tolerance: float) -> str:
        """Format evaluation history for LLM context."""
        if not self.evaluation_history:
            return "No evaluations yet."

        include_structures = self.context_config.get(
            "include_structure_in_history", False
        )

        lines = []
        recent = self.evaluation_history[-self.max_history_length :]
        for i, entry in enumerate(recent, 1):
            comp = entry.get("composition", "?")
            e_hull = entry.get("e_above_hull", float("inf"))
            is_stable = entry.get("is_stable", False)
            is_novel = entry.get("is_newly_discovered", False)

            status = []
            if is_stable:
                status.append("STABLE")
            elif e_hull <= stability_tolerance:
                status.append("METASTABLE")
            if is_novel:
                status.append("NOVEL")
            status_str = ", ".join(status) if status else "unstable"

            line = f"{i}. {comp} [{status_str}, e_above_hull={e_hull:.4f}]"

            # Optionally add full structure info
            if include_structures and entry.get("structure"):
                structure = entry["structure"]
                struct_str = str(structure)
                # Indent each line for better formatting
                indented_struct = "\n   ".join(struct_str.split("\n"))
                line += f"\n   {indented_struct}"

            lines.append(line)

        return "\n".join(lines)

    def _format_known_stable_materials(self, state: dict[str, Any]) -> str:
        """Format known stable materials from phase diagram."""
        entries_raw = state.get("phase_diagram_all_entries", [])
        if not entries_raw:
            return "No known materials."

        include_structures = self.context_config.get(
            "include_structure_in_known_materials", False
        )

        entries = []
        for e in entries_raw:
            if isinstance(e, dict):
                entries.append(PDEntry.from_dict(e))
            elif isinstance(e, PDEntry):
                entries.append(e)

        if not entries:
            return "No known materials."

        try:
            pd = PhaseDiagram(entries)
            stable = pd.stable_entries

            # Filter to compounds only
            stable_compounds = [e for e in stable if len(e.composition.elements) >= 2]

            if not stable_compounds:
                return "No known compound stable materials."

            if not include_structures:
                # Simple format - just formulas
                formulas = [e.composition.reduced_formula for e in stable_compounds]
                return f"Known stable: {', '.join(sorted(set(formulas)))}"
            else:
                # Detailed format with structure info
                lines = ["Known stable materials:"]
                for entry in stable_compounds:
                    comp = entry.composition.reduced_formula
                    e_form = pd.get_form_energy_per_atom(entry)

                    line = f"  {comp} (formation_energy={e_form:.4f} eV/atom)"

                    # Add full structure info if available
                    structure = None
                    if hasattr(entry, "structure") and entry.structure:
                        structure = entry.structure
                    elif hasattr(entry, "attribute") and entry.attribute:
                        structure = entry.attribute.get("structure")

                    if structure:
                        struct_str = str(structure)
                        # Indent each line for better formatting
                        indented_struct = "\n    ".join(struct_str.split("\n"))
                        line += f"\n    {indented_struct}"

                    lines.append(line)

                return "\n".join(lines)
        except Exception as e:
            logger.warning(
                f"[LLMReActOrchestrator] Failed to compute stable materials: {e}"
            )
            return "Unable to compute stable materials."

    def update_state(self, state: dict[str, Any]) -> None:
        """Update agent state from environment observation (uses base Agent step tracking)."""
        action = self._update_last_step(state)

        # Update chemical system on init
        if action == "init":
            self.chemical_system_elements = state.get("elements", [])
            logger.info(
                f"[LLMReActOrchestrator] Initialized: {self.chemical_system_elements}"
            )

        if action == "skip":
            return

        # Update component states (on init or update)
        for gen in self.generators.values():
            gen.update_state(state)
        if self.static_filter:
            self.static_filter.update_state(state)
        if self.uniqueness_filter:
            self.uniqueness_filter.update_state(state)
        for scorer in self.scorers_dict.values():
            scorer.update_state(state)

        # Only record evaluation on update (not init)
        if action != "update":
            return

        last_obs = state.get("last_observation")
        e_hull = last_obs.get("e_above_hull", float("inf"))
        stability_tolerance = state.get("stability_tolerance", 1e-8)

        # Store structure for optional context inclusion
        structure = None
        if "proposal" in last_obs:
            try:
                proposal = last_obs["proposal"]
                if isinstance(proposal, dict):
                    structure = Structure.from_dict(proposal)
                else:
                    structure = proposal
            except Exception:
                pass

        history_entry = {
            "composition": last_obs.get("reduced_formula", "?"),
            "e_above_hull": e_hull,
            "is_stable": last_obs.get("is_stable", e_hull <= stability_tolerance),
            "is_newly_discovered": last_obs.get("is_newly_discovered", False),
            "energy_per_atom": last_obs.get("energy_per_atom"),
            "structure": structure,  # Store for optional context
        }
        self.evaluation_history.append(history_entry)

        # Remove from cache
        if "proposal" in last_obs:
            try:
                proposal = last_obs["proposal"]
                if isinstance(proposal, dict):
                    proposal = Structure.from_dict(proposal)
                proposal_hash = structure_hash(proposal)
                if proposal_hash in self.structure_cache:
                    del self.structure_cache[proposal_hash]
            except Exception:
                pass

        status = "STABLE" if history_entry["is_stable"] else "unstable"
        if history_entry["is_newly_discovered"]:
            status += ", NOVEL"

        logger.info(
            f"[LLMReActOrchestrator] Recorded: {history_entry['composition']} [{status}, e={e_hull:.4f}]"
        )

    def get_state(self) -> dict[str, Any]:
        """Get agent state for checkpointing."""
        # Serialize composition-based buffer
        buffer_dict = {}
        for comp, entries in self.buffer.items():
            buffer_dict[comp] = [
                {
                    "structure_dict": entry["structure"].as_dict(),
                    "hash": entry["hash"],
                    "composition": entry["composition"],
                    "full_formula": entry.get("full_formula", entry["composition"]),
                    "source": entry.get("source"),
                    "scores": entry.get("scores", {}),
                    "num_sites": entry.get("num_sites", 0),
                }
                for entry in entries
            ]

        # Serialize evaluation history with structures
        serialized_history = []
        for entry in self.evaluation_history:
            serialized_entry = {
                "composition": entry["composition"],
                "e_above_hull": entry["e_above_hull"],
                "is_stable": entry["is_stable"],
                "is_newly_discovered": entry["is_newly_discovered"],
                "energy_per_atom": entry.get("energy_per_atom"),
            }
            if entry.get("structure"):
                serialized_entry["structure_dict"] = entry["structure"].as_dict()
            serialized_history.append(serialized_entry)

        return {
            "buffer": buffer_dict,
            "structure_cache_hashes": list(self.structure_cache.keys()),
            "evaluation_history": serialized_history,
            "chemical_system_elements": self.chemical_system_elements,
            "last_step": self.last_step,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load agent state from checkpoint."""
        if "buffer" in state:
            self.buffer = {}
            self.structure_cache = {}
            buffer_data = state["buffer"]

            # Handle both old (list) and new (dict) formats
            if isinstance(buffer_data, list):
                # Old format - convert to new format
                for entry_dict in buffer_data:
                    structure = Structure.from_dict(entry_dict["structure_dict"])
                    comp = entry_dict["composition"]
                    full_formula = entry_dict.get(
                        "full_formula", structure.composition.formula.replace(" ", "")
                    )
                    entry = {
                        "structure": structure,
                        "hash": entry_dict["hash"],
                        "composition": comp,
                        "full_formula": full_formula,
                        "source": entry_dict.get("source"),
                        "scores": entry_dict.get("scores", {}),
                        "num_sites": entry_dict.get("num_sites", len(structure)),
                    }
                    if comp not in self.buffer:
                        self.buffer[comp] = []
                    self.buffer[comp].append(entry)
                    self.structure_cache[entry["hash"]] = entry
            else:
                # New format (dict of compositions)
                for comp, entries_list in buffer_data.items():
                    self.buffer[comp] = []
                    for entry_dict in entries_list:
                        structure = Structure.from_dict(entry_dict["structure_dict"])
                        full_formula = entry_dict.get(
                            "full_formula",
                            structure.composition.formula.replace(" ", ""),
                        )
                        entry = {
                            "structure": structure,
                            "hash": entry_dict["hash"],
                            "composition": entry_dict["composition"],
                            "full_formula": full_formula,
                            "source": entry_dict.get("source"),
                            "scores": entry_dict.get("scores", {}),
                            "num_sites": entry_dict.get("num_sites", len(structure)),
                        }
                        self.buffer[comp].append(entry)
                        self.structure_cache[entry["hash"]] = entry

        if "evaluation_history" in state:
            # Deserialize structures in history
            self.evaluation_history = []
            for entry_dict in state["evaluation_history"]:
                entry = {
                    "composition": entry_dict["composition"],
                    "e_above_hull": entry_dict["e_above_hull"],
                    "is_stable": entry_dict["is_stable"],
                    "is_newly_discovered": entry_dict["is_newly_discovered"],
                    "energy_per_atom": entry_dict.get("energy_per_atom"),
                }
                # Deserialize structure if present
                if "structure_dict" in entry_dict:
                    entry["structure"] = Structure.from_dict(
                        entry_dict["structure_dict"]
                    )
                else:
                    entry["structure"] = None
                self.evaluation_history.append(entry)

        if "chemical_system_elements" in state:
            self.chemical_system_elements = state["chemical_system_elements"]

        if "last_step" in state:
            self.last_step = state["last_step"]
