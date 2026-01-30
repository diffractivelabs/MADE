"""
Grid search planner that enumerates all compositions and iterates over them.

Supports multiple selection strategies:
- Sequential iteration
- Random selection
- UCB (Upper Confidence Bound) selection
- Diversity-based selection
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from typing import Any

import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from made.evaluation.metrics import is_smact_valid
from made.utils.convex_hull_utils import safe_e_above_hull

from ..base import Plan, Planner

logger = logging.getLogger(__name__)


# Base Selection Strategy Class
class SelectionStrategy(ABC):
    """Base class for composition selection strategies."""

    @abstractmethod
    def select(
        self,
        composition_stats: dict[str, "CompositionStats"],
        num_compositions: int,
        rng: np.random.RandomState,
    ) -> list[Composition]:
        """Select compositions based on the strategy.

        Args:
            composition_stats: Dictionary mapping formula -> CompositionStats
            num_compositions: Number of compositions to select
            rng: Random number generator
            **kwargs: Strategy-specific parameters

        Returns:
            List of selected Composition objects
        """


class RandomSelectionStrategy(SelectionStrategy):
    """Randomly select compositions each time, filtering out those that have reached max attempts or success criteria."""

    def __init__(
        self,
        max_attempts_per_composition: int = 5,
        move_on_success: bool = True,
        **kwargs,
    ):
        self.max_attempts_per_composition = max_attempts_per_composition
        self.move_on_success = move_on_success

    def select(
        self,
        composition_stats: dict[str, "CompositionStats"],
        num_compositions: int,
        rng: np.random.RandomState,
    ) -> list[Composition]:
        all_stats = list(composition_stats.values())

        # Filter out compositions that have reached their limits
        available_stats = []
        for stats in all_stats:
            should_include = stats.num_attempts < self.max_attempts_per_composition
            if self.move_on_success:
                should_include = should_include and stats.num_successes == 0

            if should_include:
                available_stats.append(stats)

        # If no compositions are available, fall back to all compositions
        if not available_stats:
            available_stats = all_stats

        # Randomly select from available compositions
        selected_compositions = []
        for _ in range(min(num_compositions, len(available_stats))):
            selected_idx = rng.choice(len(available_stats))
            selected_compositions.append(available_stats[selected_idx].composition)

        return selected_compositions


class UCBSelectionStrategy(SelectionStrategy):
    """Select compositions using Upper Confidence Bound strategy."""

    def __init__(
        self,
        exploration_factor: float = 1.0,
        ucb_reward: str = "success_rate",
        **kwargs,
    ):
        self.ucb_reward = ucb_reward
        self.exploration_factor = exploration_factor

    def select(
        self,
        composition_stats: dict[str, "CompositionStats"],
        num_compositions: int,
        rng: np.random.RandomState,
    ) -> list[Composition]:
        total_attempts = sum(stats.num_attempts for stats in composition_stats.values())

        # Calculate UCB scores for all compositions
        all_stats = list(composition_stats.values())
        ucb_scores = []
        for stats in all_stats:
            if self.ucb_reward == "success_rate":
                exploitation = stats.success_rate
            elif self.ucb_reward == "failure_rate":
                exploitation = 1.0 - stats.success_rate
            elif self.ucb_reward == "best_score":
                exploitation = stats.best_score
            else:
                raise ValueError(
                    f"Invalid ucb_reward: {self.ucb_reward}. Expected 'success_rate', 'failure_rate', or 'best_score'"
                )
            exploration = self.exploration_factor * np.sqrt(
                np.log(total_attempts) / stats.num_attempts
            )
            score = exploitation + exploration
            ucb_scores.append(score)

        # Convert to numpy array for easier manipulation
        ucb_scores = np.array(ucb_scores)

        # Select compositions, handling ties by random sampling
        selected_indices = []
        available_indices = set(range(len(all_stats)))

        for _ in range(num_compositions):
            if not available_indices:
                break

            # Find the maximum score among remaining compositions
            available_mask = np.zeros(len(ucb_scores), dtype=bool)
            available_mask[list(available_indices)] = True
            masked_scores = np.where(available_mask, ucb_scores, -np.inf)
            max_score = np.max(masked_scores)

            # Find all indices with the maximum score (ties)
            tied_indices = [i for i in available_indices if ucb_scores[i] == max_score]

            # Randomly sample from tied indices
            selected_idx = rng.choice(tied_indices)
            selected_indices.append(selected_idx)
            available_indices.remove(selected_idx)

        return [all_stats[i].composition for i in selected_indices]


class UnexploredSelectionStrategy(SelectionStrategy):
    """Select compositions that haven't been tried yet, then fall back to least tried."""

    def __init__(self, **kwargs):
        # UnexploredSelectionStrategy doesn't use any initialization parameters
        pass

    def select(
        self,
        composition_stats: dict[str, "CompositionStats"],
        num_compositions: int,
        rng: np.random.RandomState,
    ) -> list[Composition]:
        all_stats = list(composition_stats.values())
        unexplored = [stats for stats in all_stats if stats.num_attempts == 0]

        if len(unexplored) >= num_compositions:
            # Randomly sample from unexplored
            selected_indices = rng.choice(
                len(unexplored), size=num_compositions, replace=False
            )
            return [unexplored[i].composition for i in selected_indices]
        else:
            # Take all unexplored, then fill with least attempted
            selected_compositions = [stats.composition for stats in unexplored]
            remaining_needed = num_compositions - len(selected_compositions)

            if remaining_needed > 0:
                # Sort by number of attempts (ascending) and take the least tried
                explored = [stats for stats in all_stats if stats.num_attempts > 0]
                explored.sort(key=lambda x: x.num_attempts)
                selected_compositions.extend(
                    [stats.composition for stats in explored[:remaining_needed]]
                )

            return selected_compositions


class IterativeSelectionStrategy(SelectionStrategy):
    """Iteratively go through each composition until max attempts or optionally until success is achieved."""

    def __init__(
        self,
        max_attempts_per_composition: int = 5,
        move_on_success: bool = True,
        **kwargs,
    ):
        self.current_composition_index = 0
        self.composition_order = None  # Will be set on first use
        self.max_attempts_per_composition = max_attempts_per_composition
        self.move_on_success = move_on_success

    def select(
        self,
        composition_stats: dict[str, "CompositionStats"],
        num_compositions: int,
        rng: np.random.RandomState,
    ) -> list[Composition]:
        all_stats = list(composition_stats.values())

        # Initialize composition order on first use (deterministic ordering)
        if self.composition_order is None:
            # Sort by formula for deterministic ordering
            self.composition_order = sorted(
                all_stats, key=lambda x: x.composition.formula
            )

        selected_compositions = []

        for _ in range(num_compositions):
            if not self.composition_order:
                break

            # Get current composition
            current_stats = self.composition_order[self.current_composition_index]

            # Check if we should continue with this composition
            should_continue = (
                current_stats.num_attempts < self.max_attempts_per_composition
            )
            if self.move_on_success:
                should_continue = should_continue and current_stats.num_successes == 0

            if should_continue:
                # Continue with current composition
                selected_compositions.append(current_stats.composition)
            else:
                # Move to next composition
                self.current_composition_index = (
                    self.current_composition_index + 1
                ) % len(self.composition_order)
                current_stats = self.composition_order[self.current_composition_index]
                selected_compositions.append(current_stats.composition)

        return selected_compositions


class DiversitySelectionStrategy(SelectionStrategy):
    """Select compositions that are maximally distant from previously attempted compositions."""

    def __init__(
        self,
        distance_metric: str = "euclidean",
        unattempted_weight: float = 2.0,
        attempt_weight_factor: float = 0.7,
        failure_weight_factor: float = 0.3,
        mask_same_reduced_formula: bool = True,
        **kwargs,
    ):
        self.distance_metric = distance_metric
        self.unattempted_weight = unattempted_weight
        self.attempt_weight_factor = attempt_weight_factor
        self.failure_weight_factor = failure_weight_factor
        self.mask_same_reduced_formula = mask_same_reduced_formula
        self.distance_matrix = None
        self.ref_compositions = None  # list of reference compositions (all compositions + elemental references)

    def select(
        self,
        composition_stats: dict[str, "CompositionStats"],
        num_compositions: int,
        rng: np.random.RandomState,
    ) -> list[Composition]:
        all_stats = list(composition_stats.values())
        all_compositions = [stats.composition for stats in all_stats]
        # Create mapping from composition to stats for easy lookup
        comp_to_stats = {stats.composition.formula: stats for stats in all_stats}

        # Build elemental reference compositions from the element set present
        if self.distance_matrix is None:
            unique_element_symbols: set[str] = set()
            for comp in all_compositions:
                for el in comp.elements:
                    unique_element_symbols.add(el.symbol)
            elemental_refs: list[Composition] = [
                Composition(sym) for sym in sorted(unique_element_symbols)
            ]
            # Candidates are all_compositions (rows); references are candidates + elemental references (cols)
            self.ref_compositions: list[Composition] = all_compositions + elemental_refs
            cand_mat = self._compositions_to_matrix(all_compositions)
            ref_mat = self._compositions_to_matrix(self.ref_compositions)
            self.distance_matrix = self._compute_vectorized_distances(cand_mat, ref_mat)

        # Compute weights for all compositions (unattempted are given high weight via config)
        weights = self._compute_composition_weights(all_compositions, comp_to_stats)

        # Create cross mask for candidate-to-reference comparisons to block same/reduced comparisons among candidates
        cand_formulas = np.array([c.formula for c in all_compositions])
        ref_formulas = np.array([c.formula for c in self.ref_compositions])
        formula_mask = cand_formulas[:, None] == ref_formulas[None, :]
        if self.mask_same_reduced_formula:
            cand_reduced = np.array([c.reduced_formula for c in all_compositions])
            ref_reduced = np.array([c.reduced_formula for c in self.ref_compositions])
            reduced_mask = cand_reduced[:, None] == ref_reduced[None, :]
            cross_mask = formula_mask | reduced_mask
        else:
            cross_mask = formula_mask

        # Create mask for attempted compositions (initially based on num_attempts > 0)
        attempted_mask = np.array([stats.num_attempts > 0 for stats in all_stats])

        # Initialize list of compositions to select and a set of chosen indices for this batch
        selected_compositions = []
        selected_indices: set[int] = set()

        for _ in range(num_compositions):
            # References: first N columns correspond to candidates; remaining are elemental references (always active)
            num_candidates = len(all_compositions)
            num_refs = self.distance_matrix.shape[1]
            attempted_cols = np.zeros(num_refs, dtype=bool)
            attempted_cols[:num_candidates] = attempted_mask
            if num_refs > num_candidates:
                attempted_cols[num_candidates:] = True

            # Valid references exclude cross-comparisons blocked by cross_mask
            valid_ref_mask = attempted_cols[None, :] & (~cross_mask)

            # Apply the valid mask to the distance matrix; invalid entries become +inf
            distances_masked = np.where(valid_ref_mask, self.distance_matrix, np.inf)
            # Take the minimum along attempted columns for each candidate row
            min_distances = np.min(distances_masked, axis=1)
            # Rows with no valid attempted references will be +inf → treat as not selectable now
            # so that unattempted (with finite distances) are prioritized via weights
            min_distances = np.where(np.isinf(min_distances), -np.inf, min_distances)
            # Combine with weights (higher weight = higher priority)
            weighted_scores = min_distances * weights
            # Avoid reselection within the same batch
            if selected_indices:
                for idx in selected_indices:
                    weighted_scores[idx] = -np.inf

                # Additionally, if configured, mask any candidates that share the
                # same reduced formula as compositions already selected in this batch
                if self.mask_same_reduced_formula and selected_compositions:
                    selected_reduced = {
                        c.reduced_formula for c in selected_compositions
                    }
                    cand_reduced_arr = np.array(
                        [c.reduced_formula for c in all_compositions]
                    )
                    same_reduced_mask = np.isin(
                        cand_reduced_arr, list(selected_reduced)
                    )
                    weighted_scores[same_reduced_mask] = -np.inf
            # Select composition with highest weighted score
            best_idx = np.argmax(weighted_scores)
            selected_compositions.append(all_compositions[best_idx])

            # Update attempted mask to include the newly selected composition
            attempted_mask[best_idx] = True

            selected_indices.add(best_idx)

        return selected_compositions

    # utils for distance computation
    def _compositions_to_matrix(self, compositions: list[Composition]) -> np.ndarray:
        """Convert list of compositions to a matrix representation."""
        all_elements = set()
        for comp in compositions:
            all_elements.update(comp.elements)
        sorted_elements = sorted(all_elements, key=lambda x: x.symbol)
        element_to_idx = {el: i for i, el in enumerate(sorted_elements)}

        mat = np.zeros((len(compositions), len(sorted_elements)))
        for i, comp in enumerate(compositions):
            fractional = comp.fractional_composition
            for el, frac in fractional.items():
                j = element_to_idx[el]
                mat[i, j] = frac
        return mat

    def _compute_vectorized_distances(
        self, candidate_matrix: np.ndarray, reference_matrix: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise distances between candidate and reference compositions.
        Args:
            candidate_matrix: matrix of candidate compositions, shape: (n_candidates, n_elements)
            reference_matrix: matrix of reference compositions, shape: (n_references, n_elements)
        Returns:
            distances: matrix of distances, shape: (n_candidates, n_references)
        """
        if self.distance_metric == "euclidean":
            diff = candidate_matrix[:, None, :] - reference_matrix[None, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
        elif self.distance_metric == "manhattan":
            diff = candidate_matrix[:, None, :] - reference_matrix[None, :, :]
            distances = np.sum(np.abs(diff), axis=2)
        elif self.distance_metric == "cosine":
            cand_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
            ref_norms = np.linalg.norm(reference_matrix, axis=1, keepdims=True)
            cand_norms = np.where(cand_norms == 0, 1, cand_norms)
            ref_norms = np.where(ref_norms == 0, 1, ref_norms)
            cand_norm = candidate_matrix / cand_norms
            ref_norm = reference_matrix / ref_norms
            cos_sim = np.dot(cand_norm, ref_norm.T)
            distances = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def _create_comparison_mask(self, compositions: list[Composition]) -> np.ndarray:
        """Create mask matrix for compositions that should not be compared.

        Args:
            compositions: List of all compositions

        Returns:
            Boolean mask matrix, shape (n_compositions, n_compositions)
            True means compositions should be excluded from distance comparison
        """
        formulas = np.array([comp.formula for comp in compositions])

        # Create formula comparison mask
        formula_mask = formulas[:, None] == formulas[None, :]

        # Optionally add reduced formula mask
        if self.mask_same_reduced_formula:
            reduced_formulas = np.array([comp.reduced_formula for comp in compositions])
            reduced_mask = reduced_formulas[:, None] == reduced_formulas[None, :]
            return formula_mask | reduced_mask
        else:
            return formula_mask

    def _compute_composition_weights(
        self,
        compositions: list[Composition],
        comp_to_stats: dict[str, "CompositionStats"],
    ) -> np.ndarray:
        """Compute weights for compositions based on attempt history and success rate."""
        weights = []

        for comp in compositions:
            stats = comp_to_stats[comp.formula]

            # Weight based on: 1) fewer attempts, 2) lower success rate (explore failures)
            attempt_weight = 1.0 / (
                stats.num_attempts + 1
            )  # +1 to avoid division by zero
            failure_weight = (
                1.0 - stats.success_rate
            )  # Higher weight for compositions with more failures

            # For unattempted compositions, give them high priority
            if stats.num_attempts == 0:
                combined_weight = (
                    self.unattempted_weight
                )  # High base weight for unattempted
            else:
                # Combine weights for attempted compositions
                combined_weight = (
                    attempt_weight * self.attempt_weight_factor
                    + failure_weight * self.failure_weight_factor
                )

            weights.append(combined_weight)

        return np.array(weights)


# Strategy registry for class-based strategies
SELECTION_STRATEGIES = {
    "random": RandomSelectionStrategy,
    "ucb": UCBSelectionStrategy,
    "unexplored": UnexploredSelectionStrategy,
    "iterative": IterativeSelectionStrategy,
    "diversity": DiversitySelectionStrategy,
}


@dataclass
class CompositionStats:
    """Statistics for a composition in the grid search."""

    composition: Composition
    num_attempts: int = 0
    num_successes: int = 0
    best_score: float = -10000000
    last_score: float = -10000000
    last_attempted_step: int = -1

    @property
    def success_rate(self) -> float:
        """Success rate (0-1) for this composition."""
        return self.num_successes / max(self.num_attempts, 1)


class GridSearchPlanner(Planner):
    def __init__(
        self,
        max_stoichiometry: int = 20,
        num_compositions: int = 8,
        num_candidates: int = 1,
        seed: int | None = None,
        selection_strategy: str | SelectionStrategy = "random",
        selection_strategy_kwargs: dict[str, Any] | None = None,
        score_function: str = "e_above_hull",
        filter_by_smact_validity: bool = True,
        return_all_compositions: bool = False,
    ):
        self.max_stoichiometry = max_stoichiometry
        self.num_compositions = num_compositions
        self.num_candidates = num_candidates
        self.rng = np.random.RandomState(seed)
        self.composition_stats: dict[str, CompositionStats] = {}
        self.selection_strategy_kwargs = selection_strategy_kwargs or {}
        self.current_step = 0
        self.score_function = score_function
        self.filter_by_smact_validity = filter_by_smact_validity
        self.return_all_compositions = return_all_compositions

        # Initialize selection strategy
        if isinstance(selection_strategy, str):
            if selection_strategy not in SELECTION_STRATEGIES:
                raise ValueError(
                    f"Unknown selection strategy: {selection_strategy}. Available: {list(SELECTION_STRATEGIES.keys())}"
                )
            # Instantiate strategy class with kwargs
            self.selection_strategy = SELECTION_STRATEGIES[selection_strategy](
                **self.selection_strategy_kwargs
            )
        elif isinstance(selection_strategy, SelectionStrategy):
            self.selection_strategy = selection_strategy
        else:
            raise ValueError(
                "selection_strategy must be a string or SelectionStrategy instance"
            )

    def get_state(self) -> dict[str, Any]:
        """Get full state including composition stats for checkpointing."""
        composition_stats_state = {
            formula: {
                "composition": stats.composition.formula,  # Save composition formula
                "num_attempts": int(stats.num_attempts),
                "num_successes": int(stats.num_successes),
                "success_rate": float(
                    stats.success_rate
                ),  # Convert numpy float to Python float
                "best_score": float(stats.best_score)
                if stats.best_score is not None
                else None,
                "last_score": float(stats.last_score)
                if stats.last_score is not None
                else None,
                "last_attempted_step": int(stats.last_attempted_step),
            }
            for formula, stats in self.composition_stats.items()
        }

        return {
            "current_step": int(self.current_step),
            "composition_stats": composition_stats_state,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load full state including composition stats from checkpoint."""
        if "current_step" in state:
            self.current_step = state["current_step"]

        if "composition_stats" in state:
            from pymatgen.core.composition import Composition

            self.composition_stats = {}
            for formula, stats_dict in state["composition_stats"].items():
                comp_formula = stats_dict.get("composition", formula)
                composition = Composition(comp_formula)
                self.composition_stats[formula] = CompositionStats(
                    composition=composition,
                    num_attempts=stats_dict.get("num_attempts", 0),
                    num_successes=stats_dict.get("num_successes", 0),
                    best_score=stats_dict.get("best_score", None),
                    last_score=stats_dict.get("last_score", None),
                    last_attempted_step=stats_dict.get("last_attempted_step", 0),
                )

    def update_state(self, state: dict[str, Any]) -> None:
        """Update composition statistics based on the last observation from the environment state."""

        # Initialize composition stats if first time
        if not self.composition_stats:
            elems = state.get("elements", [])
            if not elems:
                raise ValueError("GridSearchPlanner requires 'elements' in state")
            self.composition_stats = self._init_composition_stats(elems)
            # Update statistics based on initial state
            self._update_stats_from_state(state)
            logger.info(f"Initialized composition stats with {self.composition_stats}")

        if state.get("last_observation"):
            # make structure from dict
            obs = state.get("last_observation")
            structure = Structure.from_dict(obs["proposal"])
            if structure.composition.formula in self.composition_stats:
                stats = self.composition_stats[structure.composition.formula]
                stats.num_attempts += 1
                if self.score_function == "energy_per_atom":
                    score = -float(obs["energy_per_atom"])  # lower is better
                elif self.score_function == "formation_energy_per_atom":
                    score = -float(obs["formation_energy_per_atom"])  # lower is better
                elif self.score_function == "e_above_hull":
                    score = -float(obs["e_above_hull"])  # lower is better
                else:
                    raise ValueError(
                        f"Invalid score_function: {self.score_function}. Expected 'energy_per_atom', 'formation_energy_per_atom', or 'e_above_hull'"
                    )
                if obs["is_stable"]:
                    stats.num_successes += 1

                stats.best_score = max(stats.best_score, score)
                stats.last_score = score
                stats.last_attempted_step = self.current_step
            else:
                logger.warning(
                    f"Composition {structure.composition.formula} not found in composition stats, skipping (probably over the max stoichiometry or elemental: {self.max_stoichiometry})"
                )

    def propose(
        self, state: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> Plan:
        # If return_all_compositions is enabled, return all valid compositions
        if self.return_all_compositions:
            all_compositions = [
                stats.composition for stats in self.composition_stats.values()
            ]
            logger.info(f"Returning all {len(all_compositions)} compositions in plan")
            self.current_step += 1
            return Plan(
                compositions=all_compositions, num_candidates=self.num_candidates
            )

        # Otherwise, select compositions based on strategy
        selected_compositions = self.selection_strategy.select(
            composition_stats=self.composition_stats,
            num_compositions=self.num_compositions,
            rng=self.rng,
        )

        # Update step counter
        self.current_step += 1

        return Plan(
            compositions=selected_compositions, num_candidates=self.num_candidates
        )

    def _init_composition_stats(self, elems: list[str]) -> dict[str, CompositionStats]:
        """
        Enumerate all compositions and create CompositionStats objects for each.
        Returns a dictionary mapping formula -> CompositionStats for O(1) lookup.
        e.g. with max stoichiometry 2, and elements A, B, C, the result will be:
        {"A": CompositionStats(A), "B": CompositionStats(B), "C": CompositionStats(C),
         "A2": CompositionStats(A2), "B2": CompositionStats(B2), "C2": CompositionStats(C2),
         "AB": CompositionStats(AB), "AC": CompositionStats(AC), "BC": CompositionStats(BC)}

        If filter_by_smact_validity is True, only compositions that pass SMACT validity
        checks will be included. When SMACT filtering is enabled, the enumeration approach:
        - Uses itertools.product to generate all stoichiometry combinations (0 to max_stoichiometry)
        - Filters by SMACT validity (which internally uses reduced composition for the check)
        - Keeps all unique compositions by their full formula
        """
        composition_stats: dict[str, CompositionStats] = {}

        if self.filter_by_smact_validity:
            # Reference implementation approach: enumerate all stoichiometry combinations
            # Generate all combinations where the sum of atoms is at most max_stoichiometry
            all_compositions = []
            for amt_list in product(
                range(self.max_stoichiometry + 1), repeat=len(elems)
            ):
                # Skip the all-zeros combination
                if max(amt_list) == 0:
                    continue

                # Constrain total atoms to be at most max_stoichiometry
                total_atoms = sum(amt_list)
                if total_atoms > self.max_stoichiometry:
                    continue

                # Create composition from stoichiometry amounts (keep full composition, not reduced)
                comp_dict = dict(zip(elems, amt_list, strict=True))
                composition = Composition(comp_dict)

                # Skip elemental proposals (compositions with only one element)
                if len(composition.elements) == 1:
                    continue

                all_compositions.append(composition)

            # Filter by SMACT validity using metrics utility
            valid_compositions = [
                comp for comp in all_compositions if is_smact_valid(comp)
            ]

            for comp in valid_compositions:
                formula = comp.formula
                # Use full formula as key to keep all unique compositions
                if formula not in composition_stats:
                    composition_stats[formula] = CompositionStats(composition=comp)

            logger.info(
                f"Enumerated {len(composition_stats)} SMACT-valid compositions "
                f"from {len(all_compositions)} total compositions"
            )
        else:
            # Original approach: enumerate by total atoms
            # Generate all possible compositions for each total number of atoms from 1 to max_stoichiometry
            for total_atoms in range(1, self.max_stoichiometry + 1):
                # Use combinations_with_replacement to generate all ways to select total_atoms elements
                # This gives us all multisets of size total_atoms from the element list
                for element_combination in combinations_with_replacement(
                    elems, total_atoms
                ):
                    # Count occurrences of each element in the combination
                    comp_dict = {}
                    for elem in element_combination:
                        comp_dict[elem] = comp_dict.get(elem, 0) + 1

                    # Create composition from the counts
                    composition = Composition(comp_dict)

                    # Skip elemental proposals (compositions with only one element)
                    if len(composition.elements) == 1:
                        continue

                    formula = composition.formula
                    composition_stats[formula] = CompositionStats(
                        composition=composition
                    )

        return composition_stats

    def _update_stats_from_state(self, state: dict[str, Any]) -> None:
        """Update composition statistics based on observed results from the environment state."""

        phase_diagram = PhaseDiagram(
            [PDEntry.from_dict(e) for e in state.get("phase_diagram_all_entries", [])]
        )
        for entry in phase_diagram.all_entries:
            composition = entry.composition
            formula = composition.formula
            if formula in self.composition_stats:
                stats = self.composition_stats[formula]
                stats.num_attempts += 1
                if entry in phase_diagram.stable_entries:
                    stats.num_successes += 1
                # score per configured score_function
                if self.score_function == "energy_per_atom":
                    score = -entry.energy_per_atom
                elif self.score_function == "formation_energy_per_atom":
                    score = -phase_diagram.get_form_energy_per_atom(entry)
                elif self.score_function == "e_above_hull":
                    score = -safe_e_above_hull(phase_diagram, entry)
                else:
                    raise ValueError(
                        f"Invalid score_function: {self.score_function}. Expected 'energy_per_atom', 'formation_energy_per_atom', or 'e_above_hull'"
                    )
                stats.best_score = max(stats.best_score, score)
                stats.last_score = score
                stats.last_attempted_step = self.current_step
            else:
                logger.warning(
                    f"Composition {formula} not found in composition stats, skipping (probably over the max stoichiometry or elemental: {self.max_stoichiometry})"
                )
