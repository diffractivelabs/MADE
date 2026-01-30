"""
Cached Generator that generates and caches candidates upfront.

On first generation, generates many candidates using a base generator and caches them.
Subsequent calls return the same cached candidates (excluding already-selected ones).
"""

import logging
import random
from typing import Any

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from ...utils.structure_hash import structure_hash
from ..base import Generator, Plan

logger = logging.getLogger(__name__)


class CachedGenerator(Generator):
    """
    Generator that generates candidates once and caches them for reuse.

    Generates candidates only on the first call, then returns the same cached candidates
    on subsequent calls (excluding already-selected ones that have been removed from cache).

    Supports two modes:
    - Per-composition: Separate candidate cache per composition
    - Global: Single candidate cache across all compositions

    Args:
        base_generator: Base generator to use for creating candidates de novo
        num_initial_candidates: Number of candidates to generate and cache initially
        cache_by_composition: If True, maintain separate cache per composition.
            If False, maintain single global cache.
    """

    def __init__(
        self,
        base_generator: Generator,
        num_initial_candidates: int = 100,
        cache_by_composition: bool = True,
        **kwargs,  # Absorb extra parameters from Hydra config merging
    ) -> None:
        self.base_generator = base_generator
        self.num_initial_candidates = num_initial_candidates
        self.cache_by_composition = cache_by_composition

        # Cache: composition formula -> list[tuple[Structure, str]] (per-composition mode)
        # or None -> list[tuple[Structure, str]] (global mode)
        # Each tuple contains (structure, precomputed_hash) to avoid redundant hash computations
        self._cached_candidates: dict[str | None, list[tuple[Structure, str]]] = {}

        # Track selected structures: set of structure hashes
        # Used to filter out already-selected structures from cache
        self._selected_hashes: set[str] = set()

    def get_state(self) -> dict[str, Any]:
        """Return generator state including full cache."""
        # Serialize cached candidates: convert Structure objects to dicts
        cached_candidates_serialized: dict[
            str | None, list[tuple[dict[str, Any], str]]
        ] = {}
        for key, candidates in self._cached_candidates.items():
            cached_candidates_serialized[key] = [
                (struct.as_dict(), hash_val) for struct, hash_val in candidates
            ]

        state: dict[str, Any] = {
            "cached_candidates": cached_candidates_serialized,  # Full cache for checkpointing
            "selected_hashes": list(
                self._selected_hashes
            ),  # Convert set to list for JSON serialization
        }

        # Include base generator state
        base_state = self.base_generator.get_state()
        if base_state:
            state["base_generator"] = base_state

        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Load generator state including full cache."""
        if "cached_candidates" in state:
            # Deserialize cached candidates: convert dicts back to Structure objects
            from pymatgen.core.structure import Structure

            cached_candidates_serialized = state["cached_candidates"]
            self._cached_candidates = {}
            for key, candidates_serialized in cached_candidates_serialized.items():
                self._cached_candidates[key] = [
                    (Structure.from_dict(struct_dict), hash_val)
                    for struct_dict, hash_val in candidates_serialized
                ]
        if "selected_hashes" in state:
            self._selected_hashes = set(state["selected_hashes"])

        # Load base generator state if present
        if "base_generator" in state and hasattr(self.base_generator, "load_state"):
            self.base_generator.load_state(state["base_generator"])

    def update_state(self, state: dict[str, Any]) -> None:
        """Update generator state and remove selected structures from cache."""
        self.base_generator.update_state(state)

        # Remove selected structure from cache if present
        last_obs = state.get("last_observation")
        if last_obs:
            structure = Structure.from_dict(last_obs["proposal"])
            structure_hash_key = structure_hash(structure)

            # Add to selected set
            self._selected_hashes.add(structure_hash_key)

            # Remove from all caches
            removed_count = 0
            for cache_key in list(self._cached_candidates.keys()):
                candidates = self._cached_candidates[cache_key]
                original_count = len(candidates)

                # Filter out structures matching the selected hash using precomputed hashes
                remaining = [
                    (s, hash_val)
                    for s, hash_val in candidates
                    if hash_val != structure_hash_key
                ]

                if len(remaining) < original_count:
                    removed_count += original_count - len(remaining)
                    if remaining:
                        self._cached_candidates[cache_key] = remaining
                    else:
                        # Cache exhausted for this key
                        del self._cached_candidates[cache_key]
                        logger.info(
                            f"Cache exhausted for {cache_key or 'global mode'} "
                            f"after removing selected structure"
                        )

            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} selected structure(s) from cache "
                    f"(selected hash: {structure_hash_key[:8]}...)"
                )

    def generate(self, plan: Plan, state: dict[str, Any]) -> list[Structure]:
        """
        Generate candidates from cache, or generate and cache if first time.

        Returns all remaining cached candidates (excluding already-selected ones).
        Selected structures are removed from cache in update_state().

        Returns:
            List of all remaining structures in cache (excluding selected ones)
        """
        if not plan.compositions:
            return []

        structures: list[Structure] = []

        if self.cache_by_composition:
            # Per-composition mode: cache separately for each composition
            for composition in plan.compositions:
                comp_formula = composition.formula
                cached = self._get_or_generate_candidates(
                    composition, comp_formula, state, plan
                )

                # Return all remaining candidates (excluding selected ones)
                remaining = self._get_remaining_candidates(cached)
                structures.extend(remaining)
        else:
            # Global mode: single cache for all compositions
            # Use None as the key for global cache
            cached = self._get_or_generate_candidates(None, None, state, plan)

            # Return all remaining candidates (excluding selected ones)
            remaining = self._get_remaining_candidates(cached)
            structures.extend(remaining)

        return structures

    def _get_or_generate_candidates(
        self,
        composition: Composition | None,
        cache_key: str | None,
        state: dict[str, Any],
        plan: Plan,
    ) -> list[tuple[Structure, str]]:
        """
        Get cached candidates or generate and cache if first time.

        Args:
            composition: Composition to generate for (or None for global mode)
            cache_key: Key for cache (composition formula or None)
            state: Current state
            plan: Current plan (needed for global mode to get compositions)

        Returns:
            List of cached candidates as tuples of (structure, precomputed_hash)
        """
        if cache_key in self._cached_candidates:
            return self._cached_candidates[cache_key]

        # First time: generate candidates using base generator
        logger.info(
            f"Generating {self.num_initial_candidates} initial candidates"
            f"{f' for {composition.formula}' if composition else ' (global mode)'}"
        )

        # Create a plan for the base generator
        if composition:
            base_plan = Plan(
                compositions=[composition],
                num_candidates=self.num_initial_candidates,
            )
        else:
            # Global mode: generate candidates for all compositions in the plan
            # Distribute num_initial_candidates across all compositions
            if not plan.compositions:
                raise ValueError(
                    "Global mode requires compositions to be specified in the plan"
                )

            # If more compositions than initial candidates, randomly sample compositions
            if len(plan.compositions) > self.num_initial_candidates:
                logger.info(
                    f"More compositions ({len(plan.compositions)}) than initial candidates "
                    f"({self.num_initial_candidates}). Randomly sampling compositions."
                )
                sampled_compositions = random.sample(
                    plan.compositions, self.num_initial_candidates
                )
                base_plan = Plan(
                    compositions=sampled_compositions,
                    num_candidates=1,  # Generate 1 candidate per sampled composition
                )
            else:
                # Distribute num_initial_candidates across all compositions
                num_per_composition = max(
                    1, self.num_initial_candidates // len(plan.compositions)
                )
                base_plan = Plan(
                    compositions=plan.compositions,
                    num_candidates=num_per_composition,
                )

        candidates = self.base_generator.generate(base_plan, state)

        if not candidates:
            raise RuntimeError(
                f"Base generator produced no candidates for {composition.formula if composition else 'global mode'}"
            )

        # Pre-compute hashes and cache candidates with their hashes
        cached_with_hashes = [(s, structure_hash(s)) for s in candidates]
        self._cached_candidates[cache_key] = cached_with_hashes

        logger.info(
            f"Cached {len(candidates)} candidates"
            f"{f' for {composition.formula}' if composition else ' (global mode)'}"
        )

        return cached_with_hashes

    def _get_remaining_candidates(
        self, cached: list[tuple[Structure, str]]
    ) -> list[Structure]:
        """
        Get all remaining candidates from cache (excluding already-selected ones).

        Args:
            cached: List of cached candidates as tuples of (structure, precomputed_hash)

        Returns:
            List of remaining candidates (excluding selected ones)
        """
        remaining = [
            s for s, hash_val in cached if hash_val not in self._selected_hashes
        ]

        if len(remaining) < len(cached):
            logger.info(
                f"Filtered out {len(cached) - len(remaining)} already-selected candidates "
                f"from {len(cached)} total cached"
            )

        return remaining
