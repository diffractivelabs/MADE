"""
Surrogate Scorer - uses an oracle (MLIP) to evaluate structures and score them.

This scorer acts as a surrogate model for pre-ranking candidates before ground-truth
oracle evaluation. It wraps any Oracle implementation (MACE, ORB, CHGNet, etc.) and
converts evaluation results into scores for selection.

Key features:
- Batching: Evaluates multiple structures efficiently
- Caching: Avoids redundant surrogate evaluations
- Reranking: Can recompute e_above_hull scores when phase diagram changes (using cached energies)

Note: This is NOT the ground-truth oracle - it's a surrogate model for candidate ranking.
"""

import hashlib
import logging
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Structure

from made.agents.base import Scorer, ScoreResult
from made.oracles.base import Oracle
from made.utils.convex_hull_utils import (
    safe_e_above_hull,
    structure_result_to_entry,
)
from made.utils.structure_hash import structure_hash

logger = logging.getLogger(__name__)


class OracleScorer(Scorer):
    """
    Wrap any `Oracle` to score candidate structures (acts as a surrogate model).

    Supports caching and reranking when state changes.
    Batching is handled internally by the oracle.

    Args:
        oracle: Oracle instance used to evaluate structures (e.g., MACE, ORB, CHGNet)
        score_function: One of "energy_per_atom", "formation_energy_per_atom", "e_above_hull"
        enable_cache: Whether to cache oracle results (default: True)
        rerank_on_state_change: Whether to recompute scores when state changes (default: False)
            Only relevant for e_above_hull scoring.
    """

    scorer_name = "SurrogateScorer"  # Default, overridden in __init__

    def __init__(
        self,
        oracle: Oracle,
        score_function: str = "energy_per_atom",
        enable_cache: bool = True,
        rerank_on_state_change: bool = False,
    ) -> None:
        self.oracle = oracle
        self.score_function = score_function
        self.enable_cache = enable_cache
        self.rerank_on_state_change = rerank_on_state_change

        # Set scorer name to include oracle type for clarity
        oracle_name = getattr(oracle, "name", oracle.__class__.__name__)
        self.scorer_name = f"surrogate_{oracle_name}"

        logger.info(
            f"[SurrogateScorer] Initialized with oracle={oracle_name}, score_function={score_function}, cache={enable_cache}"
        )

        # Cache: structure_hash -> oracle_result
        self._cache: dict[str, dict[str, Any]] = {}

        # Track last phase diagram state for reranking detection
        self._last_pd_state_hash: str | None = None

    def get_state(self) -> dict[str, Any]:
        """Return scorer state including full cache."""
        return {
            "cache": self._cache.copy(),  # Full cache for checkpointing
            "last_pd_state_hash": self._last_pd_state_hash,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load scorer state including full cache."""
        if "cache" in state:
            self._cache = state["cache"].copy()
        if "last_pd_state_hash" in state:
            self._last_pd_state_hash = state["last_pd_state_hash"]

    def update_state(self, state: dict[str, Any]) -> None:
        """Update scorer state."""
        # State tracking for reranking is now handled in score_candidates

    # ---- Scorer API ----
    def _score_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[float], list[ScoreResult]]:
        """
        Score all candidates with caching and return detailed results.

        Returns scores in the same order as candidates.
        Batching is handled by the oracle internally.
        """
        if not candidates:
            raise ValueError("No candidates provided")

        # Hash all candidates
        candidate_hashes = [structure_hash(c) for c in candidates]

        # Find uncached structures
        uncached_indices: list[int] = []
        uncached_structures: list[Structure] = []

        if not self.enable_cache:
            uncached_indices = list(range(len(candidates)))
            uncached_structures = candidates
            logger.info(
                f"[{self.scorer_name}] Cache disabled: evaluating all {len(candidates)} candidates"
            )
        else:
            for i, hash_key in enumerate(candidate_hashes):
                if hash_key not in self._cache:
                    uncached_indices.append(i)
                    uncached_structures.append(candidates[i])

            cached_count = len(candidates) - len(uncached_structures)
            if cached_count > 0:
                logger.info(
                    f"[{self.scorer_name}] Cache hit: {cached_count}/{len(candidates)} candidates"
                )
            if uncached_structures:
                logger.info(
                    f"[{self.scorer_name}] Evaluating {len(uncached_structures)}/{len(candidates)} candidates"
                )

        # Evaluate uncached structures (oracle handles batching internally)
        if uncached_structures:
            logger.info(
                f"[{self.scorer_name}] Calling batch_evaluate for {len(uncached_structures)} structures"
            )
            results = self.oracle.batch_evaluate(uncached_structures)

            # Store results in cache
            if self.enable_cache:
                for i, hash_key in enumerate(
                    [candidate_hashes[idx] for idx in uncached_indices]
                ):
                    self._cache[hash_key] = results[i]
                logger.info(
                    f"[{self.scorer_name}] Cached {len(uncached_indices)} evaluations "
                    f"(total cache size: {len(self._cache)})"
                )

        # For e_above_hull with reranking enabled, check if we can skip oracle evaluation
        # and recompute using cache (when state changed and all candidates are cached)
        if (
            self.score_function == "e_above_hull"
            and self.rerank_on_state_change
            and self.enable_cache
        ):
            pd_entries_dicts = state.get("phase_diagram_all_entries")
            if pd_entries_dicts:
                pd_state_str = str(sorted(str(e) for e in pd_entries_dicts))
                current_pd_state_hash = hashlib.sha256(
                    pd_state_str.encode("utf-8")
                ).hexdigest()

                # If state changed and all candidates are cached, recompute e_above_hull using cache
                if (
                    self._last_pd_state_hash is not None
                    and current_pd_state_hash != self._last_pd_state_hash
                ):
                    missing_from_cache = [
                        i
                        for i, hash_key in enumerate(candidate_hashes)
                        if hash_key not in self._cache
                    ]
                    if not missing_from_cache:
                        # Fast path: recompute e_above_hull using cached energies (no surrogate evaluation needed)
                        logger.info(
                            f"[{self.scorer_name}] Phase diagram updated: recomputing e_above_hull for "
                            f"{len(candidates)} candidates using cached energies (fast reranking)"
                        )
                        pd = PhaseDiagram(
                            [PDEntry.from_dict(e) for e in pd_entries_dicts]
                        )
                        scores = []
                        score_results = []
                        for structure in candidates:
                            hash_key = structure_hash(structure)
                            result = self._cache[hash_key]
                            candidate_entry = structure_result_to_entry(
                                structure, result
                            )
                            e_hull = float(safe_e_above_hull(pd, candidate_entry))
                            # Negate so lower e_above_hull → higher score (better)
                            score = -e_hull
                            scores.append(score)
                            score_results.append(
                                ScoreResult(
                                    score=score,
                                    scorer_name=self.scorer_name,
                                    details={
                                        "score_function": self.score_function,
                                        "e_above_hull": e_hull,
                                        "energy_per_atom": result.get(
                                            "energy_per_atom"
                                        ),
                                        "cached": True,
                                        "reranked": True,
                                    },
                                )
                            )
                        # Update state hash
                        self._last_pd_state_hash = current_pd_state_hash
                        return scores, score_results
                    else:
                        logger.warning(
                            f"[{self.scorer_name}] Phase diagram updated but {len(missing_from_cache)}/{len(candidates)} "
                            f"candidates missing from cache. Falling back to normal evaluation."
                        )

                # Update state hash (first time or no change)
                self._last_pd_state_hash = current_pd_state_hash

        # Collect all surrogate results (normal path or first time with reranking)
        oracle_results = []
        cache_failures = []
        for i, hash_key in enumerate(candidate_hashes):
            if self.enable_cache and hash_key in self._cache:
                oracle_results.append(self._cache[hash_key])
            else:
                # Shouldn't happen if caching worked, but fallback
                if self.enable_cache:
                    cache_failures.append(i)
                result = self.oracle.evaluate(candidates[i])
                if self.enable_cache:
                    self._cache[hash_key] = result
                oracle_results.append(result)

        if cache_failures:
            logger.warning(
                f"[{self.scorer_name}] Cache failure: {len(cache_failures)} candidates not in cache. "
                f"Evaluated individually as fallback."
            )

        # Compute scores from surrogate results
        # Note: We negate energy-based scores so that higher scores = better
        # (lower energy → higher negated score → better)
        scores = []
        score_results = []
        if self.score_function == "energy_per_atom":
            for result in oracle_results:
                if "energy_per_atom" not in result:
                    raise KeyError(
                        f"[{self.scorer_name}] Result missing required key 'energy_per_atom' for score_function='{self.score_function}'. Available: {list(result.keys())}"
                    )
                energy = float(result["energy_per_atom"])
                # Negate so lower energy → higher score (better)
                score = -energy
                scores.append(score)
                score_results.append(
                    ScoreResult(
                        score=score,
                        scorer_name=self.scorer_name,
                        details={
                            "score_function": self.score_function,
                            "energy_per_atom": energy,
                        },
                    )
                )
        elif self.score_function in ["formation_energy_per_atom", "e_above_hull"]:
            # Build PD from observed entries in state
            pd_entries_dicts = state.get("phase_diagram_all_entries")
            if not pd_entries_dicts:
                raise ValueError(
                    "State must provide 'phase_diagram_all_entries' to compute formation energy or e_above_hull."
                )
            pd = PhaseDiagram([PDEntry.from_dict(e) for e in pd_entries_dicts])

            for structure, result in zip(candidates, oracle_results, strict=True):
                # Create PDEntry for candidate from oracle results
                candidate_entry = structure_result_to_entry(structure, result)

                if self.score_function == "formation_energy_per_atom":
                    form_energy = float(pd.get_form_energy_per_atom(candidate_entry))
                    # Negate so lower formation energy → higher score (better)
                    score = -form_energy
                    scores.append(score)
                    score_results.append(
                        ScoreResult(
                            score=score,
                            scorer_name=self.scorer_name,
                            details={
                                "score_function": self.score_function,
                                "formation_energy_per_atom": form_energy,
                                "energy_per_atom": result.get("energy_per_atom"),
                            },
                        )
                    )
                else:
                    e_hull = float(safe_e_above_hull(pd, candidate_entry))
                    # Negate so lower e_above_hull → higher score (better)
                    score = -e_hull
                    scores.append(score)
                    score_results.append(
                        ScoreResult(
                            score=score,
                            scorer_name=self.scorer_name,
                            details={
                                "score_function": self.score_function,
                                "e_above_hull": e_hull,
                                "energy_per_atom": result.get("energy_per_atom"),
                            },
                        )
                    )
        else:
            raise ValueError(f"Invalid score function: {self.score_function}")

        return scores, score_results
