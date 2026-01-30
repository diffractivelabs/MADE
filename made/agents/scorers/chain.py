"""
Multiobjective Scorer that chains multiple scorers together.

Combines scores from multiple scorers to enable multiobjective optimization.
Supports weighted combination and Pareto-based selection.
"""

from typing import Any

import numpy as np
from pymatgen.core.structure import Structure

from made.agents.base import Scorer, ScoreResult


class ScorerChain(Scorer):
    """
    Chain multiple scorers together for multiobjective scoring.

    Combines scores from multiple scorers using weighted sum or Pareto dominance.
    Optionally normalizes scorer scores to [0, 1] before combination.

    Args:
        scorers: List of scorers to combine
        weights: Optional list of weights for each scorer (default: equal weights)
        combination_method: "weighted_sum" or "pareto" (default: "weighted_sum")
            - "weighted_sum": Combine scores using weighted sum
            - "pareto": Select from Pareto-optimal candidates (all objectives maximized)
        normalize_scores: Whether to normalize each scorer's scores to [0, 1] before
            combination (default: True). If False, scores are used as-is (may need
            manual scaling to ensure comparable ranges).
    """

    scorer_name = "ScorerChain"

    def __init__(
        self,
        scorers: list[Scorer],
        weights: list[float] | None = None,
        combination_method: str = "weighted_sum",
        normalize_scores: bool = True,
    ) -> None:
        if not scorers:
            raise ValueError("ScorerChain requires at least one scorer")
        self.scorers = scorers

        # Normalize weights
        if weights is None:
            weights = [1.0] * len(scorers)
        if len(weights) != len(scorers):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of scorers ({len(scorers)})"
            )
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum == 0:
            raise ValueError("Weights cannot all be zero")
        self.weights = [w / weight_sum for w in weights]

        if combination_method not in ["weighted_sum", "pareto"]:
            raise ValueError(
                f"combination_method must be 'weighted_sum' or 'pareto', got '{combination_method}'"
            )
        self.combination_method = combination_method
        self.normalize_scores = normalize_scores

    def get_state(self) -> dict[str, Any]:
        """Return the current state of all scorers in the chain."""
        return {
            "scorers": [s.get_state() for s in self.scorers],
            "weights": self.weights,
            "combination_method": self.combination_method,
            "normalize_scores": self.normalize_scores,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load the state of all scorers in the chain."""
        if "scorers" in state:
            for scorer, scorer_state in zip(
                self.scorers, state["scorers"], strict=True
            ):
                if hasattr(scorer, "load_state"):
                    scorer.load_state(scorer_state)
        if "weights" in state:
            self.weights = state["weights"]
        if "combination_method" in state:
            self.combination_method = state["combination_method"]
        if "normalize_scores" in state:
            self.normalize_scores = state["normalize_scores"]

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of all scorers in the chain."""
        for scorer in self.scorers:
            scorer.update_state(state)

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: List of scores to normalize

        Returns:
            Normalized scores
        """
        if not self.normalize_scores:
            return scores

        scores_array = np.array(scores)
        score_min = scores_array.min()
        score_max = scores_array.max()
        if score_max == score_min:
            # All scores are the same, set to 0.5
            return np.full_like(scores_array, 0.5).tolist()
        else:
            normalized = (scores_array - score_min) / (score_max - score_min)
            return normalized.tolist()

    def _combine_scores(
        self, normalized_scores: list[list[float]], num_candidates: int
    ) -> list[float]:
        """
        Combine normalized scores using the configured combination method.

        Args:
            normalized_scores: List of normalized score lists (one per scorer)
            num_candidates: Number of candidates

        Returns:
            Combined scores
        """
        if self.combination_method == "weighted_sum":
            # Weighted sum of normalized scores
            combined_scores = [0.0] * num_candidates
            for i, normalized in enumerate(normalized_scores):
                weight = self.weights[i]
                for j, score in enumerate(normalized):
                    combined_scores[j] += weight * score
            return combined_scores

        elif self.combination_method == "pareto":
            # Pareto dominance: count how many objectives each candidate dominates
            # Candidates on the Pareto frontier get higher scores
            num_objectives = len(normalized_scores)

            # For each candidate, count how many other candidates it dominates
            # A candidate dominates another if it's better in at least one objective
            # and not worse in any objective
            dominance_scores = [0.0] * num_candidates
            for i in range(num_candidates):
                for j in range(num_candidates):
                    if i == j:
                        continue
                    # Check if candidate i dominates candidate j
                    better_in_any = False
                    worse_in_any = False
                    for obj_idx in range(num_objectives):
                        if (
                            normalized_scores[obj_idx][i]
                            > normalized_scores[obj_idx][j]
                        ):
                            better_in_any = True
                        elif (
                            normalized_scores[obj_idx][i]
                            < normalized_scores[obj_idx][j]
                        ):
                            worse_in_any = True

                    if better_in_any and not worse_in_any:
                        dominance_scores[i] += 1.0

            # Normalize dominance scores to [0, 1]
            dominance_array = np.array(dominance_scores)
            if dominance_array.max() == dominance_array.min():
                return [1.0] * num_candidates  # All equal
            normalized_dominance = (dominance_array - dominance_array.min()) / (
                dominance_array.max() - dominance_array.min()
            )
            return normalized_dominance.tolist()

        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _score_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[float], list[ScoreResult]]:
        """
        Score candidates using all scorers and combine the scores.

        Returns combined scores where higher values are better, plus detailed results.
        """
        if not candidates:
            raise ValueError("No candidates provided")

        # Score with each scorer, collecting results
        all_scores: list[list[float]] = []
        all_scorer_results: list[list[ScoreResult]] = []
        for scorer in self.scorers:
            scores, results = scorer.score_candidates(
                candidates, state, return_results=True
            )
            if len(scores) != len(candidates):
                raise ValueError(
                    f"Scorer {scorer} returned {len(scores)} scores for {len(candidates)} candidates"
                )
            all_scores.append(scores)
            all_scorer_results.append(results)

        # Normalize each scorer's scores
        normalized_scores: list[list[float]] = [
            self._normalize_scores(scores) for scores in all_scores
        ]

        # Combine scores
        combined_scores = self._combine_scores(normalized_scores, len(candidates))

        # Build combined results
        score_results = []
        for i in range(len(candidates)):
            # Collect per-scorer info for this candidate
            scorer_details = []
            for j, scorer in enumerate(self.scorers):
                scorer_details.append(
                    {
                        "scorer": getattr(scorer, "scorer_name", type(scorer).__name__),
                        "raw_score": all_scores[j][i],
                        "normalized_score": normalized_scores[j][i],
                        "weight": self.weights[j],
                        "details": all_scorer_results[j][i].details
                        if all_scorer_results[j][i].details
                        else {},
                    }
                )

            score_results.append(
                ScoreResult(
                    score=combined_scores[i],
                    scorer_name=self.scorer_name,
                    details={
                        "combination_method": self.combination_method,
                        "scorer_breakdown": scorer_details,
                    },
                )
            )

        return combined_scores, score_results
