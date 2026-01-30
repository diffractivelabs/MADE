from typing import Any

import numpy as np
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from ..base import Scorer, ScoreResult


class CompositionDiversity(Scorer):
    scorer_name = "CompositionDiversity"

    def __init__(
        self, distance_metric: str = "euclidean", seed: int | None = None
    ) -> None:
        self.distance_metric = distance_metric
        self.rng = np.random.RandomState(seed)

    def get_state(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""

    def update_state(self, state: dict[str, Any]) -> None:
        pass

    def _score_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[float], list[ScoreResult]]:
        """
        Score candidates based on diversity from observed compositions.

        Returns scores where higher = more diverse (better).
        """
        if not candidates:
            raise ValueError("No candidates provided")
        if len(candidates) == 1:
            return [1.0], [
                ScoreResult(
                    score=1.0,
                    scorer_name=self.scorer_name,
                    details={"reason": "single candidate"},
                )
            ]

        observed = self._extract_observed_compositions(state)
        if not observed:
            # No observed compositions, return random scores
            scores = [float(self.rng.rand()) for _ in candidates]
            results = [
                ScoreResult(
                    score=s,
                    scorer_name=self.scorer_name,
                    details={"reason": "no observed compositions, random"},
                )
                for s in scores
            ]
            return scores, results

        all_comps = observed + [s.composition for s in candidates]
        mat = self._compositions_to_matrix(all_comps)
        n_obs = len(observed)
        obs_mat = mat[:n_obs]
        cand_mat = mat[n_obs:]
        min_dists = self._compute_vectorized_distances(cand_mat, obs_mat)
        # Return distances as scores (higher distance = more diverse = better)
        scores = [float(d) for d in min_dists]
        results = [
            ScoreResult(
                score=s,
                scorer_name=self.scorer_name,
                details={
                    "min_distance_to_observed": s,
                    "distance_metric": self.distance_metric,
                },
            )
            for s in scores
        ]
        return scores, results

    def _extract_observed_compositions(
        self, state: dict[str, Any]
    ) -> list[Composition]:
        compositions: list[Composition] = []
        if "phase_diagram_all_entries" in state:
            for e in state["phase_diagram_all_entries"]:
                if "composition" in e:
                    try:
                        compositions.append(Composition(e["composition"]))
                    except Exception:
                        continue
        else:
            raise ValueError(
                "CompositionDiversity requires 'phase_diagram_all_entries' in state"
            )
        return compositions

    def _compositions_to_matrix(self, compositions: list[Composition]) -> np.ndarray:
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
        self, candidate_matrix: np.ndarray, observed_matrix: np.ndarray
    ) -> np.ndarray:
        if self.distance_metric == "euclidean":
            diff = candidate_matrix[:, None, :] - observed_matrix[None, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
        elif self.distance_metric == "manhattan":
            diff = candidate_matrix[:, None, :] - observed_matrix[None, :, :]
            distances = np.sum(np.abs(diff), axis=2)
        elif self.distance_metric == "cosine":
            cand_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
            obs_norms = np.linalg.norm(observed_matrix, axis=1, keepdims=True)
            cand_norms = np.where(cand_norms == 0, 1, cand_norms)
            obs_norms = np.where(obs_norms == 0, 1, obs_norms)
            cand_norm = candidate_matrix / cand_norms
            obs_norm = observed_matrix / obs_norms
            cos_sim = np.dot(cand_norm, obs_norm.T)
            distances = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        return np.min(distances, axis=1)
