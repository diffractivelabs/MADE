from typing import Any

import numpy as np
from pymatgen.core.structure import Structure

from ..base import Scorer, ScoreResult


class RandomSelector(Scorer):
    scorer_name = "RandomSelector"

    def __init__(self, seed: int | None = None) -> None:
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
        """Score candidates with random values."""
        if not candidates:
            raise ValueError("No candidates provided")

        scores = [float(self.rng.rand()) for _ in candidates]
        results = [ScoreResult(score=s, scorer_name=self.scorer_name) for s in scores]
        return scores, results
