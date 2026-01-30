"""
Random Structure Generator
"""

import random
from collections.abc import Sequence
from typing import Any

from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from ..base import Generator, Plan


class RandomGenerator(Generator):
    """Generate random crystal structures for provided compositions.

    If the `Plan` includes compositions, one structure is generated per requested
    candidate by cycling through provided compositions. If no compositions are
    given, a fallback composition is constructed using one atom of each element
    provided by the environment state under key `elements`.
    """

    def __init__(
        self,
        seed: int | None = None,
        lattice_length_range: tuple[float, float] = (3.0, 15.0),
        lattice_angle_range: tuple[float, float] = (60.0, 120.0),
        **kwargs,  # Absorb extra parameters from Hydra config merging
    ) -> None:
        self._rng = random.Random(seed)
        self._len_range = lattice_length_range
        self._ang_range = lattice_angle_range

    def get_state(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""

    def update_state(self, state: dict[str, Any]) -> None:
        pass

    def generate(self, plan: Plan, state: dict[str, Any]) -> list[Structure]:
        num = max(int(plan.num_candidates), 1)
        compositions: list[Composition] = list(plan.compositions)

        # Fallback composition if none provided in plan
        if not compositions:
            elements: Sequence[str] = state.get("elements", [])
            if not elements:
                raise ValueError(
                    "RandomGenerator requires 'elements' in state when plan.compositions is empty"
                )
            fallback = Composition(dict.fromkeys(elements, 1))
            compositions = [fallback]

        structures: list[Structure] = []
        for comp in compositions:
            for _ in range(num):
                structures.append(self._random_structure_from_composition(comp))
        return structures

    def _random_structure_from_composition(self, composition: Composition) -> Structure:
        a = self._rng.uniform(*self._len_range)
        b = self._rng.uniform(*self._len_range)
        c = self._rng.uniform(*self._len_range)
        alpha = self._rng.uniform(*self._ang_range)
        beta = self._rng.uniform(*self._ang_range)
        gamma = self._rng.uniform(*self._ang_range)

        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        species: list[str] = []
        coords: list[list[float]] = []
        for element, amount in composition.items():
            for _ in range(int(amount)):
                species.append(str(element))
                coords.append(
                    [
                        self._rng.random(),
                        self._rng.random(),
                        self._rng.random(),
                    ]
                )

        return Structure(lattice, species, coords)
