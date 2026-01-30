"""
Analytic oracle that uses a pre-computed formula to score structures.
"""

from typing import Any

from ase.calculators.lj import LennardJones

from made.oracles.ase_potential import ASEPotentialOracle


class AnalyticOracle(ASEPotentialOracle):
    """ORB-based atomic simulator - only handles ORB calculator setup."""

    def __init__(
        self,
        model_name: str,
        element_reference_energies_path: str | None = None,
        relax: bool = True,
        relax_kwargs: dict[str, Any] | None = None,
        num_workers: int = 1,
    ):
        """Initialize Analytic atomic simulator.

        Args:
            model_name: Analytic model to use
            element_reference_energies_path: Path to a JSON file containing elemental reference energies.
            relax: Whether to relax the structure using ASE.
            relax_kwargs: Keyword arguments to pass to the ASE structure optimizer.
            num_workers: Number of worker threads for parallel evaluation (default: 1)
        """
        model_name = model_name.lower()

        if model_name == "lj":
            calculator = LennardJones()
        else:
            raise ValueError(f"Unknown analytic model: {model_name}")

        # Initialize parent class with calculator
        super().__init__(
            calculator=calculator,
            element_reference_energies_path=element_reference_energies_path,
            relax=relax,
            relax_kwargs=relax_kwargs or {},
            num_workers=num_workers,
        )
