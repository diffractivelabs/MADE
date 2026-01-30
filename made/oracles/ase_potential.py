"""
ASE-based oracle that reports potential energy from an attached ASE calculator.

Intended for use with MLIPs (e.g., MACE, NequIP, CHGNet-ASE wrappers) or DFT
calculators. It converts `pymatgen` Structures to ASE `Atoms`, evaluates the
potential energy, and returns both total and per-atom energies.
"""

import copy
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from made.oracles.base import Oracle

logger = logging.getLogger(__name__)


class ASEPotentialOracle(Oracle):
    """Oracle that evaluates energy using an ASE calculator.

    Parameters
    - calculator: An ASE `Calculator` instance, or a callable returning one.
    - element_reference_energies: Optional map of elemental symbol to reference energy per atom (e.g. from materials project)
    - element_reference_energies_path: Optional path to a JSON file containing elemental reference energies.
      If provided and `element_reference_energies` is not passed, this file will be loaded.
    - relax: Whether to relax the structure using ASE.
    - relax_kwargs: Keyword arguments to pass to the ASE structure optimizer (e.g. optimizer, maxstep, alpha, fmax, steps)
    """

    def __init__(
        self,
        calculator: Calculator | Callable[[], Calculator],
        element_reference_energies_path: str | None = None,
        relax: bool = True,
        relax_kwargs: dict[str, Any] | None = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(num_workers=num_workers)
        # Store calculator factory for thread-safe creation
        if callable(calculator):
            self._calculator_factory = calculator
            # Create initial calculator for single-threaded case
            self.calculator: Calculator = calculator()
        else:
            # For non-callable calculators, try to create copies for thread-safety
            calc_instance = calculator

            # Try to use copy.deepcopy if the calculator supports it
            # Some calculators (like ORB) may have complex state that can't be easily copied
            def create_calc_copy():
                try:
                    # Attempt deep copy
                    return copy.deepcopy(calc_instance)
                except (TypeError, AttributeError, ValueError) as e:
                    logger.warning(
                        f"Could not copy calculator for thread-local use: {e}. "
                        "Using same instance - this may cause thread-safety issues with num_workers > 1. "
                        "Consider passing a callable factory instead."
                    )
                    return calc_instance

            self._calculator_factory = create_calc_copy
            self.calculator: Calculator = calc_instance
        self._thread_local = threading.local()
        self._element_reference_energies: dict[str, float] = {}
        self._adaptor = AseAtomsAdaptor()
        self.relax = relax
        self.relax_kwargs = relax_kwargs or {}

        if element_reference_energies_path is not None:
            logger.info(
                f"Loading elemental reference energies from {element_reference_energies_path}"
            )
            self._element_reference_energies.update(
                self._load_element_reference_energies_from_file(
                    element_reference_energies_path
                )
            )
        else:
            logger.warning(
                "No elemental reference energies provided, make sure to compute them using compute_elemental_energies_from_structures using the oracle"
            )
            self._element_reference_energies = {}

    def _get_thread_calculator(self) -> Calculator:
        """Get a thread-local calculator instance."""
        if not hasattr(self._thread_local, "calculator"):
            self._thread_local.calculator = self._calculator_factory()
        return self._thread_local.calculator

    # ---- Public API ----
    def evaluate(self, structure: Structure) -> dict[str, Any]:
        # Use thread-local calculator if using multiple workers, otherwise use shared one
        if self.num_workers > 1:
            calc = self._get_thread_calculator()
        else:
            calc = self.calculator

        atoms = self._adaptor.get_atoms(structure)
        atoms.calc = calc
        if self.relax:
            atoms = self.relax_structure(atoms, calc=calc, **self.relax_kwargs)

        total_energy = float(atoms.get_potential_energy())
        num_atoms = int(len(atoms))
        energy_per_atom = (
            total_energy / float(num_atoms) if num_atoms > 0 else float("nan")
        )

        return {
            "energy": total_energy,
            "energy_per_atom": energy_per_atom,
            "natoms": num_atoms,
            "formula": structure.composition.formula,
        }

    def relax_structure(
        self, atoms: Atoms, calc: Calculator | None = None, **kwargs
    ) -> Atoms:
        """Relax a structure using ASE."""
        # Create a copy to avoid modifying input
        atoms = atoms.copy()
        if kwargs.get("perturb_structure"):
            atoms.rattle(stdev=kwargs.get("perturb_structure", 0.05))

        # Set up calculator (use provided calc or fall back to default)
        atoms.calc = calc if calc is not None else self.calculator

        ecf = FrechetCellFilter(atoms) if kwargs.get("relax_unit_cell", True) else atoms

        optimizer_type = kwargs.get("optimizer", "fire")

        if optimizer_type == "fire":
            optimizer = FIRE(ecf)
        elif optimizer_type == "bfgs":
            optimizer = BFGS(
                ecf,
                maxstep=kwargs.get("maxstep", 0.04),
                alpha=kwargs.get("alpha", 70.0),
            )
        elif optimizer_type == "lbfgs":
            optimizer = LBFGS(ecf)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        optimizer.run(fmax=kwargs.get("fmax", 0.02), steps=kwargs.get("steps", 500))
        return atoms

    # Optional helpers used by environments for seeding elemental references
    @property
    def element_reference_energies(self) -> dict[str, float]:
        return dict(self._element_reference_energies)

    def get_element_reference_energy(self, symbol: str) -> float:
        if symbol not in self._element_reference_energies:
            raise KeyError(f"No reference energy available for element '{symbol}'.")
        return float(self._element_reference_energies[symbol])

    def compute_elemental_energies_from_structures(
        self, elemental_entries_with_structures: list[tuple[Any, Structure]]
    ) -> dict[str, float]:
        """
        Compute elemental reference energies using the oracle from stable elemental structures.

        Args:
            elemental_entries_with_structures: List of (PDEntry, Structure) tuples for elemental entries

        Returns:
            Dictionary mapping element symbols to energy per atom
        """
        computed_energies = {}

        for entry, structure in elemental_entries_with_structures:
            # Get the element symbol from the entry's composition
            composition = entry.composition
            if len(composition.elements) != 1:
                raise ValueError(f"Expected elemental composition, got {composition}")

            element_symbol = str(composition.elements[0])

            # Evaluate the structure using the oracle
            oracle_result = self.evaluate(structure)

            # Extract energy per atom
            if "energy_per_atom" in oracle_result:
                energy_per_atom = float(oracle_result["energy_per_atom"])
            elif "energy" in oracle_result:
                energy_per_atom = float(oracle_result["energy"]) / float(
                    structure.num_sites
                )
            else:
                raise ValueError(
                    "Oracle result must contain 'energy' or 'energy_per_atom'"
                )

            computed_energies[element_symbol] = energy_per_atom
            logger.info(
                f"Computed elemental energy for {element_symbol}: {energy_per_atom:.6f} eV/atom"
            )

        # Update the oracle's internal reference energies
        self._element_reference_energies.update(computed_energies)

        return computed_energies

    def set_element_reference_energies(self, energies: dict[str, float]) -> None:
        """Set elemental reference energies directly."""
        self._element_reference_energies.update(energies)

    # ---- Internals ----
    @staticmethod
    def _load_element_reference_energies_from_file(path: str) -> dict[str, float]:
        """Load elemental reference energies from a JSON file."""
        with open(
            path,
        ) as f:
            ref_energies = json.load(f)
            ref_energies = {
                k: v["energy"] / v["composition"][k] for k, v in ref_energies.items()
            }
        return ref_energies
