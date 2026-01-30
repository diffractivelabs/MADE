from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from pymatgen.core.structure import Structure

from ..base import Generator, Plan

logger = logging.getLogger(__name__)


class ChemeleonGenerator(Generator):
    def __init__(
        self,
        task: str = "csp",
        batch_size: int = 32,
        device: str = "cuda",
        num_atom_distribution: str | dict[int, float] | None = None,
        output_dir: str | None = None,
        **kwargs,  # Absorb extra parameters from Hydra config merging
    ) -> None:
        """
        Initialize ChemeleonGenerator.

        Args:
            task: Task type - "csp" (Crystal Structure Prediction) or "dng" (De Novo Generation)
            batch_size: Batch size for generation
            device: Device to use ("cpu" or "cuda")
            num_atom_distribution: Distribution for number of atoms (DNG only), e.g., "mp-20"
            output_dir: Optional output directory for CIF files
        """
        self.task = task
        self.batch_size = batch_size
        self.device = device
        self.num_atom_distribution = num_atom_distribution
        self.output_dir = output_dir
        self.dm = None

    def get_state(self) -> dict[str, Any]:
        return {}

    def update_state(self, state: dict[str, Any]) -> None:
        pass

    def setup(self) -> None:
        """Initialize the diffusion model."""
        if self.dm is not None:
            return

        from chemeleon_dng import sample
        from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
        from chemeleon_dng.download_util import get_checkpoint_path

        model_path = get_checkpoint_path(self.task, sample.DEFAULT_MODEL_PATH)
        self.dm = DiffusionModule.load_from_checkpoint(
            model_path, map_location=self.device
        )
        logger.info(f"Loaded Chemeleon model from {model_path}")

    def generate(self, plan: Plan, state: dict[str, Any]) -> list[Structure]:
        """
        Generate crystal structures using Chemeleon.

        Args:
            plan: Plan containing compositions and number of candidates
            state: Current state dictionary

        Returns:
            List of pymatgen Structure objects
        """
        # Ensure model is loaded
        if self.dm is None:
            self.setup()

        from chemeleon_dng import sample as sample_mod

        with tempfile.TemporaryDirectory(prefix="chemeleon_samples_") as tmpdir:
            out_dir = Path(tmpdir) if self.output_dir is None else Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            num_samples = max(int(plan.num_candidates), 1)

            if self.task == "csp":
                if not hasattr(sample_mod, "sample_csp"):
                    raise RuntimeError(
                        "chemeleon_dng.sample.sample_csp not available in this version"
                    )

                if not plan.compositions:
                    raise ValueError("CSP task requires compositions in the plan")

                formulas = [str(c) for c in plan.compositions]
                samples_per_formula = num_samples // len(formulas) or 1

                logger.info(
                    f"Running Chemeleon CSP with formulas: {formulas}, "
                    f"{samples_per_formula} samples per formula"
                )

                sample_mod.sample_csp(
                    dm=self.dm,
                    formulas=formulas,
                    num_samples=samples_per_formula,
                    batch_size=self.batch_size,
                    output_path=out_dir,
                )

            elif self.task == "dng":
                if not hasattr(sample_mod, "sample_dng"):
                    raise RuntimeError(
                        "chemeleon_dng.sample.sample_dng not available in this version"
                    )

                logger.info(
                    f"Running Chemeleon DNG with {num_samples} samples, "
                    f"batch_size={self.batch_size}"
                )

                sample_mod.sample_dng(
                    dm=self.dm,
                    num_samples=num_samples,
                    batch_size=self.batch_size,
                    output_path=out_dir,
                    num_atom_distribution=self.num_atom_distribution or "mp-20",
                )

            else:
                raise ValueError(f"Unknown task: {self.task}. Must be 'csp' or 'dng'")

            # Read generated CIF files
            cif_paths = sorted(out_dir.rglob("*.cif"))
            if not cif_paths:
                raise RuntimeError(f"No CIF files found in {out_dir}")

            structures = []
            for cif in cif_paths:
                try:
                    structure = Structure.from_file(str(cif))
                    structures.append(structure)
                except Exception as e:
                    logger.warning(f"Failed to read {cif} as a CIF file: {e}")
                    continue

            logger.info(f"Loaded {len(structures)} structures from {out_dir}")
            return structures
