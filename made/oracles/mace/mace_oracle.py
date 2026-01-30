"""
MACE-based oracle - handles MACE calculator setup
"""

from enum import Enum
from typing import Any

import torch
from mace.calculators import mace_mp, mace_off

from made.oracles.ase_potential import ASEPotentialOracle


class MACEModelName(str, Enum):
    """Available MACE model types."""

    MP = "mp"  # Materials Project models
    OFF = "off"  # Organic Force Field models
    CUSTOM = "custom"  # Custom model from file path


class DeviceType(str, Enum):
    """Available device types for computation."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class MACEOracle(ASEPotentialOracle):
    """MACE-based atomic simulator - handles MACE calculator setup."""

    def __init__(
        self,
        model_name: MACEModelName | str = MACEModelName.MP,
        model: str = "medium",
        device: DeviceType | str = DeviceType.AUTO,
        default_dtype: str = "float32",
        dispersion: bool = False,
        model_path: str | None = None,
        element_reference_energies_path: str | None = None,
        relax: bool = True,
        relax_kwargs: dict[str, Any] | None = None,
        num_workers: int = 1,
    ):
        """Initialize MACE atomic simulator.

        Args:
            model_name: Type of MACE model to use ("mp", "off", or "custom")
            model: Model size/name ("small", "medium", "large", or custom model name)
            device: Device to run on ("auto", "cpu", or "cuda")
            default_dtype: Default dtype for calculations ("float32" or "float64")
            dispersion: Whether to include dispersion correction (for mace_mp)
            model_path: Path to custom model file (required if model_type is "custom")
            element_reference_energies_path: Optional path to JSON file containing elemental reference energies
            relax: Whether to relax the structure using ASE
            relax_kwargs: Keyword arguments to pass to the ASE structure optimizer
            num_workers: Number of worker threads for parallel evaluation (default: 1)
        """

        # Allow passing strings via Hydra config and convert to enums
        self.model_name = (
            MACEModelName(model_name) if isinstance(model_name, str) else model_name
        )
        self.device = DeviceType(device) if isinstance(device, str) else device
        if self.device == DeviceType.AUTO:
            self.device = (
                DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
            )
        self.model = model
        self.default_dtype = default_dtype
        self.dispersion = dispersion
        self.model_path = model_path

        # Validate model_name and model_path
        if self.model_name == MACEModelName.CUSTOM and model_path is None:
            raise ValueError("model_path must be provided when model_name is 'custom'")

        device_value = self.device.value

        # Create a factory function that creates a new calculator instance
        # This ensures thread-safety when using multiple workers
        def create_mace_calculator():
            if self.model_name == MACEModelName.MP:
                return mace_mp(
                    model=self.model,
                    device=device_value,
                    default_dtype=self.default_dtype,
                    dispersion=self.dispersion,
                )
            elif self.model_name == MACEModelName.OFF:
                return mace_off(
                    model=self.model,
                    device=device_value,
                    default_dtype=self.default_dtype,
                )
            elif self.model_name == MACEModelName.CUSTOM:
                # For custom models, we need to load from file
                # MACE custom models are typically loaded using MACECalculator
                from mace.calculators import MACECalculator

                return MACECalculator(
                    model_paths=self.model_path,
                    device=device_value,
                    default_dtype=self.default_dtype,
                )
            else:
                raise ValueError(f"Unknown model_name: {self.model_name}")

        # Initialize parent class with calculator factory
        super().__init__(
            calculator=create_mace_calculator,
            element_reference_energies_path=element_reference_energies_path,
            relax=relax,
            relax_kwargs=relax_kwargs or {},
            num_workers=num_workers,
        )
