"""
ORB-based oracle - only handles ORB calculator setup
"""

from enum import Enum
from typing import Any

import torch
from orb_models.forcefield.calculator import ORBCalculator

from made.oracles.ase_potential import ASEPotentialOracle


class ORBModelName(str, Enum):
    """Available ORB model names."""

    # ORB v3 Conservative models
    ORB_V3_CONSERVATIVE_INF_OMAT = "orb-v3-conservative-inf-omat"
    ORB_V3_CONSERVATIVE_20_OMAT = "orb-v3-conservative-20-omat"
    ORB_V3_CONSERVATIVE_INF_MPA = "orb-v3-conservative-inf-mpa"
    ORB_V3_CONSERVATIVE_20_MPA = "orb-v3-conservative-20-mpa"

    # ORB v3 Direct models
    ORB_V3_DIRECT_INF_OMAT = "orb-v3-direct-inf-omat"
    ORB_V3_DIRECT_20_OMAT = "orb-v3-direct-20-omat"
    ORB_V3_DIRECT_INF_MPA = "orb-v3-direct-inf-mpa"
    ORB_V3_DIRECT_20_MPA = "orb-v3-direct-20-mpa"

    # ORB v2 models
    ORB_V2 = "orb-v2"
    ORB_D3_V2 = "orb-d3-v2"
    ORB_D3_SM_V2 = "orb-d3-sm-v2"
    ORB_D3_XS_V2 = "orb-d3-xs-v2"
    ORB_MPTRAJ_ONLY_V2 = "orb-mptraj-only-v2"

    # ORB v1 models
    ORB_V1 = "orb-v1"
    ORB_D3_V1 = "orb-d3-v1"
    ORB_D3_SM_V1 = "orb-d3-sm-v1"
    ORB_D3_XS_V1 = "orb-d3-xs-v1"
    ORB_V1_MPTRAJ_ONLY = "orb-v1-mptraj-only"

    @classmethod
    def get_function_mapping(cls):
        """Get mapping from model names to pretrained function names."""
        return {
            cls.ORB_V3_CONSERVATIVE_INF_OMAT: "orb_v3_conservative_inf_omat",
            cls.ORB_V3_CONSERVATIVE_20_OMAT: "orb_v3_conservative_20_omat",
            cls.ORB_V3_CONSERVATIVE_INF_MPA: "orb_v3_conservative_inf_mpa",
            cls.ORB_V3_CONSERVATIVE_20_MPA: "orb_v3_conservative_20_mpa",
            cls.ORB_V3_DIRECT_INF_OMAT: "orb_v3_direct_inf_omat",
            cls.ORB_V3_DIRECT_20_OMAT: "orb_v3_direct_20_omat",
            cls.ORB_V3_DIRECT_INF_MPA: "orb_v3_direct_inf_mpa",
            cls.ORB_V3_DIRECT_20_MPA: "orb_v3_direct_20_mpa",
            cls.ORB_V2: "orb_v2",
            cls.ORB_D3_V2: "orb_d3_v2",
            cls.ORB_D3_SM_V2: "orb_d3_sm_v2",
            cls.ORB_D3_XS_V2: "orb_d3_xs_v2",
            cls.ORB_MPTRAJ_ONLY_V2: "orb_mptraj_only_v2",
            cls.ORB_V1: "orb_v1",
            cls.ORB_D3_V1: "orb_d3_v1",
            cls.ORB_D3_SM_V1: "orb_d3_sm_v1",
            cls.ORB_D3_XS_V1: "orb_d3_xs_v1",
            cls.ORB_V1_MPTRAJ_ONLY: "orb_v1_mptraj_only",
        }

    @property
    def function_name(self) -> str:
        """Get the corresponding pretrained function name."""
        mapping = self.get_function_mapping()
        if self in mapping:
            return mapping[self]
        # Fallback: replace hyphens with underscores
        return self.value.replace("-", "_")


class DeviceType(str, Enum):
    """Available device types for computation."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ORBOracle(ASEPotentialOracle):
    """ORB-based atomic simulator - only handles ORB calculator setup."""

    def __init__(
        self,
        model_name: ORBModelName | str,
        device: DeviceType | str = DeviceType.AUTO,
        element_reference_energies_path: str | None = None,
        relax: bool = True,
        relax_kwargs: dict[str, Any] | None = None,
        num_workers: int = 1,
    ):
        """Initialize ORB atomic simulator.

        Args:
            model_name: ORB model to use
            device: Device to run on
            element_reference_energies: Optional map of elemental symbol to reference energy per atom (e.g. from materials project)
            relax: Whether to relax the structure using ASE.
            relax_kwargs: Keyword arguments to pass to the ASE structure optimizer.
            num_workers: Number of worker threads for parallel evaluation (default: 1)
        """

        # Allow passing strings via Hydra config and convert to enums
        self.model_name = (
            ORBModelName(model_name) if isinstance(model_name, str) else model_name
        )
        self.device = DeviceType(device) if isinstance(device, str) else device
        if device == DeviceType.AUTO:
            self.device = (
                DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
            )

        # Store model configuration for factory function
        func_name = self.model_name.function_name
        device_value = self.device.value

        # Create a factory function that creates a new calculator instance
        # This ensures thread-safety when using multiple workers
        def create_orb_calculator():
            from orb_models.forcefield import pretrained

            model_func = getattr(pretrained, func_name)
            model = model_func(device=device_value, compile=False)
            return ORBCalculator(model, device=device_value)

        # Initialize parent class with calculator factory
        super().__init__(
            calculator=create_orb_calculator,
            element_reference_energies_path=element_reference_energies_path,
            relax=relax,
            relax_kwargs=relax_kwargs or {},
            num_workers=num_workers,
        )
