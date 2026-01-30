"""
Structure hashing utility for caching.

Creates a hash key from composition + simplified structure representation.
Simplified representation: primitive cell, normalized (volume-normalized).
"""

import hashlib

from pymatgen.core.structure import Structure


def structure_hash(structure: Structure) -> str:
    """
    Create a hash key from composition + simplified structure representation.

    The simplified representation:
    - Gets primitive structure
    - Normalizes to a canonical form for hashing

    Args:
        structure: The structure to hash

    Returns:
        A string hash key
    """
    # Create a copy to avoid modifying the original
    simplified = structure.copy()

    # Get primitive structure
    simplified = simplified.get_primitive_structure()

    # Get reduced structure (Niggli reduction) for canonical form
    simplified = simplified.get_reduced_structure(reduction_algo="LLL")

    # Create hash components
    # 1. Composition (reduced formula)
    composition_str = simplified.composition.reduced_formula

    # 2. Structure features: lattice parameters, species, fractional coordinates
    lattice_params = simplified.lattice.abc + simplified.lattice.angles
    lattice_str = ",".join(f"{v:.6f}" for v in lattice_params)

    # Species as atomic numbers (sorted by fractional coordinates)
    species = [s.Z for s in simplified.species]
    species_str = ",".join(str(z) for z in species)

    # Fractional coordinates (rounded to reasonable precision)
    frac_coords = simplified.frac_coords
    # Sort by species then coordinates for consistency (if same composition)
    # Round to 6 decimal places for hashing
    frac_coords_str = ",".join(f"{c:.6f}" for coord in frac_coords for c in coord)

    # Combine all components
    hash_input = f"{composition_str}|{lattice_str}|{species_str}|{frac_coords_str}"

    # Create SHA256 hash
    hash_obj = hashlib.sha256(hash_input.encode("utf-8"))
    return hash_obj.hexdigest()
