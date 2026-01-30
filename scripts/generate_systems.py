"""
Generate lists of chemical systems (binary, ternary, quaternary) from Materials Project.

Outputs JSON files containing lists of element lists, e.g., [[Co, Nb], [Li, O], ...].
Filters systems to those with at least a minimum number of stable structures (per size).
To reproduce 
"""

import itertools
import json
import logging
import os
import random
from pathlib import Path

import dotenv
import fire
from ase.data import atomic_numbers, chemical_symbols
from mp_api.client import MPRester

from made.evaluation.metrics import is_smact_valid

logger = logging.getLogger(__name__)

# Default excluded elements: radioactive elements and noble gases
DEFAULT_EXCLUDED_ELEMENTS = {
    # Radioactive elements (all isotopes are radioactive)
    "Tc",  # Technetium (43)
    "Pm",  # Promethium (61)
    "Po",  # Polonium (84)
    # Noble gas elements
    "He",  # Helium
    "Ne",  # Neon
    "Ar",  # Argon
    "Kr",  # Krypton
    "Xe",  # Xenon
    "Rn",  # Radon
    # problematic elements
    "Yb",
}

# Metallic elements (including alkali, alkaline earth, transition, post-transition, lanthanides, actinides)
METAL_ELEMENTS = {
    # Alkali metals
    "Li", "Na", "K", "Rb", "Cs", "Fr",
    # Alkaline earth metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
    # Transition metals
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs",
    # Post-transition metals
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
    # Lanthanides
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # Actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}


def get_all_elements_from_ase(
    max_atomic_number: int = 84,
    excluded_elements: set[str] | None = None,
) -> list[str]:
    """Return list of element symbols from ASE periodic table.

    Filters out:
    - The placeholder 'X' and any non-standard entries
    - Elements with atomic numbers > max_atomic_number
    - Excluded elements (default: radioactive elements and noble gases)

    Args:
        max_atomic_number: Maximum atomic number to include (default: 84)
        excluded_elements: Set of element symbols to exclude.
            If None, uses DEFAULT_EXCLUDED_ELEMENTS (Tc, Pm, Po, He, Ne, Ar, Kr, Xe, Rn).

    Returns:
        List of filtered element symbols.
    """
    if excluded_elements is None:
        excluded_elements = DEFAULT_EXCLUDED_ELEMENTS

    elems = [sym for sym in chemical_symbols if sym and sym != "X"]
    # Filter to valid element symbols
    valid_elems = [e for e in elems if e.isalpha() and e[0].isupper()]

    # Filter out elements with atomic numbers > max_atomic_number and excluded elements
    filtered = []
    for elem in valid_elems:
        if elem in atomic_numbers:
            z = atomic_numbers[elem]
            if z <= max_atomic_number and elem not in excluded_elements:
                filtered.append(elem)

    return filtered


def is_binary_metal_oxide(system: list[str]) -> bool:
    """Check if a binary system is a metal oxide (one metal + oxygen).
    
    Args:
        system: List of element symbols
        
    Returns:
        True if the system is a binary metal oxide, False otherwise
    """
    if len(system) != 2:
        return False
    
    # Check if one element is O and the other is a metal
    has_oxygen = "O" in system
    if not has_oxygen:
        return False
    
    other_elem = [e for e in system if e != "O"][0]
    return other_elem in METAL_ELEMENTS


def is_intermetallic(system: list[str]) -> bool:
    """Check if a system is intermetallic (all elements are metals).
    
    Args:
        system: List of element symbols
        
    Returns:
        True if all elements are metals, False otherwise
    """
    return all(elem in METAL_ELEMENTS for elem in system)


def generate_systems(
    elements: list[str],
    size: int,
    shuffle: bool = True,
    seed: int = 123,
    only_binary_metal_oxides: bool = False,
    only_intermetallics: bool = False,
) -> list[list[str]]:
    """Generate chemical systems of a given size.
    
    Args:
        elements: List of element symbols to use
        size: Size of the chemical system (2 for binary, 3 for ternary, etc.)
        shuffle: Whether to shuffle the systems
        seed: Random seed for shuffling (ensures reproducibility)
        only_binary_metal_oxides: If True and size==2, only include metal oxide systems
        only_intermetallics: If True, only include intermetallic systems (all metals)
        
    Returns:
        List of chemical systems (each system is a list of element symbols)
    """
    systems = [list(comb) for comb in itertools.combinations(sorted(elements), size)]
    
    # Apply filters if requested
    if only_binary_metal_oxides and size == 2:
        systems = [sys for sys in systems if is_binary_metal_oxide(sys)]
    
    if only_intermetallics:
        systems = [sys for sys in systems if is_intermetallic(sys)]
    
    # Shuffle using a seeded random generator for reproducibility
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(systems)

    return systems


def count_stable_entries_for_system(
    elements: list[str],
    mpr: MPRester,
    thermo_types: list[str] | None = None,
    max_atoms_per_unit_cell: int | None = None,
    energy_above_hull_threshold: float | None = None,
    filter_by_smact: bool = False,
) -> int:
    """Return number of stable entries for the given chemical system according to MP Summary API.

    Uses mpr.summary.search() with num_sites and energy_above_hull filtering for efficient API-level filtering when possible.
    Falls back to get_entries_in_chemsys() when thermo_types filtering is needed.
    Reference: https://github.com/hspark1212/chemeleon/blob/main/data/mp-40/data_preparation.ipynb

    Args:
        elements: List of element symbols in the chemical system
        mpr: MPRester instance
        thermo_types: List of thermo types to filter by. If specified, uses get_entries_in_chemsys()
            instead of summary.search() since thermo_types filtering is not directly supported.
        max_atoms_per_unit_cell: Maximum number of atoms per unit cell to include.
            If None, no filtering by atom count is applied.
        energy_above_hull_threshold: Maximum energy above hull (eV/atom) to include.
            If None, no filtering by energy above hull is applied.
        filter_by_smact: If True, filter entries by SMACT validity.

    Returns:
        Number of stable entries matching the criteria.
    """
    try:
        # If thermo_types is specified, use get_entries_in_chemsys() which supports it
        # Note: entries with is_stable=True have e_above_hull <= 0, so energy_above_hull_threshold
        # filtering is not directly supported with thermo_types filtering
        if thermo_types is not None:
            entries = mpr.get_entries_in_chemsys(
                elements=elements,
                additional_criteria={"thermo_types": thermo_types, "is_stable": True},
            )

            filtered_entries = []
            for entry in entries:
                # Filter by number of atoms in unit cell if specified
                if max_atoms_per_unit_cell is not None:
                    if hasattr(entry, "structure") and entry.structure is not None:
                        num_atoms = len(entry.structure)
                        if num_atoms > max_atoms_per_unit_cell:
                            continue
                    else:
                        continue

                # Filter by SMACT validity if specified
                if filter_by_smact:
                    if hasattr(entry, "structure") and entry.structure is not None:
                        if not is_smact_valid(entry.structure):
                            continue
                    else:
                        continue

                filtered_entries.append(entry)

            return len(filtered_entries)

        # Otherwise, use summary.search() for efficient API-level filtering
        search_criteria = {
            "elements": elements,
        }

        # If energy_above_hull_threshold is specified, use it to filter
        # Otherwise, use is_stable=True to get only stable entries
        if energy_above_hull_threshold is not None:
            search_criteria["energy_above_hull"] = [0, energy_above_hull_threshold]
        else:
            search_criteria["is_stable"] = True

        # Add num_sites filter if specified (more efficient than filtering in Python)
        if max_atoms_per_unit_cell is not None:
            search_criteria["num_sites"] = [0, max_atoms_per_unit_cell]

        # Use summary.search() for efficient API-level filtering
        # If SMACT filtering is needed, we need structure data, so fetch it
        fields = ["material_id", "structure"] if filter_by_smact else ["material_id"]
        docs = mpr.summary.search(
            **search_criteria,
            fields=fields,
        )

        # If SMACT filtering is needed, check each document
        if filter_by_smact:
            count = 0
            for doc in docs:
                if hasattr(doc, "structure") and doc.structure is not None:
                    if is_smact_valid(doc.structure):
                        count += 1
            return count
        else:
            # Count the results (docs is a generator/iterator)
            count = sum(1 for _ in docs)
            return count
    except Exception as e:
        logger.warning(f"Error counting entries for {elements}: {e}")
        return 0


def filter_systems_by_min_stable(
    systems: list[list[str]],
    min_stable: int,
    mpr: MPRester,
    thermo_types: list[str] | None = None,
    max_entries: int | None = None,
    max_atoms_per_unit_cell: int | None = None,
    energy_above_hull_threshold: float | None = None,
    filter_by_smact: bool = False,
) -> list[list[str]]:
    kept: list[list[str]] = []
    for sys in systems:
        cnt = count_stable_entries_for_system(
            sys,
            mpr,
            thermo_types=thermo_types,
            max_atoms_per_unit_cell=max_atoms_per_unit_cell,
            energy_above_hull_threshold=energy_above_hull_threshold,
            filter_by_smact=filter_by_smact,
        )
        logger.info(f"Found {cnt} stable entries for {sys}")
        if cnt >= int(min_stable):
            kept.append(sys)
        if max_entries is not None and len(kept) >= int(max_entries):
            break
    return kept


def main(
    output_dir: str = "./data/systems",
    elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    min_stable: int = 2,
    thermo_types: list[str] | None = None,
    max_systems_per_size: int = 10,
    seed: int = 123,
    max_atomic_number: int = 84,
    excluded_elements: list[str] | None = None,
    max_atoms_per_unit_cell: int = 20,
    filter_by_smact: bool = False,
    energy_above_hull_threshold: float = 0.1,
    only_binary_metal_oxides: bool = False,
    only_intermetallics: bool = False,
    system_sizes: list[int] | None = None,
    log_level: str = "INFO",
) -> None:
    """Generate chemical systems with stability filtering from Materials Project.
    
    Args:
        output_dir: Directory to save output JSON files (default: ./data/systems)
        elements: List of element symbols to include (default: all up to max_atomic_number)
        exclude_elements: List of element symbols to exclude
        min_stable: Minimum number of stable structures required for each system (default: 4)
        thermo_types: MP thermo_types filter (default: ['GGA_GGA+U'])
        max_systems_per_size: Cap on number of systems per size (default: 20)
        seed: Random seed for shuffling systems (default: 123)
        max_atomic_number: Maximum atomic number to include (default: 84)
        excluded_elements: Element symbols to exclude (default: Tc, Pm, Po, He, Ne, Ar, Kr, Xe, Rn)
        max_atoms_per_unit_cell: Maximum atoms per unit cell (default: 20)
        filter_by_smact: Filter entries by SMACT validity (default: False)
        energy_above_hull_threshold: Maximum energy above hull in eV/atom (default: 0.1)
        only_binary_metal_oxides: Only generate binary metal oxide systems (default: False)
        only_intermetallics: Only generate intermetallic systems (default: False)
        system_sizes: Sizes of systems to generate, e.g., [2, 3, 4] (default: [2, 3, 4])
        log_level: Logging level (default: INFO)
    """
    # Setup logging
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    
    dotenv.load_dotenv()
    api_key = os.environ.get("MATERIALS_PROJECT_API_KEY")
    if not api_key:
        raise RuntimeError("Set MATERIALS_PROJECT_API_KEY env var")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    
    # Convert excluded_elements list to set if provided
    excluded_elements_set = set(excluded_elements) if excluded_elements else None
    
    # Default thermo_types
    if thermo_types is None:
        thermo_types = ["GGA_GGA+U"]
    
    # Default to generating binaries, ternaries, and quaternaries if not specified
    if system_sizes is None:
        system_sizes = [2, 3, 4]
    
    # Mapping from size to name for output files
    size_names = {
        1: "unary",
        2: "binary",
        3: "ternary",
        4: "quaternary",
        5: "quinary",
        6: "senary",
        7: "septenary",
        8: "octonary",
    }

    with MPRester(api_key) as mpr:
        all_elems = (
            get_all_elements_from_ase(
                max_atomic_number=max_atomic_number,
                excluded_elements=excluded_elements_set,
            )
            if elements is None
            else elements
        )
        if exclude_elements is not None:
            all_elems = [e for e in all_elems if e not in exclude_elements]

        # Generate and process systems for each requested size
        for size in system_sizes:
            # Get the name for this size
            size_name = size_names.get(size, f"{size}-component")
            
            # Use a deterministic seed for each size (combine main seed with size)
            # This ensures reproducibility while giving different shuffles per size
            size_seed = int(seed) + size
            
            # Generate systems of this size
            systems = generate_systems(
                all_elems, 
                size, 
                shuffle=True, 
                seed=size_seed,
                only_binary_metal_oxides=only_binary_metal_oxides if size == 2 else False,
                only_intermetallics=only_intermetallics,
            )
            logger.info(f"Generated {len(systems)} {size_name} systems")

            # Keep only systems with sufficient number of stable structures
            filtered_systems = filter_systems_by_min_stable(
                systems,
                min_stable,
                mpr,
                thermo_types=thermo_types,
                max_entries=max_systems_per_size,
                max_atoms_per_unit_cell=max_atoms_per_unit_cell,
                energy_above_hull_threshold=energy_above_hull_threshold,
                filter_by_smact=filter_by_smact,
            )
            
            # Build filename with metadata
            filename_parts = [f"systems_{size_name}"]
            
            # Add number of systems
            filename_parts.append(f"n{len(filtered_systems)}")
            
            # Add max atoms per unit cell if specified
            if max_atoms_per_unit_cell is not None:
                filename_parts.append(f"maxatoms{max_atoms_per_unit_cell}")
            
            # Add filter flags
            if only_binary_metal_oxides and size == 2:
                filename_parts.append("metaloxides")
            if only_intermetallics:
                filename_parts.append("intermetallic")
            if filter_by_smact:
                filename_parts.append("smact")
            
            filename = "_".join(filename_parts) + ".json"
            output_file = output_path / filename
            
            # Write to file
            with open(output_file, "w") as f:
                json.dump(filtered_systems, f, indent=2)
            logger.info(f"Wrote {len(filtered_systems)} {size_name} systems to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
