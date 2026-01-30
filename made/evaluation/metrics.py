"""Evaluation utilities for materials discovery.

This module provides modular metric calculators inspired by Microsoft's
`mattergen` repository. Metrics are grouped into logical categories
(validity, diversity, novelty, stability, convex hull) and exposed via both
individual helper functions and a high-level `MaterialsMetrics` façade.

Each calculator exposes a static `calculate` method that accepts the minimal
data required for the metric family, making it easy to reuse components in
custom workflows while still offering a batteries-included pipeline.
"""

import itertools
import logging
from collections import Counter
from typing import Any

import amd
import numpy as np
import smact
from amd.io import periodicset_from_pymatgen_structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.distance import pdist as scipy_pdist
from smact.screening import pauling_test

from made.utils.convex_hull_utils import safe_e_above_hull

# -----------------------------------------------------------------------------
# Validity metrics (chemical sanity checks)
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# Ref: https://github.com/microsoft/mattergen/blob/main/mattergen/evaluation/metrics/structure.py
def is_smact_valid(structure_or_composition: Structure | Composition) -> bool:
    """
    Returns True if the structure or composition is valid according to the
    smact validity checker else False.

    Args:
        structure_or_composition: pymatgen Structure or Composition object to check

    Returns:
        True if the structure/composition is SMACT-valid, False otherwise
    """
    # Extract atomic_numbers from either Structure or Composition
    if isinstance(structure_or_composition, Structure):
        atomic_numbers = structure_or_composition.atomic_numbers
    elif isinstance(structure_or_composition, Composition):
        # Use reduced composition to normalize stoichiometry
        reduced_comp = structure_or_composition.reduced_composition
        # Build atomic_numbers list from composition (mirroring structure.atomic_numbers)
        atomic_numbers = []
        for element, amount in reduced_comp.items():
            element_z = Element(element).Z
            atomic_numbers.extend([element_z] * int(round(amount)))
    else:
        raise TypeError(
            f"Expected Structure or Composition, got {type(structure_or_composition)}"
        )

    # Common validation logic (same for both Structure and Composition)
    elem_counter = Counter(atomic_numbers)
    composition_list = [
        (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
    ]
    elems, counts = list(zip(*composition_list, strict=True))
    counts = np.array(counts)
    counts = counts / np.gcd.reduce(counts)
    comps: tuple[int, ...] = tuple(np.array(counts).astype("int"))
    return smact_validity(
        comp=elems, count=comps, use_pauling_test=True, include_alloys=True
    )


# Ref: https://github.com/microsoft/mattergen/blob/main/mattergen/evaluation/metrics/structure.py
def smact_validity(
    comp: tuple[int, ...] | tuple[str, ...],
    count: tuple[int, ...],
    use_pauling_test: bool = True,
    include_alloys: bool = True,
    include_cutoff: bool = True,
    use_element_symbol: bool = False,
) -> bool:
    """Computes SMACT validity.

    Args:
        comp: Tuple of atomic number or element names of elements in a crystal.
        count: Tuple of counts of elements in a crystal.
        use_pauling_test: Whether to use electronegativity test. That is, at least in one
            combination of oxidation states, the more positive the oxidation state of a site,
            the lower the electronegativity of the element for all pairs of sites.
        include_alloys: if True, returns True without checking charge balance or electronegativity
            if the crystal is an alloy (consisting only of metals) (default: True).
        include_cutoff: assumes valid crystal if the combination of oxidation states is more
            than 10^6 (default: True).

    Returns:
        True if the crystal is valid, False otherwise.
    """
    assert len(comp) == len(count)
    if use_element_symbol:
        elem_symbols = comp
    else:
        elem_symbols = tuple([str(Element.from_Z(Z=elem)) for elem in comp])  # type:ignore
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    n_comb = np.prod([len(ls) for ls in ox_combos])
    # If the number of possible combinations is big, it'd take too much time to run the smact checker
    # In this case, we assume that at least one of the combinations is valid
    if n_comb > 1e6 and include_cutoff:
        return True
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_0k = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_0k = True
            else:
                electroneg_0k = True
            if electroneg_0k:
                for ratio in cn_r:
                    compositions.append((elem_symbols, ox_states, ratio))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    return len(compositions) > 0


def is_min_distance_valid(
    structure: Structure,
    min_distance_threshold: float = 0.5,
    return_distance: bool = False,
) -> bool | tuple[bool, float]:
    """
    Returns True if the minimum interatomic distance in the structure is greater than the threshold.
    """
    distance = get_min_interatomic_distance(structure)
    if return_distance:
        return distance >= min_distance_threshold, distance
    else:
        return distance >= min_distance_threshold


# Ref: https://github.com/microsoft/mattergen/blob/main/mattergen/evaluation/metrics/structure.py
def get_min_interatomic_distance(structure: Structure) -> float:
    """
    Calculate the minimum interatomic distance in a structure.

    Args:
        structure: The crystal structure to analyze.

    Returns:
        The minimum distance between any pair of atoms in Angstroms.
    """
    all_distances = structure.distance_matrix
    # Set diagonal elements (self-distances) to infinity
    np.fill_diagonal(all_distances, np.inf)
    return float(np.min(all_distances))


class ValidityMetrics:
    """Distance and SMACT-based validity checks."""

    @classmethod
    def distance_validity(
        cls,
        structures: list[Structure],
        *,
        min_distance_threshold: float = 0.5,
    ) -> dict[str, float]:
        """
        Compute distance-based validity metrics.

        Args:
            structures: List of crystal structures to analyze.
            min_distance_threshold: Minimum interatomic distance threshold in Angstroms.

        Returns:
            Dictionary with distance validity counts, fractions, and statistics.
        """
        if not structures:
            return {
                "distance_valid_count": 0,
                "distance_valid_fraction": 0.0,
                "min_interatomic_distance_min": 0.0,
                "min_interatomic_distance_mean": 0.0,
                "min_interatomic_distance_std": 0.0,
            }

        output = [
            is_min_distance_valid(
                structure, min_distance_threshold, return_distance=True
            )
            for structure in structures
        ]
        distance_valid_flags = np.array([flag for flag, _ in output])
        distances = np.array([distance for _, distance in output])

        return {
            "distance_valid_count": float(np.sum(distance_valid_flags)),
            "distance_valid_fraction": float(np.mean(distance_valid_flags)),
            "min_interatomic_distance_min": float(np.min(distances)),
            "min_interatomic_distance_mean": float(np.mean(distances)),
            "min_interatomic_distance_std": float(np.std(distances)),
        }

    @classmethod
    def smact_validity(cls, structures: list[Structure]) -> dict[str, float]:
        """
        Compute SMACT-based validity metrics.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary with SMACT validity counts and fractions.
        """
        if not structures:
            return {
                "smact_valid_count": 0.0,
                "smact_valid_fraction": 0.0,
            }

        smact_flags = [is_smact_valid(structure) for structure in structures]
        total = len(structures)

        return {
            "smact_valid_count": float(np.sum(smact_flags)),
            "smact_valid_fraction": float(np.sum(smact_flags)) / total,
        }

    @classmethod
    def calculate(
        cls,
        structures: list[Structure] | None,
        *,
        min_distance_threshold: float = 0.5,
    ) -> dict[str, float]:
        """
        Calculate all validity metrics including distance and SMACT checks.

        Args:
            structures: List of crystal structures to analyze.
            min_distance_threshold: Minimum interatomic distance threshold in Angstroms.

        Returns:
            Dictionary containing all validity metrics.
        """
        if structures is None:
            structures = []

        total = len(structures)
        if total == 0:
            return {
                "distance_valid_count": 0,
                "distance_valid_fraction": 0.0,
                "min_interatomic_distance_min": 0.0,
                "min_interatomic_distance_mean": 0.0,
                "min_interatomic_distance_std": 0.0,
                "smact_valid_count": 0.0,
                "smact_valid_fraction": 0.0,
                "distance_and_smact_valid_count": 0.0,
                "distance_and_smact_valid_fraction": 0.0,
            }

        distance_metrics = cls.distance_validity(
            structures, min_distance_threshold=min_distance_threshold
        )
        smact_metrics = cls.smact_validity(structures)

        # Combine distance and SMACT validity
        distance_valid_flags = np.array(
            [is_min_distance_valid(s, min_distance_threshold) for s in structures]
        )
        smact_flags = np.array([is_smact_valid(s) for s in structures])
        combined_valid = distance_valid_flags & smact_flags

        results = {**distance_metrics, **smact_metrics}
        results.update(
            {
                "distance_and_smact_valid_count": float(np.sum(combined_valid)),
                "distance_and_smact_valid_fraction": float(np.sum(combined_valid))
                / total,
            }
        )

        return results


class CompositionDiversityMetrics:
    """Metrics for compositional diversity in a set of structures."""

    @staticmethod
    def _build_composition_matrix(
        compositions: list[Composition],
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Build a composition matrix where each row is a composition vector.

        Args:
            compositions: List of Composition objects.

        Returns:
            Tuple of (composition_matrix, element_index_map).
        """
        all_elements = sorted(
            {el.symbol for comp in compositions for el in comp.elements}
        )

        if not all_elements:
            return np.zeros((len(compositions), 0), dtype=float), {}

        comp_matrix = np.zeros((len(compositions), len(all_elements)), dtype=float)
        elem_index = {elem: i for i, elem in enumerate(all_elements)}

        for row_idx, comp in enumerate(compositions):
            for el, amt in comp.get_el_amt_dict().items():
                comp_matrix[row_idx, elem_index[el]] = float(amt)

        return comp_matrix, elem_index

    @classmethod
    def unique_compositions(cls, structures: list[Structure]) -> dict[str, float]:
        """
        Compute unique composition metrics.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary with unique composition counts and fractions.
        """
        if not structures:
            return {
                "unique_composition_count": 0.0,
                "unique_composition_fraction": 0.0,
            }

        compositions = [
            struct.composition.fractional_composition for struct in structures
        ]
        unique_compositions = {comp.reduced_formula for comp in compositions}
        stoich_diversity = len(unique_compositions) / len(structures)

        return {
            "unique_composition_count": float(len(unique_compositions)),
            "unique_composition_fraction": float(stoich_diversity),
        }

    @classmethod
    def _pairwise_distance_metrics(
        cls,
        structures: list[Structure],
        metric: str,
        prefix: str,
    ) -> dict[str, float]:
        """
        Compute pairwise distance metrics between compositions.

        Args:
            structures: List of crystal structures to analyze.
            metric: Distance metric to use (e.g., "cityblock", "euclidean").
            prefix: Prefix for the metric keys (e.g., "l1_distance", "l2_distance").

        Returns:
            Dictionary with distance statistics (min, mean, max, std).
        """
        if not structures:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_std": 0.0,
            }

        compositions = [
            struct.composition.fractional_composition for struct in structures
        ]
        comp_matrix, _ = cls._build_composition_matrix(compositions)

        if len(compositions) < 2 or comp_matrix.shape[1] == 0:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_std": 0.0,
            }

        dists_array = scipy_pdist(comp_matrix, metric=metric)
        if dists_array.size == 0:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_std": 0.0,
            }

        return {
            f"{prefix}_min": float(np.min(dists_array)),
            f"{prefix}_mean": float(np.mean(dists_array)),
            f"{prefix}_max": float(np.max(dists_array)),
            f"{prefix}_std": float(np.std(dists_array)),
        }

    @classmethod
    def l1_distances(cls, structures: list[Structure]) -> dict[str, float]:
        """
        Compute L1 (Manhattan) distance metrics between compositions.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary with L1 distance statistics (min, mean, max, std).
        """
        return cls._pairwise_distance_metrics(structures, "cityblock", "l1_distance")

    @classmethod
    def l2_distances(cls, structures: list[Structure]) -> dict[str, float]:
        """
        Compute L2 (Euclidean) distance metrics between compositions.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary with L2 distance statistics (min, mean, max, std).
        """
        return cls._pairwise_distance_metrics(structures, "euclidean", "l2_distance")

    @classmethod
    def calculate(cls, structures: list[Structure]) -> dict[str, float]:
        """
        Calculate all composition diversity metrics.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary containing all composition diversity metrics.
        """
        unique_metrics = cls.unique_compositions(structures)
        l1_metrics = cls.l1_distances(structures)
        l2_metrics = cls.l2_distances(structures)

        return {**unique_metrics, **l1_metrics, **l2_metrics}


class StructureDiversityMetrics:
    """Metrics for structural diversity in a set of structures."""

    @classmethod
    def unique_structures(
        cls,
        structures: list[Structure],
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> dict[str, float]:
        """
        Compute unique structure metrics using structure matching.

        Args:
            structures: List of crystal structures to analyze.
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance for structure matching in degrees.

        Returns:
            Dictionary with unique structure counts and fractions.
        """
        if not structures:
            return {
                "unique_structure_count": 0.0,
                "unique_structure_fraction": 0.0,
            }

        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        unique_structures = []
        for struct in structures:
            is_unique = True
            for unique in unique_structures:
                try:
                    if matcher.fit(struct, unique):
                        is_unique = False
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {struct.composition} and {unique.composition}: {e}"
                    )
                    # Treat as non-matching on error, continue checking
            if is_unique:
                unique_structures.append(struct)

        return {
            "unique_structure_count": float(len(unique_structures)),
            "unique_structure_fraction": float(
                len(unique_structures) / len(structures)
            ),
        }

    @classmethod
    def spacegroup_diversity(cls, structures: list[Structure]) -> dict[str, float]:
        """
        Compute spacegroup diversity metrics.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary with spacegroup entropy, counts, and fractions.
        """
        if not structures:
            return {
                "spacegroup_entropy": 0.0,
                "unique_spacegroups_count": 0.0,
                "unique_spacegroups_fraction": 0.0,
            }

        spacegroups = []
        for struct in structures:
            try:
                sg_num = SpacegroupAnalyzer(struct).get_space_group_number()
                spacegroups.append(sg_num)
            except Exception:
                # Skip structures where spacegroup analysis fails
                # (e.g., invalid structures, symmetry issues)
                logger.warning(f"Failed to analyze spacegroup for {struct}")
                continue

        # If all structures failed spacegroup analysis, return default values
        if not spacegroups:
            return {
                "spacegroup_entropy": 0.0,
                "unique_spacegroups_count": 0.0,
                "unique_spacegroups_fraction": 0.0,
            }

        spacegroup_counts = {}
        for sg in spacegroups:
            spacegroup_counts[sg] = spacegroup_counts.get(sg, 0) + 1

        # Calculate entropy of spacegroup distribution
        sg_probabilities = (
            np.array([count / len(spacegroups) for count in spacegroup_counts.values()])
            if spacegroups
            else np.array([])
        )
        spacegroup_entropy = (
            -np.sum(sg_probabilities * np.log(sg_probabilities))
            if len(sg_probabilities) > 0
            else 0.0
        )

        return {
            "spacegroup_entropy": float(spacegroup_entropy),
            "unique_spacegroups_count": float(len(set(spacegroups))),
            "unique_spacegroups_fraction": float(
                len(set(spacegroups)) / len(spacegroups)
            )
            if spacegroups
            else 0.0,
        }

    @classmethod
    def lattice_parameter_variance(
        cls, structures: list[Structure]
    ) -> dict[str, float]:
        """
        Compute lattice parameter variance metrics.

        Args:
            structures: List of crystal structures to analyze.

        Returns:
            Dictionary with lattice parameter variance.
        """
        if not structures:
            return {"lattice_parameter_variance": 0.0}

        lattice_params = np.array([list(struct.lattice.abc) for struct in structures])
        lattice_param_variance = (
            float(np.mean(np.var(lattice_params, axis=0)))
            if len(lattice_params) > 0
            else 0.0
        )

        return {"lattice_parameter_variance": float(lattice_param_variance)}

    @classmethod
    def amd_distances(
        cls, structures: list[Structure], k: int = 100
    ) -> dict[str, float]:
        """
        Compute AMD (Average Minimum Distance) metrics.

        Args:
            structures: List of crystal structures to analyze.
            k: Number of nearest neighbors for AMD calculation.

        Returns:
            Dictionary with AMD distance statistics (min, mean, max).
        """
        if not structures or len(structures) < 2:
            return {
                "amd_distance_min": 0.0,
                "amd_distance_mean": 0.0,
                "amd_distance_max": 0.0,
            }

        crystals = [
            periodicset_from_pymatgen_structure(struct) for struct in structures
        ]
        amds = [amd.AMD(cr, k) for cr in crystals]
        cdm = amd.AMD_pdist(amds)

        if hasattr(cdm, "__len__") and len(cdm) > 0:
            return {
                "amd_distance_min": float(np.min(cdm)),
                "amd_distance_mean": float(np.mean(cdm)),
                "amd_distance_max": float(np.max(cdm)),
            }
        else:
            return {
                "amd_distance_min": 0.0,
                "amd_distance_mean": 0.0,
                "amd_distance_max": 0.0,
            }

    @classmethod
    def calculate(
        cls,
        structures: list[Structure],
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> dict[str, float]:
        """
        Calculate all structural diversity metrics.

        Args:
            structures: List of crystal structures to analyze.
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance for structure matching in degrees.

        Returns:
            Dictionary containing all structural diversity metrics.
        """
        unique_metrics = cls.unique_structures(structures, ltol, stol, angle_tol)
        spacegroup_metrics = cls.spacegroup_diversity(structures)
        lattice_metrics = cls.lattice_parameter_variance(structures)
        amd_metrics = cls.amd_distances(structures)

        return {
            **unique_metrics,
            **spacegroup_metrics,
            **lattice_metrics,
            **amd_metrics,
        }


class NoveltyMetrics:
    """Metrics for novelty/uniqueness and stability filtered counts/fractions."""

    @classmethod
    def uniqueness(
        cls,
        candidate_structures: list[Structure],
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Compute uniqueness among candidate structures.

        Args:
            candidate_structures: List of candidate structures to analyze.
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance for structure matching in degrees.

        Returns:
            Tuple of (is_unique_array, metrics_dict) where is_unique is a boolean array.
        """
        total = len(candidate_structures)
        if total == 0:
            return np.array([], dtype=bool), {
                "unique_structure_count": 0.0,
                "unique_structure_fraction": 0.0,
            }

        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        unique_indices: list[int] = []
        for i, s in enumerate(candidate_structures):
            is_unique = True
            for j in unique_indices:
                try:
                    if matcher.fit(s, candidate_structures[j]):
                        is_unique = False
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {s.composition} and {candidate_structures[j].composition}: {e}"
                    )
                    # Treat as non-matching on error, continue checking
            if is_unique:
                unique_indices.append(i)

        is_unique = np.zeros(total, dtype=bool)
        is_unique[unique_indices] = True

        return is_unique, {
            "unique_structure_count": float(is_unique.sum()),
            "unique_structure_fraction": float(is_unique.sum() / total),
        }

    @classmethod
    def novelty(
        cls,
        candidate_structures: list[Structure],
        reference_structures: list[Structure],
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Compute novelty vs reference structures.

        Args:
            candidate_structures: List of candidate structures to analyze.
            reference_structures: List of reference structures to compare against.
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance for structure matching in degrees.

        Returns:
            Tuple of (is_novel_array, metrics_dict) where is_novel is a boolean array.
        """
        total = len(candidate_structures)
        if total == 0:
            return np.array([], dtype=bool), {
                "novel_structure_count": 0.0,
                "novel_structure_fraction": 0.0,
            }

        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        novel_flags: list[bool] = []
        for s in candidate_structures:
            is_novel = True
            for ref in reference_structures:
                try:
                    if matcher.fit(s, ref):
                        is_novel = False
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {s.composition} and {ref.composition}: {e}"
                    )
                    # Treat as non-matching on error, continue checking
            novel_flags.append(is_novel)

        is_novel = np.array(novel_flags, dtype=bool)

        return is_novel, {
            "novel_structure_count": float(is_novel.sum()),
            "novel_structure_fraction": float(is_novel.sum() / total),
        }

    @classmethod
    def calculate(
        cls,
        candidate_structures: list[Structure],
        reference_structures: list[Structure],
        stable_flags: list[bool] | None = None,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> dict[str, float]:
        """
        Compute counts and fractions for stable/unique/novel proposals.

        - unique: distinct among all proposed structures (by StructureMatcher)
        - novel: does not match any reference structure (e.g., initial structures)
        - stable: provided via stable_flags for each candidate
        """
        total = len(candidate_structures)
        if total == 0:
            return {
                "novel_structure_count": 0.0,
                "novel_structure_fraction": 0.0,
                "unique_structure_count": 0.0,
                "unique_structure_fraction": 0.0,
                "stable_count": 0.0,
                "stable_unique_count": 0.0,
                "stable_novel_count": 0.0,
                "stable_unique_novel_count": 0.0,
                "stable_fraction": 0.0,
                "stable_unique_fraction": 0.0,
                "stable_novel_fraction": 0.0,
                "stable_unique_novel_fraction": 0.0,
            }

        is_unique, unique_metrics = cls.uniqueness(
            candidate_structures, ltol, stol, angle_tol
        )
        is_novel, novel_metrics = cls.novelty(
            candidate_structures, reference_structures, ltol, stol, angle_tol
        )
        is_stable = (
            np.array(stable_flags, dtype=bool)
            if stable_flags is not None
            else np.zeros(total, dtype=bool)
        )

        stable_count = int(is_stable.sum())
        stable_unique = int((is_stable & is_unique).sum())
        stable_novel = int((is_stable & is_novel).sum())
        stable_unique_novel = int((is_stable & is_unique & is_novel).sum())

        results = {**unique_metrics, **novel_metrics}
        results.update(
            {
                "stable_count": float(stable_count),
                "stable_unique_count": float(stable_unique),
                "stable_novel_count": float(stable_novel),
                "stable_unique_novel_count": float(stable_unique_novel),
                "stable_fraction": float(stable_count / total),
                "stable_unique_fraction": float(stable_unique / total),
                "stable_novel_fraction": float(stable_novel / total),
                "stable_unique_novel_fraction": float(stable_unique_novel / total),
            }
        )

        return results


class ReferenceMetrics:
    """Metrics for comparing compositions and structures with reference compositions and structures."""

    @classmethod
    def composition_metrics(
        cls,
        structures: list[Structure],
        reference_structures: list[Structure],
    ) -> dict[str, float]:
        """
        Compute composition-based precision and recall metrics.

        Args:
            structures: List of structures to evaluate.
            reference_structures: List of reference structures.

        Returns:
            Dictionary with composition precision and recall.
        """
        if not structures or not reference_structures:
            return {
                "composition_precision": 0.0,
                "composition_recall": 0.0,
            }

        compositions = [struct.composition.reduced_formula for struct in structures]
        reference_compositions = [
            struct.composition.reduced_formula for struct in reference_structures
        ]
        reference_unique = set(reference_compositions)
        overlap = set(compositions).intersection(reference_unique)

        return {
            "composition_precision": float(len(overlap) / len(compositions)),
            "composition_recall": float(len(overlap) / len(reference_unique)),
        }

    @classmethod
    def structure_metrics(
        cls,
        structures: list[Structure],
        reference_structures: list[Structure],
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> dict[str, float]:
        """
        Compute structure-based precision and recall metrics.

        Args:
            structures: List of structures to evaluate.
            reference_structures: List of reference structures.
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance for structure matching in degrees.

        Returns:
            Dictionary with structure precision and recall.
        """
        if not structures or not reference_structures:
            return {
                "structure_precision": 0.0,
                "structure_recall": 0.0,
            }

        matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
        matched_structures_indices = set()
        for i, struct in enumerate(structures):
            matches = False
            for ref in reference_structures:
                try:
                    if matcher.fit(struct, ref):
                        matches = True
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {struct.composition} and {ref.composition}: {e}"
                    )
                    # Treat as non-matching on error, continue checking
            if matches:
                matched_structures_indices.add(i)

        return {
            "structure_precision": float(
                len(matched_structures_indices) / len(structures)
            ),
            "structure_recall": float(
                len(matched_structures_indices) / len(reference_structures)
            ),
        }

    @classmethod
    def calculate(
        cls,
        structures: list[Structure],
        reference_structures: list[Structure],
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
    ) -> dict[str, float]:
        """
        Calculate all reference comparison metrics.

        Args:
            structures: List of structures to evaluate.
            reference_structures: List of reference structures.
            ltol: Fractional length tolerance for structure matching.
            stol: Site tolerance for structure matching.
            angle_tol: Angle tolerance for structure matching in degrees.

        Returns:
            Dictionary containing all reference metrics.
        """
        comp_metrics = cls.composition_metrics(structures, reference_structures)
        struct_metrics = cls.structure_metrics(
            structures, reference_structures, ltol, stol, angle_tol
        )

        return {**comp_metrics, **struct_metrics}


class ConvexHullMetrics:
    """Metrics for assessing the convex hull properties of proposed materials."""

    @classmethod
    def energy_above_hull(
        cls,
        environment,  # ConvexHullEnvironment instance
        proposed_entries,  # List of PDEntries
    ) -> dict[str, float]:
        """
        Compute energy above hull metrics.

        Args:
            environment: A ConvexHullEnvironment instance.
            proposed_entries: List of PDEntries to evaluate.

        Returns:
            Dictionary with energy above hull statistics.
        """
        if not proposed_entries:
            return {
                "e_above_hull_min": 0.0,
                "e_above_hull_mean": 0.0,
                "e_above_hull_max": 0.0,
                "e_above_hull_std": 0.0,
            }

        hull_distances = [
            safe_e_above_hull(environment.observed_pd, entry)
            for entry in proposed_entries
        ]
        hull_distances_array = np.array(hull_distances, dtype=float)

        return {
            "e_above_hull_min": float(np.min(hull_distances_array)),
            "e_above_hull_mean": float(np.mean(hull_distances_array)),
            "e_above_hull_max": float(np.max(hull_distances_array)),
            "e_above_hull_std": float(np.std(hull_distances_array)),
        }

    @classmethod
    def energy_per_atom(cls, proposed_entries) -> dict[str, float]:
        """
        Compute energy per atom metrics.

        Args:
            proposed_entries: List of PDEntries to evaluate.

        Returns:
            Dictionary with energy per atom statistics.
        """
        if not proposed_entries:
            return {
                "energy_per_atom_min": 0.0,
                "energy_per_atom_max": 0.0,
                "energy_per_atom_mean": 0.0,
                "energy_per_atom_std": 0.0,
            }

        energies_array = np.array(
            [entry.energy_per_atom for entry in proposed_entries], dtype=float
        )

        return {
            "energy_per_atom_min": float(np.min(energies_array)),
            "energy_per_atom_max": float(np.max(energies_array)),
            "energy_per_atom_mean": float(np.mean(energies_array)),
            "energy_per_atom_std": float(np.std(energies_array)),
        }

    @classmethod
    def formation_energy_per_atom(
        cls,
        environment,  # ConvexHullEnvironment instance
        proposed_entries,  # List of PDEntries
    ) -> dict[str, float]:
        """
        Compute formation energy per atom metrics.

        Args:
            environment: A ConvexHullEnvironment instance.
            proposed_entries: List of PDEntries to evaluate.

        Returns:
            Dictionary with formation energy per atom statistics.
        """
        if not proposed_entries:
            return {
                "formation_energy_per_atom_min": 0.0,
                "formation_energy_per_atom_max": 0.0,
                "formation_energy_per_atom_mean": 0.0,
                "formation_energy_per_atom_std": 0.0,
            }

        formation_energies_array = np.array(
            [
                environment.observed_pd.get_form_energy_per_atom(entry)
                for entry in proposed_entries
            ],
            dtype=float,
        )

        return {
            "formation_energy_per_atom_min": float(np.min(formation_energies_array)),
            "formation_energy_per_atom_max": float(np.max(formation_energies_array)),
            "formation_energy_per_atom_mean": float(np.mean(formation_energies_array)),
            "formation_energy_per_atom_std": float(np.std(formation_energies_array)),
        }

    @classmethod
    def stability_metrics(
        cls,
        environment,  # ConvexHullEnvironment instance
        proposed_entries,  # List of PDEntries
    ) -> dict[str, float]:
        """
        Compute stability metrics based on energy above hull.

        Args:
            environment: A ConvexHullEnvironment instance.
            proposed_entries: List of PDEntries to evaluate.

        Returns:
            Dictionary with stability counts and fractions.
        """
        if not proposed_entries:
            return {
                "stable_fraction": 0.0,
                "stable_count": 0.0,
            }

        stable_count = sum(
            1
            for entry in proposed_entries
            if safe_e_above_hull(environment.observed_pd, entry)
            <= environment.stability_tolerance
        )
        entries_count = len(proposed_entries)

        return {
            "stable_fraction": float(stable_count / entries_count),
            "stable_count": float(stable_count),
        }

    @classmethod
    def calculate(
        cls,
        environment,  # ConvexHullEnvironment instance
        proposed_entries=None,  # List of PDEntries to evaluate, or None to use env.proposed_entries
    ) -> dict[str, float]:
        """
        Calculate all convex hull metrics.

        Args:
            environment: A ConvexHullEnvironment instance.
            proposed_entries: Optional list of PDEntries to evaluate. If None,
                                    uses env.proposed_entries.

        Returns:
            Dictionary containing all convex hull metrics.
        """
        if proposed_entries is None:
            proposed_entries = environment.proposed_entries

        if not proposed_entries:
            return {
                "e_above_hull_min": 0.0,
                "e_above_hull_mean": 0.0,
                "e_above_hull_max": 0.0,
                "e_above_hull_std": 0.0,
                "energy_per_atom_min": 0.0,
                "energy_per_atom_max": 0.0,
                "energy_per_atom_mean": 0.0,
                "energy_per_atom_std": 0.0,
                "formation_energy_per_atom_min": 0.0,
                "formation_energy_per_atom_max": 0.0,
                "formation_energy_per_atom_mean": 0.0,
                "formation_energy_per_atom_std": 0.0,
                "stable_fraction": 0.0,
                "stable_count": 0.0,
            }

        hull_metrics = cls.energy_above_hull(environment, proposed_entries)
        energy_metrics = cls.energy_per_atom(proposed_entries)
        formation_metrics = cls.formation_energy_per_atom(environment, proposed_entries)
        stability_metrics = cls.stability_metrics(environment, proposed_entries)

        return {
            **hull_metrics,
            **energy_metrics,
            **formation_metrics,
            **stability_metrics,
        }


class DiscoveryCurveMetrics:
    """Metrics for evaluating discovery performance over time."""

    @staticmethod
    def compute_area_under_curve(
        discoveries: np.ndarray,
        queries: np.ndarray | None = None,
        normalize: bool = False,
    ) -> float:
        """
        Compute the area under the discovery curve from arrays.

        Args:
            discoveries: Array of cumulative discoveries at each step.
            queries: Optional array of query counts. If None, assumes discoveries represents
                    step-by-step discoveries and queries will be [0, 1, 2, ..., len(discoveries)-1].
            normalize: If True, normalize by queries^2 / 2, which is the maximum area for
                       1 discovery per query (linear growth). This gives a value representing
                       efficiency relative to the ideal 1 discovery/query rate.

        Returns:
            Area under the discovery curve using trapezoidal integration.
            If normalize=True, returns normalized value.
        """
        discoveries = np.array(discoveries, dtype=float)

        if queries is None:
            # Assume discoveries represents step-by-step discoveries
            queries = np.arange(len(discoveries), dtype=float)
        else:
            queries = np.array(queries, dtype=float)

        if len(queries) == 0 or len(discoveries) == 0:
            return 0.0

        audc = float(np.trapezoid(discoveries, queries))

        if normalize:
            # Normalize by queries^2 / 2, which is the maximum area under curve for
            # 1 discovery per query (linear growth from 0 to queries[-1] discoveries)
            max_area = (queries[-1] ** 2) / 2.0
            if max_area > 0:
                return audc / max_area
            else:
                return 0.0

        return audc

    @staticmethod
    def _extract_discovery_curve(
        metrics_history: list[dict[str, Any]],
        metric_key: str = "num_newly_discovered_structures",
        query_key: str = "queries_used",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract queries and cumulative discoveries from metrics history.

        Args:
            metrics_history: List of metric dictionaries from each step of an episode.
                Each dictionary should contain at least the specified metric_key and query_key.
                Queries should be in order (not sorted).
            metric_key: Key to extract discoveries from metrics dictionary
                       (default: 'num_newly_discovered_structures').
            query_key: Key to extract query counts from metrics dictionary
                      (default: 'queries_used').

        Returns:
            Tuple of (queries_unique, discoveries_unique) arrays, with duplicates removed
            and starting from query 0 with 0 discoveries.
        """
        if not metrics_history:
            return np.array([0.0]), np.array([0.0])

        queries = []
        cumulative_discoveries = []

        for metrics in metrics_history:
            query_count = metrics.get(query_key, 0)
            num_discovered = metrics.get(metric_key, 0)
            queries.append(query_count)
            cumulative_discoveries.append(num_discovered)

        if not queries:
            return np.array([0.0]), np.array([0.0])

        queries = np.array(queries, dtype=float)
        cumulative_discoveries = np.array(cumulative_discoveries, dtype=float)

        # Remove duplicates (queries are assumed to be in order)
        unique_indices = np.concatenate(([True], np.diff(queries) > 0))
        queries_unique = queries[unique_indices]
        discoveries_unique = cumulative_discoveries[unique_indices]

        # Ensure we start from query 0 with 0 discoveries
        if queries_unique[0] > 0:
            queries_unique = np.concatenate([[0.0], queries_unique])
            discoveries_unique = np.concatenate([[0.0], discoveries_unique])

        return queries_unique, discoveries_unique

    @classmethod
    def area_under_discovery_curve(
        cls,
        metrics_history: list[dict[str, Any]] | None = None,
        metric_key: str = "num_newly_discovered_structures",
        query_key: str = "queries_used",
        normalize: bool = False,
        discoveries: np.ndarray | None = None,
        queries: np.ndarray | None = None,
    ) -> float:
        """
        Compute the area under the discovery curve (discoveries vs queries).

        Can be called with either:
        1. metrics_history (from episode) - extracts queries and discoveries from dicts
        2. discoveries array (and optional queries array) - computes directly from arrays

        Args:
            metrics_history: Optional list of metric dictionaries from each step of an episode.
                            If provided, extracts queries and discoveries from these dicts.
            metric_key: Key to extract discoveries from metrics dictionary
                       (default: 'num_newly_discovered_structures').
            query_key: Key to extract query counts from metrics dictionary
                      (default: 'queries_used').
            normalize: If True, normalize by queries^2 / 2, which is the maximum area for
                       1 discovery per query (linear growth). This gives a value representing
                       efficiency relative to the ideal 1 discovery/query rate.
            discoveries: Optional array of cumulative discoveries. If provided, uses this directly
                         instead of extracting from metrics_history.
            queries: Optional array of query counts. If provided with discoveries, uses these
                    directly. If None and discoveries provided, assumes step-by-step (0, 1, 2, ...).

        Returns:
            Area under the discovery curve using trapezoidal integration.
            If normalize=True, returns normalized value between 0 and 1.
        """
        if discoveries is not None:
            # Direct array computation
            return cls.compute_area_under_curve(discoveries, queries, normalize)
        elif metrics_history is not None:
            # Extract from metrics history
            queries_array, discoveries_array = cls._extract_discovery_curve(
                metrics_history, metric_key, query_key
            )
            return cls.compute_area_under_curve(
                discoveries_array, queries_array, normalize
            )
        else:
            raise ValueError("Must provide either metrics_history or discoveries array")

    @classmethod
    def enhancement_factor(
        cls,
        proposal_metrics_history: list[dict[str, Any]],
        baseline_metrics_history: list[dict[str, Any]],
        cap: float = 100.0,
        return_array: bool = False,
        metric_key: str = "num_newly_discovered_structures",
        query_key: str = "queries_used",
    ) -> float | np.ndarray:
        """
        Compute enhancement factor: ratio of proposal discoveries to baseline discoveries.

        Args:
            proposal_metrics_history: List of metric dictionaries for the proposal method.
            baseline_metrics_history: List of metric dictionaries for the baseline method.
            cap: Maximum value for the enhancement factor (default: 10.0).
            return_array: If True, returns array of enhancement factors at each query point.
                          If False, returns only the final enhancement factor.
            metric_key: Key to extract discoveries from metrics dictionary
                       (default: 'num_newly_discovered_structures').
            query_key: Key to extract query counts from metrics dictionary
                      (default: 'queries_used').

        Returns:
            If return_array=False: Single enhancement factor at final query.
            If return_array=True: Array of enhancement factors at each query point.
        """
        proposal_queries, proposal_discoveries = cls._extract_discovery_curve(
            proposal_metrics_history, metric_key, query_key
        )
        baseline_queries, baseline_discoveries = cls._extract_discovery_curve(
            baseline_metrics_history, metric_key, query_key
        )

        if len(proposal_queries) == 0 or len(baseline_queries) == 0:
            if return_array:
                return np.array([])
            return 0.0

        if return_array:
            # Compute enhancement factor at each query point
            # Interpolate baseline discoveries to match proposal query points
            baseline_at_proposal_queries = np.interp(
                proposal_queries,
                baseline_queries,
                baseline_discoveries,
                left=0.0,
                right=baseline_discoveries[-1]
                if len(baseline_discoveries) > 0
                else 0.0,
            )
            # Match notebook pattern: use safe baseline and cap result
            baseline_safe = np.maximum(baseline_at_proposal_queries, 1e-8)
            enhancement_factors = np.minimum(proposal_discoveries / baseline_safe, cap)
            return enhancement_factors.astype(float)
        else:
            # Return only final enhancement factor (backward compatibility)
            final_query = proposal_queries[-1]
            final_proposal_discoveries = proposal_discoveries[-1]

            baseline_at_final_query = np.interp(
                final_query,
                baseline_queries,
                baseline_discoveries,
                left=0.0,
                right=baseline_discoveries[-1]
                if len(baseline_discoveries) > 0
                else 0.0,
            )

            baseline_safe = max(baseline_at_final_query, 1e-8)
            return float(min(final_proposal_discoveries / baseline_safe, cap))

    @classmethod
    def acceleration_factor(
        cls,
        proposal_metrics_history: list[dict[str, Any]],
        baseline_metrics_history: list[dict[str, Any]],
        performance_bins: np.ndarray | list[float] | None = None,
        percentage: bool = False,
        target_discoveries: int | None = None,
        metric_key: str = "num_newly_discovered_structures",
        query_key: str = "queries_used",
    ) -> float | np.ndarray:
        """
        Compute acceleration factor: ratio of baseline queries to proposal queries needed
        to reach target performance levels.

        Args:
            proposal_metrics_history: List of metric dictionaries for the proposal method.
            baseline_metrics_history: List of metric dictionaries for the baseline method.
            performance_bins: Array of performance levels to evaluate. If None and target_discoveries
                is None, uses final baseline discoveries as single target.
            percentage: If True, performance_bins are interpreted as percentages (0.0-1.0) of max
                baseline performance. If False, they are interpreted as absolute discovery counts.
            target_discoveries: Deprecated. Use performance_bins instead. If provided and
                performance_bins is None, uses this as a single target.
            metric_key: Key to extract discoveries from metrics dictionary
                       (default: 'num_newly_discovered_structures').
            query_key: Key to extract query counts from metrics dictionary
                      (default: 'queries_used').

        Returns:
            If performance_bins is None and target_discoveries is provided: Single acceleration factor.
            If performance_bins is provided: Array of acceleration factors for each performance bin.
        """
        proposal_queries, proposal_discoveries = cls._extract_discovery_curve(
            proposal_metrics_history, metric_key, query_key
        )
        baseline_queries, baseline_discoveries = cls._extract_discovery_curve(
            baseline_metrics_history, metric_key, query_key
        )

        if len(proposal_queries) == 0 or len(baseline_queries) == 0:
            if performance_bins is not None:
                return np.array([0.0] * len(performance_bins))
            return 0.0

        # Handle backward compatibility: if target_discoveries provided and no performance_bins
        if performance_bins is None and target_discoveries is not None:
            performance_bins = [target_discoveries]
            percentage = False

        # Determine performance levels to evaluate
        if performance_bins is None:
            # Default: use final baseline discoveries as single target
            performance_levels = np.array([baseline_discoveries[-1]])
        else:
            performance_bins = np.array(performance_bins)
            if percentage:
                max_baseline = np.max(baseline_discoveries)
                performance_levels = performance_bins * max_baseline
            else:
                performance_levels = performance_bins

        accel_factors = []
        for target_performance in performance_levels:
            # Find first index where performance >= target_performance
            proposal_reaches_target = np.where(
                proposal_discoveries >= target_performance
            )[0]
            baseline_reaches_target = np.where(
                baseline_discoveries >= target_performance
            )[0]

            # If target performance is never achieved, set to inf or 0
            if len(proposal_reaches_target) == 0:
                proposal_queries_needed = float("inf")
            else:
                proposal_query_idx = proposal_reaches_target[0]
                proposal_queries_needed = proposal_queries[proposal_query_idx]

            if len(baseline_reaches_target) == 0:
                baseline_queries_needed = float("inf")
            else:
                baseline_query_idx = baseline_reaches_target[0]
                baseline_queries_needed = baseline_queries[baseline_query_idx]

            # Calculate acceleration factor (baseline_queries / proposal_queries)
            if proposal_queries_needed == 0 and baseline_queries_needed == 0:
                accel_factor_val = 1.0
            elif proposal_queries_needed == 0:
                accel_factor_val = float("inf")
            elif baseline_queries_needed == float(
                "inf"
            ) or proposal_queries_needed == float("inf"):
                accel_factor_val = 0.0
            else:
                accel_factor_val = baseline_queries_needed / proposal_queries_needed

            accel_factors.append(accel_factor_val)

        result = np.array(accel_factors, dtype=float)

        # Return single value if only one target (backward compatibility)
        if len(result) == 1:
            return float(result[0])
        return result

    @classmethod
    def calculate(
        cls,
        metrics_history: list[dict[str, Any]],
        *,
        baseline_metrics_history: list[dict[str, Any]] | None = None,
        metric_key: str = "num_newly_discovered_structures",
        query_key: str = "queries_used",
    ) -> dict[str, float]:
        """
        Calculate discovery curve metrics including area under curve, enhancement factor,
        and acceleration factor.

        Note: This method returns final values only. To get enhancement factors at each
        query point, use `enhancement_factor(..., return_array=True)`. To get acceleration
        factors for multiple performance bins, use `acceleration_factor(..., performance_bins=...)`.

        Args:
            metrics_history: List of metric dictionaries from each step of an episode.
                Each dictionary should contain at least the specified metric_key and query_key.
                Queries should be in order (not sorted).
            baseline_metrics_history: Optional list of baseline metric dictionaries with
                the same structure as metrics_history. If provided, enhancement and
                acceleration factors will be computed against this baseline.
            metric_key: Key to extract discoveries from metrics dictionary
                       (default: 'num_newly_discovered_structures').
                Common options: 'num_newly_discovered_structures', 'num_newly_discovered_stable'.
            query_key: Key to extract query counts from metrics dictionary
                      (default: 'queries_used').

        Returns:
            Dictionary containing:
            - area_under_discovery_curve: Area under the cumulative discoveries vs queries curve (unnormalized)
            - area_under_discovery_curve_normalized: Normalized area under curve (divided by queries^2/2)
            - enhancement_factor: Final enhancement factor (actual/baseline discoveries at final query)
            - acceleration_factor: Final acceleration factor (baseline queries / actual queries for final discoveries)
        """
        audc = cls.area_under_discovery_curve(
            metrics_history, metric_key=metric_key, query_key=query_key, normalize=False
        )
        audc_normalized = cls.area_under_discovery_curve(
            metrics_history, metric_key=metric_key, query_key=query_key, normalize=True
        )

        enhancement_factor = 0.0
        acceleration_factor = 0.0

        if baseline_metrics_history is not None:
            enhancement_factor = cls.enhancement_factor(
                metrics_history,
                baseline_metrics_history,
                return_array=False,
                metric_key=metric_key,
                query_key=query_key,
            )
            acceleration_factor = cls.acceleration_factor(
                metrics_history,
                baseline_metrics_history,
                metric_key=metric_key,
                query_key=query_key,
            )

        return {
            "area_under_discovery_curve": audc,
            "area_under_discovery_curve_normalized": audc_normalized,
            "enhancement_factor": enhancement_factor,
            "acceleration_factor": acceleration_factor,
        }
