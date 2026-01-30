"""
Convex hull exploration environment.

Maintains an observed phase diagram that starts from elemental references and
is updated with each oracle-evaluated proposal. Computes formation energies and
energy-above-hull relative to both the observed PD and the ground-truth PD from
the dataset, and tracks discovery metrics against the ground truth.

IMPORTANT: Dataset vs Environment Filtering
-------------------------------------------
The PhaseDiagramDataset provides raw, unfiltered data from Materials Project (only
thermo_types filtering). The ConvexHullEnvironment applies task-specific filters to
define the "effective ground truth" for discovery:
  - filter_by_smact: Exclude chemically implausible phases
  - max_stoichiometry: Exclude phases with too many atoms per unit cell
  - stability_tolerance + include_near_stable_from_ground_truth: Include/exclude
    phases near the convex hull

All discovery metrics (recall, precision, novelty) are computed against this
filtered ground truth, not the raw dataset. This allows the same dataset to be
reused for different discovery task difficulties.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
from pymatgen.analysis.phase_diagram import PDEntry, PDPlotter, PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure

from made.data.chemical_system import PhaseDiagramDataset
from made.envs.base import Environment
from made.evaluation import (
    CompositionDiversityMetrics,
    ConvexHullMetrics,
    DiscoveryCurveMetrics,
    NoveltyMetrics,
    ReferenceMetrics,
    StructureDiversityMetrics,
    ValidityMetrics,
)
from made.evaluation.metrics import is_smact_valid
from made.oracles.base import Oracle
from made.utils.convex_hull_utils import (
    extract_structure_from_entry,
    formulas_within_epsilon,
    safe_e_above_hull,
    structure_result_to_entry,
)

logger = logging.getLogger(__name__)


class ConvexHullEnvironment(Environment):
    """Environment for phase-diagram (convex hull) discovery.

    Tracks both formula-based and structure-based discovery metrics:
    - Formula-based: Traditional phase diagram metrics (recall, precision)
    - Structure-based: Structural novelty using pymatgen StructureMatcher

    Args:
        budget: Maximum number of oracle queries allowed
        dataset: PhaseDiagramDataset containing ground truth phase diagram
        oracle: Oracle for energy evaluation
        start_with_all_stable: If True, start with all stable entries re-evaluated by oracle.
                              If False, start with elemental references only (default)
        compute_elemental_from_oracle: If True, compute elemental reference energies
                                     using oracle from dataset structures (default)
        stability_tolerance: Energy tolerance (eV/atom) for considering a phase stable
                           relative to the convex hull (default: 1e-8). Also used as the
                           epsilon for near-stable inclusion when the flag below is true.
        include_near_stable_from_ground_truth: If true, treat ground-truth entries within
                           stability_tolerance of the hull as part of the stable set for
                           initialization and target metrics; if false, use strictly stable only.
        structure_matcher_ltol: The maximum absolute length difference for matching two structures.
        structure_matcher_stol: The maximum absolute site tolerance for matching two structures.
        structure_matcher_angle_tol: The maximum absolute angle difference for matching two structures.
        structure_matcher_primitive_cell: If True, use the primitive cell for matching two structures.
        filter_by_smact: If True, filter out entries that don't pass SMACT validity checks
                         from the ground truth phase diagram on initialization.
        max_stoichiometry: If specified, filter out entries with number of atoms in the unit cell
                          greater than this value (default: None, no filtering).
    """

    def __init__(
        self,
        dataset: PhaseDiagramDataset,
        oracle: Oracle,
        budget: int,
        start_with_all_stable: bool = False,
        compute_elemental_from_oracle: bool = True,
        stability_tolerance: float = 1e-8,
        include_near_stable_from_ground_truth: bool = False,
        structure_matcher_ltol: float = 0.2,
        structure_matcher_stol: float = 0.3,
        structure_matcher_angle_tol: float = 5,
        structure_matcher_primitive_cell: bool = True,
        filter_by_smact: bool = False,
        max_stoichiometry: int | None = None,
    ):
        super().__init__(dataset, oracle, budget)

        if not isinstance(dataset, PhaseDiagramDataset):
            raise TypeError(
                "ConvexHullEnvironment requires a PhaseDiagramDataset as the dataset."
            )

        self.start_with_all_stable = start_with_all_stable
        self.compute_elemental_from_oracle = compute_elemental_from_oracle
        self.stability_tolerance = stability_tolerance
        self.include_near_stable_from_ground_truth = bool(
            include_near_stable_from_ground_truth
        )
        self.filter_by_smact = filter_by_smact
        self.max_stoichiometry = max_stoichiometry
        # Initialize structure matcher for novelty detection
        self.structure_matcher = StructureMatcher(
            ltol=structure_matcher_ltol,
            stol=structure_matcher_stol,
            angle_tol=structure_matcher_angle_tol,
            primitive_cell=structure_matcher_primitive_cell,
        )
        # Persist matcher parameters for use in other metrics
        self.structure_matcher_ltol = structure_matcher_ltol
        self.structure_matcher_stol = structure_matcher_stol
        self.structure_matcher_angle_tol = structure_matcher_angle_tol

        # Ground-truth PD from dataset
        ground_truth_pd = dataset.get_ground_truth_pd()

        # Filter by SMACT validity and/or max stoichiometry if requested
        if self.filter_by_smact or self.max_stoichiometry is not None:
            filtered_entries = []
            for entry in ground_truth_pd.all_entries:
                structure = extract_structure_from_entry(entry)

                # Apply max stoichiometry filter (number of atoms in unit cell)
                if self.max_stoichiometry is not None and structure is not None:
                    num_atoms = structure.num_sites
                    if num_atoms > self.max_stoichiometry:
                        continue

                # Apply SMACT filter
                if self.filter_by_smact:
                    if structure is not None and is_smact_valid(structure):
                        filtered_entries.append(entry)
                    elif structure is None:
                        # Keep entries without structures (e.g., elemental references)
                        filtered_entries.append(entry)
                else:
                    filtered_entries.append(entry)

            filter_msg = []
            if self.filter_by_smact:
                filter_msg.append("SMACT-valid")
            if self.max_stoichiometry is not None:
                filter_msg.append(f"num_atoms<={self.max_stoichiometry}")
            logger.info(
                f"Filtered ground truth PD from {len(ground_truth_pd.all_entries)} to "
                f"{len(filtered_entries)} entries ({', '.join(filter_msg)})"
            )
            self.ground_truth_pd: PhaseDiagram = PhaseDiagram(filtered_entries)
        else:
            self.ground_truth_pd: PhaseDiagram = ground_truth_pd

        # Extract ground-truth stable structures and formulas from the filtered PD
        # (optionally include near-stable entries within stability_tolerance)
        epsilon_gt = (
            self.stability_tolerance
            if self.include_near_stable_from_ground_truth
            else 0.0
        )
        self.ground_truth_stable_structures: list[Structure] = []
        self.ground_truth_stable_formulas: set[str] = set()

        # Compute stable entries from the already-filtered ground truth PD
        for entry in self.ground_truth_pd.all_entries:
            e_above_hull = safe_e_above_hull(self.ground_truth_pd, entry)
            if e_above_hull <= epsilon_gt:
                structure = extract_structure_from_entry(entry)
                self.ground_truth_stable_formulas.add(
                    entry.composition.reduced_composition.alphabetical_formula
                )
                if structure is not None:
                    self.ground_truth_stable_structures.append(structure)
        # Compute elemental energies using oracle if requested
        if self.compute_elemental_from_oracle:
            self._compute_elemental_energies_with_oracle()

        # Initialize observed PD - either with elemental references only or all stable entries
        if self.start_with_all_stable:
            # Include (near-)stable entries from ground truth, re-evaluated with oracle
            self.observed_entries: list[PDEntry] = (
                self._get_oracle_evaluated_stable_entries()
            )
            logger.info(
                f"Starting PD with {len(self.observed_entries)} oracle-evaluated (near-)stable entries (epsilon={epsilon_gt})"
            )
        else:
            # Start with elemental references only,
            self.observed_entries: list[PDEntry] = (
                self._get_oracle_evaluated_elemental_entries()
            )
            logger.info(
                f"Starting PD with {len(self.observed_entries)} elemental reference entries"
            )

        self.observed_pd: PhaseDiagram = PhaseDiagram(self.observed_entries)

        # Store initial structures for novelty comparison
        self.initial_structures: list[Structure] = self._get_initial_structures()

        # Track formulas within the observed stability tolerance at initialization
        self.initial_observed_stable_formulas: set[str] = formulas_within_epsilon(
            self.observed_pd, self.stability_tolerance
        )
        # Cache GT items missing at initialization (formulas and structures)
        self.gt_formulas_missing_initial: set[str] = (
            self.ground_truth_stable_formulas - self.initial_observed_stable_formulas
        )
        # Precompute ground-truth stable structures that were not present initially
        self.gt_structures_missing_initial: list[Structure] = []
        for gt_struct in self.ground_truth_stable_structures:
            matches_initial = False
            for init_struct in self.initial_structures:
                try:
                    if self.structure_matcher.fit(gt_struct, init_struct):
                        matches_initial = True
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {gt_struct.composition} and {init_struct.composition}: {e}"
                    )
                    # Treat as non-matching on error
            if not matches_initial:
                self.gt_structures_missing_initial.append(gt_struct)

        # Track metrics over time
        self.metrics_history: list[dict[str, Any]] = []

        # Track proposed structures and metrics
        self.proposed_entries: list[PDEntry] = []

        # Track newly discovered entries (structures are in PDEntry attributes)
        self.newly_discovered_entries: list[PDEntry] = []

    # ---- Lifecycle ----
    def reset(self) -> dict[str, Any]:
        self.query_count = 0
        self.history = []

        # Reset observed PD based on initialization setting
        if self.start_with_all_stable:
            self.observed_entries = self._get_oracle_evaluated_stable_entries()
            message = f"Environment reset: {len(self.observed_entries)} oracle-evaluated (near-)stable entries."
        else:
            self.observed_entries = self._get_oracle_evaluated_elemental_entries()
            message = "Environment reset: elemental references only."

        self.observed_pd = PhaseDiagram(self.observed_entries)
        self.initial_observed_stable_formulas = formulas_within_epsilon(
            self.observed_pd, self.stability_tolerance
        )
        self.metrics_history = []

        # Reset proposal tracking
        self.proposed_entries = []

        # Reset novelty tracking
        self.initial_structures = self._get_initial_structures()
        # Recompute caches for GT items missing at initialization
        self.gt_formulas_missing_initial = (
            self.ground_truth_stable_formulas - self.initial_observed_stable_formulas
        )
        self.gt_structures_missing_initial = []
        for gt_struct in self.ground_truth_stable_structures:
            matches_initial = False
            for init_struct in self.initial_structures:
                try:
                    if self.structure_matcher.fit(gt_struct, init_struct):
                        matches_initial = True
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {gt_struct.composition} and {init_struct.composition}: {e}"
                    )
                    # Treat as non-matching on error
            if not matches_initial:
                self.gt_structures_missing_initial.append(gt_struct)
        self.newly_discovered_entries: list[PDEntry] = []

        self.done = False
        return {"message": message}

    # ---- Interaction ----
    def step(self, proposal: Structure) -> tuple[dict[str, Any], bool]:
        if self.is_done():
            raise RuntimeError("Environment is done. Reset required.")

        oracle_result = self.oracle.evaluate(proposal)
        self.query_count += 1

        # Convert oracle result to PDEntry
        pd_entry = structure_result_to_entry(proposal, oracle_result)

        # Add to proposed entries list for tracking
        self.proposed_entries.append(pd_entry)

        # Update observed PD with new entry
        self.observed_entries.append(pd_entry)
        self.observed_pd = PhaseDiagram(self.observed_entries)

        # Compute energies and stabilities vs observed PD only for the observation
        obs_e_form_pa = self.observed_pd.get_form_energy_per_atom(pd_entry)
        obs_ehull = self._safe_e_above_hull(self.observed_pd, pd_entry)
        is_obs_stable = obs_ehull <= self.stability_tolerance

        reduced_formula = pd_entry.composition.reduced_composition.alphabetical_formula

        # Check if this is a newly discovered structure (structurally novel) and add to newly_discovered_entries
        is_newly_discovered = self._check_structural_novelty(pd_entry)

        # Compute validity metrics for this individual proposal
        proposal_validity = ValidityMetrics.calculate(structures=[proposal])

        # Update top-line discovery metrics (internal) and add to metrics_history
        self._compute_metrics()

        # Do not include any ground-truth-derived values in the observation
        obs = {
            "proposal": proposal,
            "oracle_result": oracle_result,
            "reduced_formula": str(reduced_formula),
            "formation_energy_per_atom": float(obs_e_form_pa),
            "energy_per_atom": float(oracle_result["energy_per_atom"]),
            "e_above_hull": float(obs_ehull),
            "is_stable": bool(is_obs_stable),
            "is_newly_discovered": bool(is_newly_discovered),
            **{f"validity_{k}": v for k, v in proposal_validity.items()},
        }

        self.history.append(obs)

        # Stop when budget exhausted
        done = self.is_done()
        return obs, done

    # ---- Accessors ----
    def get_state(self) -> dict[str, Any]:
        """Return the current state of the environment."""
        last_observation = self.history[-1] if self.history else None
        if last_observation and isinstance(last_observation["proposal"], Structure):
            last_observation["proposal"] = last_observation["proposal"].as_dict()
        return {
            "query_count": self.query_count,
            "elements": [str(e) for e in self.dataset.elements],
            "phase_diagram_all_entries": [
                e.as_dict() for e in self.observed_pd.all_entries
            ],
            "last_observation": last_observation,
            "stability_tolerance": self.stability_tolerance,
        }

    def get_pd_plot(
        self,
        ground_truth: bool = False,
        show_unstable: float = 1.0,
        backend: str = "plotly",
        show: bool = False,
    ):
        """Get a plot of the phase diagram.

        Args:
            ground_truth: Whether to plot the ground truth PD. If False, plots the observed PD.
            show_unstable: The energy above hull threshold for showing unstable phases.
            backend: The backend to use for plotting.
            show: Whether to show the plot.
        """
        fig = PDPlotter(
            self.ground_truth_pd if ground_truth else self.observed_pd,
            show_unstable=show_unstable,
            backend=backend,
        ).get_plot()
        if show:
            if backend == "plotly":
                fig.show()
            elif backend == "matplotlib":
                plt.show()
        return fig

    def get_stable_entries(self, epsilon: float | None = None) -> list[PDEntry]:
        """Return observed entries on or within epsilon of the hull.

        Args:
            epsilon: eV/atom threshold. If None or <= 0, returns strictly stable entries.
        """
        if epsilon is None or epsilon <= 0:
            return list(self.observed_pd.stable_entries)
        entries: list[PDEntry] = []
        for e in self.observed_pd.all_entries:
            try:
                if float(self._safe_e_above_hull(self.observed_pd, e)) <= float(
                    epsilon
                ):
                    entries.append(e)
            except Exception:
                continue
        return entries

    # ---- Evaluation-only accessors (may include ground-truth-derived metrics) ----
    def get_metrics_history(self) -> list[dict[str, Any]]:
        return list(self.metrics_history)

    def get_latest_metrics(self) -> dict[str, Any]:
        return self.metrics_history[-1] if self.metrics_history else {}

    def get_proposal_structures(self) -> list[Structure]:
        """Get all proposed structures.

        Returns a list of all structures that were proposed, excluding initial structures.
        """
        structures = []
        for entry in self.proposed_entries:
            struct = extract_structure_from_entry(entry)
            if struct is not None:
                structures.append(struct)
        return structures

    # ---- Helpers ----

    def _compute_metrics(self) -> dict[str, Any]:
        """Compute comprehensive discovery metrics including structural novelty."""

        ##### deprecated metrics, kept for backwards compatibility
        observed_stable_formulas = formulas_within_epsilon(
            self.observed_pd, self.stability_tolerance
        )
        gt_stable = self.ground_truth_stable_formulas
        # Only count newly observed stable formulas that were not present at initialization
        new_observed_stable_formulas = (
            observed_stable_formulas - self.initial_observed_stable_formulas
        )
        correct_new = new_observed_stable_formulas & gt_stable

        num_obs = len(observed_stable_formulas)
        num_gt = len(gt_stable)
        num_correct_new = len(correct_new)
        # Denominator is the number of ground-truth stable formulas that were NOT present initially (cached)
        num_gt_missing_initial = len(self.gt_formulas_missing_initial)
        recall_formula = (
            num_correct_new / num_gt_missing_initial
            if num_gt_missing_initial > 0
            else 0.0
        )
        # precision is the number of correct new formulas divided by the number of queries used
        precision_formula = (
            num_correct_new / self.query_count if self.query_count > 0 else 0.0
        )

        # New structure-based novelty metrics
        num_newly_discovered = len(self.newly_discovered_entries)
        discovery_efficiency = (
            num_newly_discovered / self.query_count if self.query_count > 0 else 0.0
        )

        # Stable newly discovered structures
        num_newly_discovered_stable = len(
            [
                entry
                for entry in self.newly_discovered_entries
                if self._safe_e_above_hull(self.observed_pd, entry)
                <= self.stability_tolerance
            ]
        )

        stable_discovery_efficiency = (
            num_newly_discovered_stable / self.query_count
            if self.query_count > 0
            else 0.0
        )

        # Ground-truth structure-based precision/recall using cached missing-initial set
        # Count how many unique GT structures (from the missing-initial set) have been discovered
        matched_gt_indices: set[int] = set()
        for discovered_entry in self.newly_discovered_entries:
            disc_struct = extract_structure_from_entry(discovered_entry)
            if disc_struct is None:
                continue
            for idx, gt_struct in enumerate(self.gt_structures_missing_initial):
                if idx in matched_gt_indices:
                    continue
                try:
                    if self.structure_matcher.fit(disc_struct, gt_struct):
                        matched_gt_indices.add(idx)
                        break
                except Exception as e:
                    logger.warning(
                        f"StructureMatcher.fit() failed for structures {disc_struct.composition} and {gt_struct.composition}: {e}"
                    )
                    # Treat as non-matching on error, continue to next structure

        num_gt_structures_missing_initial = len(self.gt_structures_missing_initial)
        num_correct_structures = len(matched_gt_indices)
        recall_structure = (
            num_correct_structures / num_gt_structures_missing_initial
            if num_gt_structures_missing_initial > 0
            else 0.0
        )
        precision_structure = (
            num_correct_structures / self.query_count if self.query_count > 0 else 0.0
        )

        metrics = {
            "queries_used": self.query_count,
            # Formula-based metrics
            "num_observed_stable_formulas": num_obs,
            "num_ground_truth_stable_formulas": num_gt,
            "num_gt_formulas_missing_initial": num_gt_missing_initial,
            "num_correct_stable_formulas": num_correct_new,
            "recall_formula": recall_formula,
            "precision_formula": precision_formula,
            # Structure-based novelty metrics
            "num_newly_discovered_structures": num_newly_discovered,
            "num_newly_discovered_stable": num_newly_discovered_stable,
            "discovery_efficiency": discovery_efficiency,
            "stable_discovery_efficiency": stable_discovery_efficiency,
            "num_initial_structures": len(self.initial_structures),
            # Ground-truth structure-based metrics (excluding initial)
            "num_ground_truth_stable_structures": len(
                self.ground_truth_stable_structures
            ),
            "num_gt_structures_missing_initial": num_gt_structures_missing_initial,
            "num_correct_stable_structures": num_correct_structures,
            "recall_structure": recall_structure,
            "precision_structure": precision_structure,
        }

        ####### new metrics

        # Extract structures from proposed entries for metrics
        proposed_structures: list[Structure] = []
        stable_proposed_structures: list[Structure] = []
        newly_discovered_structures: list[Structure] = []
        stable_flags: list[bool] = []

        for entry in self.proposed_entries:
            structure = extract_structure_from_entry(entry)
            if structure is None:
                continue

            proposed_structures.append(structure)

            hull_dist = self._safe_e_above_hull(self.observed_pd, entry)
            is_stable_flag = hull_dist <= self.stability_tolerance
            stable_flags.append(bool(is_stable_flag))
            if is_stable_flag:
                stable_proposed_structures.append(structure)

        for entry in self.newly_discovered_entries:
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                newly_discovered_structures.append(structure)

        # Validity metrics for proposed structures
        validity_metrics = ValidityMetrics.calculate(
            structures=proposed_structures,
        )

        # Diversity metrics (composition + structure) for proposed structures
        composition_metrics_all = CompositionDiversityMetrics.calculate(
            proposed_structures,
        )
        structure_metrics_all = StructureDiversityMetrics.calculate(
            proposed_structures,
            ltol=self.structure_matcher_ltol,
            stol=self.structure_matcher_stol,
            angle_tol=self.structure_matcher_angle_tol,
        )

        composition_metrics_stable: dict[str, float] = {}
        structure_metrics_stable: dict[str, float] = {}
        if stable_proposed_structures:
            composition_metrics_stable = CompositionDiversityMetrics.calculate(
                stable_proposed_structures
            )
            structure_metrics_stable = StructureDiversityMetrics.calculate(
                stable_proposed_structures,
                ltol=self.structure_matcher_ltol,
                stol=self.structure_matcher_stol,
                angle_tol=self.structure_matcher_angle_tol,
            )

        # Novelty/uniqueness metrics over ALL proposed structures vs initial references
        novelty_metrics: dict[str, float] = {}
        if proposed_structures and self.initial_structures:
            novelty_metrics = NoveltyMetrics.calculate(
                candidate_structures=proposed_structures,
                reference_structures=self.initial_structures,
                stable_flags=stable_flags,
                ltol=self.structure_matcher_ltol,
                stol=self.structure_matcher_stol,
                angle_tol=self.structure_matcher_angle_tol,
            )

        # Reference metrics
        reference_metrics = ReferenceMetrics.calculate(
            structures=proposed_structures,
            reference_structures=self.gt_structures_missing_initial,
            ltol=self.structure_matcher_ltol,
            stol=self.structure_matcher_stol,
            angle_tol=self.structure_matcher_angle_tol,
        )

        # Convex-hull metrics
        convex_hull_metrics = ConvexHullMetrics.calculate(
            environment=self,
            proposed_entries=self.proposed_entries,
        )

        metrics.update({f"reference_{k}": v for k, v in reference_metrics.items()})
        metrics.update({f"validity_{k}": v for k, v in validity_metrics.items()})
        metrics.update(
            {
                f"diversity_all_composition_{k}": v
                for k, v in composition_metrics_all.items()
            }
        )
        metrics.update(
            {
                f"diversity_all_structure_{k}": v
                for k, v in structure_metrics_all.items()
            }
        )
        metrics.update(
            {
                f"diversity_stable_composition_{k}": v
                for k, v in composition_metrics_stable.items()
            }
        )
        metrics.update(
            {
                f"diversity_stable_structure_{k}": v
                for k, v in structure_metrics_stable.items()
            }
        )
        metrics.update({f"novelty_{k}": v for k, v in novelty_metrics.items()})
        metrics.update({f"convex_hull_{k}": v for k, v in convex_hull_metrics.items()})

        self.metrics_history.append(metrics)

        # Compute discovery curve metrics using the full history (including current step)
        # This includes area under discovery curve, enhancement factor, and acceleration factor
        # Note: baseline_metrics_history can be provided if baseline comparison is needed
        # Use num_newly_discovered_stable as the metric key (y-axis) for discovery curve
        discovery_curve_metrics = DiscoveryCurveMetrics.calculate(
            metrics_history=self.metrics_history,
            metric_key="num_newly_discovered_stable",
        )
        # Add discovery curve metrics to both the current metrics dict and the history entry
        metrics.update(
            {f"discovery_curve_{k}": v for k, v in discovery_curve_metrics.items()}
        )
        # Update the last entry in metrics_history to include discovery curve metrics
        if self.metrics_history:
            self.metrics_history[-1].update(
                {f"discovery_curve_{k}": v for k, v in discovery_curve_metrics.items()}
            )

        return metrics

    def _safe_e_above_hull(self, phase_diagram: PhaseDiagram, entry: PDEntry) -> float:
        """Return e_above_hull with robust handling for elemental duplicates.

        If a valid decomposition is not found (commonly when the proposal is an
        elemental entry conflicting with the current elemental reference), warn
        and fall back to a manual computation for elemental compositions.

        Additionally, if this situation occurs on the observed PD and the new
        elemental entry has a lower energy-per-atom than the existing elemental
        reference, replace the observed reference and rebuild the observed PD
        before returning the e_above_hull.
        """
        comp = entry.composition
        # Special handling: if operating on observed PD and this is an elemental entry
        # check if we need to update the PD before computing e_above_hull
        if len(comp.elements) == 1 and phase_diagram is self.observed_pd:
            elem = list(comp.elements)[0]
            try:
                ref_entry = phase_diagram.el_refs[elem]
                entry_e_pa = entry.energy_per_atom
                ref_e_pa = ref_entry.energy_per_atom
                # If new elemental energy is lower, update the PD
                if entry_e_pa < ref_e_pa - 1e-12:
                    # Remove any existing elemental entries for this element that have higher energy
                    new_entries: list[PDEntry] = []
                    for e in self.observed_entries:
                        if (
                            len(e.composition.elements) == 1
                            and list(e.composition.elements)[0] == elem
                        ):
                            # keep only the better (lower) energy entry; prefer the new 'entry'
                            continue
                        new_entries.append(e)
                    # Keep the new elemental entry
                    new_entries.append(entry)
                    self.observed_entries = new_entries
                    # Rebuild observed PD
                    self.observed_pd = PhaseDiagram(self.observed_entries)
                    # Now use utility function on updated PD
                    return safe_e_above_hull(self.observed_pd, entry)
            except Exception:
                pass  # If update check fails, continue with normal computation

        # Use utility function for safe computation
        return safe_e_above_hull(phase_diagram, entry)

    def _compute_elemental_energies_with_oracle(self) -> None:
        """Compute elemental reference energies using the oracle from dataset structures."""
        # Check if oracle already has elemental energies computed
        # if hasattr(self.oracle, 'element_reference_energies') and self.oracle.element_reference_energies:
        #     print(f"Oracle already has elemental energies: {list(self.oracle.element_reference_energies.keys())}")
        #     return

        # Check if oracle has the method to compute from structures
        if hasattr(self.oracle, "compute_elemental_energies_from_structures"):
            logger.info(
                "Computing elemental energies using oracle from dataset structures..."
            )
            elemental_entries_with_structures = (
                self.dataset.get_elemental_entries_with_structures()
            )  # type: ignore[attr-defined]
            computed_energies = self.oracle.compute_elemental_energies_from_structures(
                elemental_entries_with_structures
            )
            logger.info(f"Computed elemental energies: {computed_energies}")
        else:
            logger.warning(
                "Oracle does not support computing elemental energies from structures"
            )

    def _get_oracle_evaluated_elemental_entries(self) -> list[PDEntry]:
        """Get all elemental entries from ground truth, re-evaluated with the oracle."""
        elemental_entries_with_structures = (
            self.dataset.get_elemental_entries_with_structures()
        )  # type: ignore[attr-defined]
        oracle_evaluated_entries = []
        for entry, structure in elemental_entries_with_structures:
            # Apply SMACT filter if enabled
            if (
                self.filter_by_smact
                and structure is not None
                and not is_smact_valid(structure)
            ):
                continue

            oracle_result = {
                "energy_per_atom": self.oracle.get_element_reference_energy(
                    str(entry.composition.elements[0])
                )
            }
            oracle_entry = structure_result_to_entry(structure, oracle_result)
            oracle_evaluated_entries.append(oracle_entry)
        return oracle_evaluated_entries

    def _get_oracle_evaluated_stable_entries(self) -> list[PDEntry]:
        """Get all stable entries from filtered ground truth PD, re-evaluated with the oracle.

        Note: This uses the ground_truth_stable_structures which have already been
        filtered by SMACT, max_stoichiometry, and stability tolerance.
        """
        oracle_evaluated_entries = []

        logger.info(
            f"Re-evaluating {len(self.ground_truth_stable_structures)} filtered (near-)stable entries with oracle..."
        )

        # Use the already-filtered ground truth stable structures
        for structure in self.ground_truth_stable_structures:
            # skip oracle evaluation for elemental entries
            composition = structure.composition
            if len(composition.elements) == 1:
                oracle_result = {
                    "energy_per_atom": self.oracle.get_element_reference_energy(
                        str(composition.elements[0])
                    )
                }
            else:
                oracle_result = self.oracle.evaluate(structure)

            # Create new PDEntry with oracle energy
            oracle_entry = structure_result_to_entry(structure, oracle_result)
            oracle_evaluated_entries.append(oracle_entry)

        logger.info(f"Created {len(oracle_evaluated_entries)} oracle-evaluated entries")
        return oracle_evaluated_entries

    def _get_initial_structures(self) -> list[Structure]:
        """Get all structures from the initial observed phase diagram."""
        initial_structures = []
        for entry in self.observed_entries:
            structure = extract_structure_from_entry(entry)
            if structure is not None:
                initial_structures.append(structure)
        return initial_structures

    def _check_structural_novelty(self, pd_entry: PDEntry) -> bool:
        """Check if the proposed structure is structurally novel compared to initial structures."""
        structure = extract_structure_from_entry(pd_entry)
        # Check against all initial structures
        for initial_structure in self.initial_structures:
            try:
                if self.structure_matcher.fit(structure, initial_structure):
                    return False  # Not novel, matches an existing structure
            except Exception as e:
                logger.warning(
                    f"StructureMatcher.fit() failed for structures {structure.composition} and {initial_structure.composition}: {e}"
                )
                # Treat as non-matching on error, continue checking

        # Check against previously discovered novel structures
        for discovered_entry in self.newly_discovered_entries:
            discovered_structure = extract_structure_from_entry(discovered_entry)
            if discovered_structure is None:
                continue
            try:
                if self.structure_matcher.fit(structure, discovered_structure):
                    return False  # Not novel, matches a previously discovered structure
            except Exception as e:
                logger.warning(
                    f"StructureMatcher.fit() failed for structures {structure.composition} and {discovered_structure.composition}: {e}"
                )
                # Treat as non-matching on error, continue checking

        # This is a novel structure!
        self.newly_discovered_entries.append(pd_entry)
        return True

    def get_newly_discovered_entries(self) -> list[PDEntry]:
        """Get all PDEntries for newly discovered structures."""
        return list(self.newly_discovered_entries)
