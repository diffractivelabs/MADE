"""
LLM planner.
"""

import logging
from typing import Any

import dspy
from pymatgen.core.composition import Composition

from ...utils.llm import summarize_context_for_llm
from ..base import Plan, Planner

logger = logging.getLogger(__name__)


class CompositionPlannerSignature(dspy.Signature):
    """You are a planner for a material discovery experiment. Your goal is to discover as many NOVEL, UNIQUE, STABLE (or metastable) structures as possible.

    CRITICAL CONSTRAINT: You MUST ONLY use elements from the provided 'elements' list in your compositions. And you MUST ONLY propose compositions that are within the max_stoichiometry.
    - Example: If elements=['Li', 'O'], you can propose Li2O, LiO2, etc., but NOT Na2O, Fe2O3, etc.
    - Example: If max_stoichiometry=20, you can propose Li2O, LiO2, etc., but NOT Li19O19.

    DEFINITIONS:
    - STABLE/METASTABLE: Structures with e_above_hull <= stability_tolerance.
    - NOVEL: Not already known on the convex hull (is_newly_discovered=True).
    - UNIQUE: Structurally distinct from previously evaluated structures.
    - Entries marked is_stable_or_metastable=True with is_newly_discovered=True are successful discoveries.

    PHASE DIAGRAM CONCEPTS:
    - Reduced formulas (e.g., Li2O) represent UNIQUE POINTS on the phase diagram.
    - Different reduced compositions = different phase diagram points → PRIORITIZE DIVERSE reduced compositions for broad coverage.

    STRUCTURE DIVERSITY AT SAME COMPOSITION:
    - Multiple DIFFERENT structures can exist at the SAME reduced composition (same phase diagram point)
    - Different structures for the SAME composition can have DIFFERENT stabilities (one unstable doesn't mean all are!)
    - Different unit cell sizes (Li2O vs Li4O2) create different structures but occupy the SAME phase diagram point

    STRATEGY:
    - Explore diverse reduced compositions (different phase diagram points) to maximize phase diagram coverage.
    - If a reduced composition has yielded stable/metastable NOVEL structures, consider proposing MORE unit cell sizes for it as additional stable polymorphs may exist.
    - Compositions with only [unstable] entries may still have stable structures.
    - Balance exploration (new reduced compositions) vs exploitation (trying to find stable structures for compositions with only [unstable] entries)

    Propose FULL formulas with specific unit cell sizes (e.g., Li2O, Li4O2, LiO2) not just reduced formulas.
    """

    context: dict[str, Any] = dspy.InputField(
        desc=(
            "State of the current experiment, including known stable/metastable structures (energies, e_above_hull), "
            "stability threshold, composition trial counts, recent trials, etc."
        )
    )
    elements: list[str] = dspy.InputField(
        desc="List of elements in the chemical system. You MUST ONLY use these elements in your proposed compositions. Any composition containing elements not in this list will be rejected."
    )
    max_stoichiometry: int = dspy.InputField(
        desc="Maximum stoichiometry to consider. e.g. 4 means consider compositions up to 4 atoms."
    )
    num_compositions: int = dspy.InputField(
        desc="Number of compositions to return. e.g. 3 means return a list of 3 compositions."
    )
    stability_tolerance: float = dspy.InputField(
        desc="e_above_hull threshold for stability. Structures with e_above_hull <= this are stable/metastable."
    )

    compositions: list[str] = dspy.OutputField(
        desc="List of compositions to explore next. e.g. ['Na2O', 'NaO3']. These should be unique from each other."
    )


class LLMPlanner(Planner):
    def __init__(
        self,
        llm_config: dict[str, Any],
        context_config: dict[str, Any],
        max_stoichiometry: int = 12,
        num_compositions: int = 1,
        num_candidates: int = 8,
        max_context_entries: int | None = None,
    ) -> None:
        """Initialize LLM Planner.

        Args:
            llm_config: LLM configuration (model, temperature, etc.)
            context_config: Context configuration (include_structure_info, etc.)
            max_stoichiometry: Maximum atoms per unit cell
            num_compositions: Number of compositions to propose per call
            num_candidates: Number of candidates to generate per composition
            max_context_entries: Maximum context entries to show LLM. If exceeded,
                randomly samples while prioritizing stable/metastable entries.
        """
        self.llm_config = llm_config
        self.context_config = context_config
        self.max_stoichiometry = max_stoichiometry
        self.num_compositions = num_compositions
        self.num_candidates = num_candidates
        self.max_context_entries = max_context_entries
        signature = CompositionPlannerSignature
        # override the signature to include the prompt in the config if provided
        if self.context_config.objective_prompt:
            signature.instructions = self.context_config.objective_prompt
        self.planner = dspy.ChainOfThought(signature)
        self.history = []

    def propose(
        self, state: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> Plan:
        with dspy.settings.context(
            lm=dspy.LM(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_output_tokens,
                temperature=self.llm_config.temperature,
                cache=self.llm_config.cache,
            )
        ):
            stability_tolerance = state.get("stability_tolerance", 1e-8)
            context = summarize_context_for_llm(
                state,
                stability_tolerance=stability_tolerance,
                include_structures=self.context_config.include_structure_info,
                include_composition_counter=self.context_config.include_composition_counter,
                include_recent_trial=self.context_config.include_recent_trial,
                max_entries=self.max_context_entries,
            )
            pred = self.planner(
                context=context,
                elements=state.get("elements", []),
                max_stoichiometry=self.max_stoichiometry,
                num_compositions=self.num_compositions,
                stability_tolerance=stability_tolerance,
            )
            self.history.append(pred.toDict())
            logger.info(
                f"LLMPlanner proposed compositions: {pred.compositions}, reasoning: {pred.reasoning}"
            )

        # Parse compositions and validate against constraints
        allowed_elements = set(state.get("elements", []))
        valid_compositions = []
        for comp_str in pred.compositions:
            try:
                comp = Composition(comp_str)

                # Check that all elements are in the allowed chemical system
                comp_elements = {str(el) for el in comp.elements}
                invalid_elements = comp_elements - allowed_elements
                if invalid_elements:
                    logger.warning(
                        f"LLMPlanner: Skipping composition {comp_str} with invalid elements {invalid_elements}. "
                        f"Allowed elements: {allowed_elements}"
                    )
                    continue

                # Check that total number of atoms is at most max_stoichiometry
                total_atoms = sum(comp.values())
                if total_atoms > self.max_stoichiometry:
                    logger.warning(
                        f"LLMPlanner: Skipping composition {comp_str} with {total_atoms} atoms "
                        f"(exceeds max_stoichiometry={self.max_stoichiometry})"
                    )
                    continue

                valid_compositions.append(comp)
            except Exception as e:
                logger.warning(
                    f"LLMPlanner: Failed to parse composition {comp_str}: {e}"
                )
                continue

        if not valid_compositions:
            logger.error(
                f"LLMPlanner: No valid compositions after filtering. "
                f"Original proposals: {pred.compositions}"
                f"defaulting to equal composition plan"
            )
            # Fallback: return default equal composition plan
            return Plan(
                compositions=[Composition(dict.fromkeys(allowed_elements, 1))],
                num_candidates=self.num_candidates,
            )

        return Plan(
            compositions=valid_compositions,
            num_candidates=self.num_candidates,
        )

    def get_state(self) -> dict[str, Any]:
        """Get full state including history for checkpointing."""
        return {"history": self.history}  # Full history

    def load_state(self, state: dict[str, Any]) -> None:
        """Load full state including history from checkpoint."""
        if "history" in state:
            self.history = state["history"]

    def update_state(self, state: dict[str, Any]) -> None:
        pass
