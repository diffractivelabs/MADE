"""
LLM-based scorer that selects which candidate structure to evaluate next.
"""

import logging
from typing import Any

import dspy
from pymatgen.core.structure import Structure

from ...utils.llm import (
    summarize_candidates_for_llm,
    summarize_context_for_llm,
)
from ..base import Scorer, ScoreResult

logger = logging.getLogger(__name__)


class CandidateScorerSignature(dspy.Signature):
    """Choose the best candidate structure index based on the experiment context.

    OBJECTIVE: Find NOVEL, STABLE (or metastable) structures.
    - Structures with e_above_hull <= stability_tolerance are stable/metastable (SUCCESS!)
    - We want to MAXIMIZE the number of novel stable structures discovered

    KEY INSIGHT: Different structures for the SAME composition can have DIFFERENT stabilities!
    - One unstable structure for a composition does NOT mean all structures for that composition are unstable
    - Multiple different structures can exist as stable/metastable polymorphs at the same composition
    - Don't dismiss a composition just because one structure was unstable - try different structures!

    SELECTION STRATEGY:
    - Prioritize structures likely to be stable/metastable (within stability_tolerance)
    - Consider both: new compositions (exploration) AND different structures for promising compositions (exploitation)
    - Balance diversity across reduced compositions with depth at successful compositions
    """

    context: dict[str, Any] = dspy.InputField(
        desc=(
            "State of the current experiment, including stable/metastable structures (e_above_hull), "
            "stability threshold, recent trials, composition trial counts, etc."
        )
    )
    candidates: list[dict[str, Any]] = dspy.InputField(
        desc=("List of candidate structures with their compositions and properties")
    )
    stability_tolerance: float = dspy.InputField(
        desc="e_above_hull threshold for stability. Structures with e_above_hull <= this are stable/metastable."
    )

    selected_index: int = dspy.OutputField(
        desc="Zero-based index of the chosen candidate to evaluate next."
    )


class LLMScorer(Scorer):
    scorer_name = "LLMScorer"

    def __init__(
        self,
        llm_config: dict[str, Any],
        context_config: dict[str, Any],
        max_candidates: int | None = None,
        max_context_entries: int | None = None,
    ) -> None:
        """Initialize LLM Scorer.

        Args:
            llm_config: LLM configuration (model, temperature, etc.)
            context_config: Context configuration (include_structure_info, etc.)
            max_candidates: Maximum candidates to show LLM. If exceeded, randomly samples.
            max_context_entries: Maximum context entries to show LLM. If exceeded,
                randomly samples while prioritizing stable/metastable entries.
        """
        self.llm_config = llm_config
        self.context_config = context_config
        self.max_candidates = max_candidates
        self.max_context_entries = max_context_entries

        signature = CandidateScorerSignature
        # Allow overriding the tool instructions from config
        if self.context_config.objective_prompt:
            signature.instructions = self.context_config.objective_prompt
        self.selector = dspy.ChainOfThought(signature)
        self.history = []

    # ---- Scorer API ----
    def _score_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[float], list[ScoreResult]]:
        """
        Score candidates using LLM selection.

        Returns scores where the selected candidate has score 1.0 (best) and
        others have 0.0 (worst). Higher scores are better.
        """
        if not candidates:
            raise ValueError("No candidates provided")

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

            # Get summaries with index mapping (for handling random sampling)
            summaries, original_indices = summarize_candidates_for_llm(
                candidates,
                include_structures=self.context_config.include_structure_info,
                max_candidates=self.max_candidates,
            )

            pred = self.selector(
                context=context,
                candidates=summaries,
                stability_tolerance=stability_tolerance,
            )
            self.history.append(pred.toDict())

            # Map LLM's selected index back to original candidate index
            llm_selected_index = pred.selected_index
            if llm_selected_index < 0 or llm_selected_index >= len(summaries):
                logger.warning(
                    f"LLMScorer selected index {llm_selected_index} out of range "
                    f"(summaries has {len(summaries)} items), using first"
                )
                llm_selected_index = 0

            # Map back to original candidate index
            original_selected_index = original_indices[llm_selected_index]

            logger.info(
                f"LLMScorer selected display_index={llm_selected_index} -> "
                f"original_index={original_selected_index}, "
                f"structure: {candidates[original_selected_index]}, "
                f"reasoning: {pred.reasoning}"
            )

            # Return scores: selected has 1.0 (best), others have 0.0 (worst)
            # Higher scores are better to match base class max() selection
            scores = [0.0] * len(candidates)
            scores[original_selected_index] = 1.0

            # Build results with reasoning
            reasoning = getattr(pred, "reasoning", None)
            results = []
            for i, score in enumerate(scores):
                results.append(
                    ScoreResult(
                        score=score,
                        scorer_name=self.scorer_name,
                        details={
                            "selected": i == original_selected_index,
                            "reasoning": reasoning
                            if i == original_selected_index
                            else None,
                            "was_shown_to_llm": i in original_indices,
                        },
                    )
                )

            return scores, results

    def get_state(self) -> dict[str, Any]:
        """Get full state including history for checkpointing."""
        return {"history": self.history}  # Full history

    def load_state(self, state: dict[str, Any]) -> None:
        """Load full state including history from checkpoint."""
        if "history" in state:
            self.history = state["history"]

    def update_state(self, state: dict[str, Any]) -> None:
        pass
