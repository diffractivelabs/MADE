"""
Workflow Agent - One-shot strategy for materials discovery.

Uses the standard workflow: plan → generate → filter → score → select
"""

import logging
from typing import Any

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from made.agents.base import Plan, WorkflowAgent

logger = logging.getLogger(__name__)


class OneShotWorkflowAgent(WorkflowAgent):
    """
    Workflow agent with one-shot strategy: plan once, generate once, select once.

    This is the standard agent for the planner → generator → filter → scorer workflow.
    """

    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Composition, Structure]:
        logger.info("Proposing compositions")
        plan: Plan = self.planner.propose(state)
        logger.info(f"Proposed {len(plan.compositions)} compositions")
        logger.info(f"Generating candidates for {plan.compositions}")
        candidates = self.generator.generate(plan, state)
        logger.info(f"Generated {len(candidates)} candidates")
        if not candidates:
            raise RuntimeError("No candidate structures generated")

        # Apply filter
        logger.info(f"Filtering {len(candidates)} candidates")
        filtered_candidates = self.filter.filter(candidates, state)
        logger.info(f"Filtered to {len(filtered_candidates)} candidates")
        if not filtered_candidates:
            # Fallback: if all candidates are filtered out, use original candidates
            logger.warning(
                "All candidate structures were filtered out. "
                "Falling back to unfiltered candidates."
            )
            filtered_candidates = candidates

        logger.info(f"Selecting from {len(filtered_candidates)} candidates")
        struct = self.scorer.select(filtered_candidates, state)
        logger.info(f"Selected {struct.composition.formula}")
        logger.info(f"Selected structure: {struct}")
        return struct.composition, struct
