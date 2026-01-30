"""
Base class for all agents
Separated composition and structure prediction.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, overload

from pydantic import BaseModel, Field
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from made.utils.agent_utils import normalize_component_list

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Result of applying a filter to a single structure."""

    passed: bool = Field(description="Whether the structure passed this filter")
    filter_name: str = Field(description="Name of the filter")
    rejection_reason: str | None = Field(
        default=None, description="Reason for rejection"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )

    class Config:
        arbitrary_types_allowed = True


class ScoreResult(BaseModel):
    """Result of scoring a single structure."""

    score: float = Field(description="The score (higher is better)")
    scorer_name: str = Field(description="Name of the scorer")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional scoring details"
    )

    class Config:
        arbitrary_types_allowed = True


class Plan(BaseModel):
    compositions: list[Composition] = Field(
        default_factory=list,
        description="List of compositions to generate structures for.",
    )
    num_candidates: int = 1  # number of structures to generate for each composition
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Constraints on the structures to generate."
    )


class Planner(ABC):
    @abstractmethod
    def propose(
        self, state: dict[str, Any], previous: dict[str, Any] | None = None
    ) -> Plan:
        """Produce a Plan given the current environment state and optional prior loop context."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return the current state of the planner."""

    @abstractmethod
    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the planner."""


class Generator(ABC):
    @abstractmethod
    def generate(self, plan: Plan, state: dict[str, Any]) -> list[Structure]:
        """Generate candidate structures from the provided plan and environment state."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return the current state of the generator."""

    @abstractmethod
    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the generator."""


class Scorer(ABC):
    """Base class for structure scorers.

    Scorers evaluate candidate structures and can optionally provide
    detailed score information when return_results=True.
    """

    # Override this in subclasses to set a human-readable scorer name
    scorer_name: str = "Scorer"

    @overload
    def score_candidates(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = False,
    ) -> list[float]: ...

    @overload
    def score_candidates(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = True,
    ) -> tuple[list[float], list[ScoreResult]]: ...

    def score_candidates(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = False,
    ) -> list[float] | tuple[list[float], list[ScoreResult]]:
        """
        Score all candidates and return scores in the same order.

        All scores should follow the convention that higher values are better.
        For metrics where lower is better (e.g., energy), negate the values.

        Args:
            candidates: List of candidate structures to score
            state: Current environment state
            return_results: If True, also return ScoreResult for each candidate
                with detailed scoring information.

        Returns:
            If return_results=False: List of scores (higher = better).
            If return_results=True: Tuple of (scores, results_per_candidate).
        """
        scores, results = self._score_with_results(candidates, state)
        if return_results:
            return scores, results
        return scores

    @abstractmethod
    def _score_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[float], list[ScoreResult]]:
        """Internal method that does the actual scoring with results.

        Subclasses must implement this to provide detailed score information.
        """

    @overload
    def select(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = False,
    ) -> Structure: ...

    @overload
    def select(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = True,
    ) -> tuple[Structure, list[ScoreResult]]: ...

    def select(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = False,
    ) -> Structure | tuple[Structure, list[ScoreResult]]:
        """
        Choose a winning structure from candidates.

        Default implementation scores all candidates and selects the best.
        All scorers should return scores where higher values are better.
        State change detection and reranking (if needed) should be handled
        internally by score_candidates.

        Args:
            candidates: List of candidate structures to select from.
            state: Current environment state.
            return_results: If True, also return ScoreResult for each candidate.

        Returns:
            If return_results=False: The selected Structure.
            If return_results=True: Tuple of (selected_structure, results_per_candidate).
        """
        if not candidates:
            raise ValueError("No candidates provided")

        logger.info(
            f"Scorer.select(): Evaluating {len(candidates)} candidates for selection"
        )

        # Score candidates (handles state changes internally if needed)
        scores, results = self.score_candidates(candidates, state, return_results=True)

        # Explicitly validate that scores length matches candidates length
        if len(scores) != len(candidates):
            raise ValueError(
                f"score_candidates returned {len(scores)} scores for {len(candidates)} candidates. "
                f"Scores and candidates must have the same length."
            )

        # Compute score statistics
        max_score = max(scores)
        min_score = min(scores)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Select best candidate
        best_index = int(max(range(len(scores)), key=lambda i: scores[i]))
        selected = candidates[best_index]

        logger.info(
            f"Scorer.select(): Selected candidate {best_index}/{len(candidates) - 1} "
            f"(composition: {selected.composition.formula}, "
            f"score: {max_score:.6f}, "
            f"score range: [{min_score:.6f}, {max_score:.6f}], "
            f"mean: {mean_score:.6f})"
        )

        if return_results:
            return selected, results
        return selected

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return the current state of the scorer."""

    @abstractmethod
    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the scorer."""


class Filter(ABC):
    """Base class for structure filters.

    Filters validate candidate structures and can optionally provide
    detailed rejection reasons when return_results=True.
    """

    # Override this in subclasses to set a human-readable filter name
    filter_name: str = "Filter"

    @overload
    def filter(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = False,
    ) -> list[Structure]: ...

    @overload
    def filter(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = True,
    ) -> tuple[list[Structure], list[FilterResult]]: ...

    def filter(
        self,
        candidates: list[Structure],
        state: dict[str, Any],
        return_results: bool = False,
    ) -> list[Structure] | tuple[list[Structure], list[FilterResult]]:
        """Filter candidate structures based on validity criteria.

        Args:
            candidates: List of candidate structures to filter.
            state: Current environment state.
            return_results: If True, also return FilterResult for each candidate
                with detailed rejection reasons.

        Returns:
            If return_results=False: Filtered list of structures.
            If return_results=True: Tuple of (filtered_structures, results_per_candidate).
        """
        passed, results = self._filter_with_results(candidates, state)
        if return_results:
            return passed, results
        return passed

    @abstractmethod
    def _filter_with_results(
        self, candidates: list[Structure], state: dict[str, Any]
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Internal method that does the actual filtering with results.

        Subclasses must implement this to provide detailed rejection reasons.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return the current state of the filter."""

    @abstractmethod
    def update_state(self, state: dict[str, Any]) -> None:
        """Update the state of the filter."""


class Agent(ABC):
    """
    Base class for agents.

    This is a minimal abstract base class that defines the interface for all agents.
    Agents must implement propose_composition_and_structure() to return a structure
    for oracle evaluation.

    For agents using the planner → generator → filter → scorer workflow,
    see WorkflowAgent.
    """

    def __init__(self):
        """Initialize base agent state."""
        self.last_step = 0  # track the last step of the environment the agent has seen

    def _update_last_step(self, state: dict[str, Any]) -> str:
        """
        Update last_step based on environment state.

        Args:
            state: Environment state containing 'query_count' and optionally 'last_observation'

        Returns:
            Action to take: 'init' (no observation yet), 'update' (new step), 'skip' (already processed)

        Raises:
            ValueError: If step is out of sequence
        """
        # Check if there's an observation to process
        if not state.get("last_observation"):
            # No observation yet - first call, initialize
            return "init"

        # query_count is at top level of state, not inside last_observation
        env_last_step = state.get("query_count", 0)

        if env_last_step == self.last_step + 1:
            # New step - update
            self.last_step = env_last_step
            return "update"
        elif env_last_step == self.last_step:
            # Already processed this step
            logger.debug(f"Step {env_last_step} already processed, skipping update")
            return "skip"
        else:
            raise ValueError(
                f"Last observation step {env_last_step} does not match expected step {self.last_step + 1}"
            )

    @abstractmethod
    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Composition, Structure]:
        """Propose a composition and structure given the environment state."""

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get full agent state for checkpointing."""

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Load full agent state from checkpoint."""

    @abstractmethod
    def update_state(self, state: dict[str, Any]) -> None:
        """Update the agent state based on environment feedback."""

    def step(self, state: dict[str, Any]) -> tuple[Composition, Structure]:
        """Step the agent given the environment state."""
        # update the agent state given the environment state (e.g. last observation)
        self.update_state(state)
        # propose a composition and structure
        return self.propose_composition_and_structure(state)

    def __call__(self, state: dict[str, Any]) -> tuple[Composition, Structure]:
        """Call the agent given the environment state."""
        return self.step(state)


class WorkflowAgent(Agent):
    """
    Agent using the planner → generator → filter → scorer workflow.

    This is the standard workflow for agents that:
    1. Plan which compositions to explore (Planner)
    2. Generate candidate structures (Generator)
    3. Filter invalid candidates (Filter)
    4. Score and select the best candidate (Scorer)
    """

    def __init__(
        self,
        planner: Planner,
        generator: Generator,
        scorer: Scorer | list[Scorer],
        filter: Filter | list[Filter],
        **kwargs: Any,
    ):
        super().__init__()
        self.planner = planner
        self.generator = generator
        self.filter = normalize_component_list(
            filter, "Filter", "FilterChain", "made.agents.filters.chain"
        )
        self.scorer = normalize_component_list(
            scorer, "Scorer", "ScorerChain", "made.agents.scorers.chain"
        )

    def get_state(self) -> dict[str, Any]:
        """Get full agent state including all caches."""
        state = {
            "planner": self.planner.get_state(),
            "generator": self.generator.get_state(),
            "filter": self.filter.get_state(),
            "scorer": self.scorer.get_state(),
            "last_step": self.last_step,
        }
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Load full agent state including all caches."""
        if "planner" in state and hasattr(self.planner, "load_state"):
            self.planner.load_state(state["planner"])
        if "generator" in state and hasattr(self.generator, "load_state"):
            self.generator.load_state(state["generator"])
        if "filter" in state and hasattr(self.filter, "load_state"):
            self.filter.load_state(state["filter"])
        if "scorer" in state and hasattr(self.scorer, "load_state"):
            self.scorer.load_state(state["scorer"])
        if "last_step" in state:
            self.last_step = state["last_step"]

    def update_state(self, state: dict[str, Any]) -> None:
        """Update the agent state."""
        action = self._update_last_step(state)
        if action == "skip":
            return

        # Update all components (on init or new step)
        self.planner.update_state(state)
        self.generator.update_state(state)
        self.filter.update_state(state)
        self.scorer.update_state(state)

    @abstractmethod
    def propose_composition_and_structure(
        self, state: dict[str, Any]
    ) -> tuple[Composition, Structure]:
        """Propose a composition and structure given the environment state."""
