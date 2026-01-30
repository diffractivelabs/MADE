"""
Base environment for experimental design benchmarks
"""

from abc import ABC, abstractmethod
from typing import Any

from made.data.chemical_system import PhaseDiagramDataset
from made.oracles.base import Oracle


class Environment(ABC):
    """
    Base environment for experimental design benchmarks.

    Key components:
    - budget: the maximum number of queries to make
    - dataset: the dataset of structures to query
    - oracle: the oracle to use for querying the structures
    - history: the history of queries and their results
    - done: whether the environment is done
    """

    def __init__(self, dataset: PhaseDiagramDataset, oracle: Oracle, budget: int):
        self.budget = budget
        self.query_count = 0
        self.dataset = dataset
        self.oracle = oracle
        self.history = []
        self.done = False

    def is_done(self) -> bool:
        """Check if budget is exhausted or objective is met."""
        return self.done or self.query_count >= self.budget

    @abstractmethod
    def reset(self):
        """
        Reset the environment
        """

    @abstractmethod
    def step(self, query: Any) -> Any:
        """
        Step the environment
        """

    @abstractmethod
    def get_state(self):
        """
        Get the state of the environment
        """
