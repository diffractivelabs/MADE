"""
Base class for all oracles
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pymatgen.core.structure import Structure

logger = logging.getLogger(__name__)


class Oracle(ABC):
    """
    Base class for all oracles
    """

    def __init__(self, num_workers: int = 1) -> None:
        """
        Initialize oracle.

        Args:
            num_workers: Number of worker threads for parallel evaluation (default: 1).
                If > 1, structures will be split into batches and processed in parallel.
        """
        self.num_workers = num_workers

    @abstractmethod
    def evaluate(self, structure: Structure) -> dict[str, Any]:
        """
        Evaluate the structure and return a dict with evaluation results.
        """

    def batch_evaluate(self, structures: list[Structure]) -> list[dict[str, Any]]:
        """
        Evaluate multiple structures in batch using threading.

        Structures are automatically split into batches based on `num_workers`.
        Results are returned in the same order as input structures.

        Args:
            structures: List of structures to evaluate

        Returns:
            List of evaluation results, one per structure (same order)
        """
        if not structures:
            return []

        # If single worker, process sequentially
        if self.num_workers == 1:
            return [self.evaluate(structure) for structure in structures]

        # Calculate number of batches based on num_workers
        # Split structures as evenly as possible across num_workers
        num_structures = len(structures)
        num_batches = min(self.num_workers, num_structures)
        batch_size = (
            num_structures + num_batches - 1
        ) // num_batches  # Ceiling division

        # Split structures into batches
        batches = []
        for i in range(0, num_structures, batch_size):
            batch = structures[i : i + batch_size]
            batches.append((i, batch))

        results = [None] * num_structures

        # Process batches in parallel using ThreadPoolExecutor
        def evaluate_batch(batch: list[Structure]) -> list[dict[str, Any]]:
            """Helper function to evaluate a batch of structures."""
            return [self.evaluate(structure) for structure in batch]

        logger.info(
            f"Evaluating {num_structures} structures in {num_batches} batches with {self.num_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batch evaluation tasks
            future_to_batch = {
                executor.submit(evaluate_batch, batch): (start_idx, batch)
                for start_idx, batch in batches
            }

            # Collect results as they complete (maintaining order)
            for future in as_completed(future_to_batch):
                start_idx, batch = future_to_batch[future]
                batch_results = future.result()
                # Store results at correct indices
                for j, result in enumerate(batch_results):
                    results[start_idx + j] = result

        return results
