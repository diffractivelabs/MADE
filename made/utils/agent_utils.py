"""Utility functions for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from made.agents.base import Filter, Scorer


def normalize_component_list(
    component: Filter | list[Filter] | Scorer | list[Scorer],
    component_name: str,
    chain_class_name: str,
    chain_module_path: str,
) -> Filter | Scorer:
    """
    Normalize a component that can be a single instance or a list.

    If a list with one item, unwrap it. If a list with multiple items,
    wrap in the appropriate Chain class.

    Args:
        component: Single component or list of components
        component_name: Name for error messages (e.g., "Filter", "Scorer")
        chain_class_name: Name of the Chain class to import
        chain_module_path: Module path to import the Chain class from

    Returns:
        Single component instance (or Chain instance if multiple)

    Examples:
        >>> # Single filter
        >>> normalize_component_list(filter, "Filter", "FilterChain", "made.agents.filters.chain")

        >>> # List of filters
        >>> normalize_component_list([f1, f2], "Filter", "FilterChain", "made.agents.filters.chain")
    """
    if isinstance(component, list):
        if len(component) == 1:
            # Single component in list - unwrap it
            return component[0]
        elif len(component) == 0:
            raise ValueError(f"{component_name} list cannot be empty")
        else:
            # Multiple components - wrap in Chain
            module = __import__(chain_module_path, fromlist=[chain_class_name])
            chain_class = getattr(module, chain_class_name)
            return chain_class(component)
    else:
        return component
