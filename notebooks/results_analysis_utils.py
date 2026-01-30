"""
Results Analysis Utilities for MADE

This module provides functions for loading, processing, and visualizing
experimental results from MADE benchmark runs.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from made.evaluation.metrics import DiscoveryCurveMetrics

# Set default matplotlib parameters
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.markersize": 6,
    "lines.linewidth": 1.5,
})


# ============================================================================
# Label Dictionaries
# ============================================================================

STRATEGY_LABELS = {
    "random_generator_baseline": "Random",
    "random_generator_baseline_no_smact": "Random (no SMACT)",
    "random_generator_diversity_planner": "Random + Diversity",
    "random_generator_mlip_ranking": "Random + MLIP",
    "random_generator_ucb_planner": "Random + UCB",
    "chemeleon_generative_baseline": "Chemeleon",
    "chemeleon_diversity_planner": "Chemeleon + Diversity",
    "chemeleon_mlip_ranking": "Chemeleon + MLIP",
    "chemeleon_mlip_ranking_chain_filter": "Chemeleon + MLIP + Filter",
    "llm_react_orchestrator": "LLM Orchestrator",
}

METRIC_LABELS = {
    "acceleration_factor": "Acceleration Factor (AF)",
    "enhancement_factor": "Enhancement Factor (EF)",
    "area_under_discovery_curve": "Area Under Discovery Curve (AUDC)",
    "area_under_discovery_curve_normalized": "AUDC (Normalized)",
    "novelty_stable_unique_novel_fraction": "mSUN Fraction",
    "novelty_stable_unique_novel_count": "mSUN Count",
    "num_newly_discovered_stable": "Stable Discoveries",
    "recall_formula": "Formula Recall",
    "precision_formula": "Formula Precision",
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_episode_trajectory(trajectory_file: Path) -> dict[str, Any] | None:
    """Load an episode trajectory JSON file."""
    try:
        with open(trajectory_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {trajectory_file}: {e}")
        return None


def extract_metrics_history(trajectory: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Extract per-step metrics history from a trajectory.

    Returns a list of dicts, one per query, containing metrics like:
    - queries_used: number of oracle queries so far
    - num_newly_discovered_stable: cumulative stable discoveries
    """
    # Try to get metrics_history from final_env_state (preferred)
    if "final_env_state" in trajectory:
        final_state = trajectory["final_env_state"]
        if isinstance(final_state, dict) and "metrics_history" in final_state:
            metrics_history = final_state["metrics_history"]
            if isinstance(metrics_history, list) and len(metrics_history) > 0:
                return metrics_history

    # Try top-level metrics_history
    if "metrics_history" in trajectory:
        metrics_history = trajectory["metrics_history"]
        if isinstance(metrics_history, list) and len(metrics_history) > 0:
            return metrics_history

    # Fallback: reconstruct from trajectory observations
    if "trajectory" in trajectory:
        observations = trajectory["trajectory"]
        if not observations or not isinstance(observations, list):
            return None

        metrics_list = []
        num_newly_discovered_stable = 0
        query_count = 0

        for obs in observations:
            if not isinstance(obs, dict):
                continue
            query_count += 1
            if obs.get("is_newly_discovered", False) and obs.get("is_stable", False):
                num_newly_discovered_stable += 1
            metrics_list.append({
                "queries_used": query_count,
                "num_newly_discovered_stable": num_newly_discovered_stable,
            })

        if metrics_list:
            return metrics_list

    return None


def extract_final_metrics(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract final metrics from trajectory JSON.

    Returns a dict of final metrics (keys like 'acceleration_factor', etc.)
    """
    if "metrics" in trajectory and isinstance(trajectory["metrics"], dict):
        final_metrics = {}
        for key, value in trajectory["metrics"].items():
            if isinstance(value, (int, float, str, bool)):
                clean_key = key.replace("final/", "") if key.startswith("final/") else key
                final_metrics[clean_key] = value
        if final_metrics:
            return final_metrics
    return None


def load_single_run_results(
    results_dir: Path,
    verbose: bool = False
) -> tuple[list[list[dict]], list[dict]]:
    """
    Load results from a single benchmark run (run_benchmark.py output).

    Args:
        results_dir: Path to results directory (e.g., results/20260125-224513-orb-random/)
        verbose: Print debug information

    Returns:
        Tuple of:
        - List of metrics_history lists (one per episode)
        - List of final_metrics dicts (one per episode)
    """
    trajectories_dir = results_dir / "trajectories"
    if not trajectories_dir.exists():
        if verbose:
            print(f"Warning: Trajectories directory not found: {trajectories_dir}")
        return [], []

    metrics_histories = []
    final_metrics_list = []

    for traj_file in sorted(trajectories_dir.glob("episode_*.json")):
        trajectory = load_episode_trajectory(traj_file)
        if trajectory is None:
            continue

        metrics_history = extract_metrics_history(trajectory)
        if metrics_history:
            metrics_histories.append(metrics_history)

        final_metrics = extract_final_metrics(trajectory)
        if final_metrics:
            final_metrics_list.append(final_metrics)

    return metrics_histories, final_metrics_list


def load_baseline_results(
    experiment_dir: Path,
    verbose: bool = False
) -> tuple[dict[str, list[list[dict]]], dict[str, list[dict]]]:
    """
    Load results from a baseline experiment (run_baseline_experiments.py output).

    Args:
        experiment_dir: Path to experiment directory containing systems/ subdirectory
        verbose: Print debug information

    Returns:
        Tuple of:
        - Dict mapping system_id -> list of metrics_history lists (one per episode)
        - Dict mapping system_id -> list of final_metrics dicts (one per episode)
    """
    systems_dir = experiment_dir / "systems"
    if not systems_dir.exists():
        if verbose:
            print(f"Warning: Systems directory not found: {systems_dir}")
        return {}, {}

    all_metrics_histories: dict[str, list[list[dict]]] = {}
    all_final_metrics: dict[str, list[dict]] = {}

    for system_dir in systems_dir.iterdir():
        if not system_dir.is_dir():
            continue

        system_id = system_dir.name
        metrics_histories, final_metrics_list = load_single_run_results(system_dir, verbose)

        if metrics_histories:
            all_metrics_histories[system_id] = metrics_histories
        if final_metrics_list:
            all_final_metrics[system_id] = final_metrics_list

    return all_metrics_histories, all_final_metrics


def load_summary(results_dir: Path) -> dict[str, Any] | None:
    """
    Load summary.json from a results directory.

    Returns the nested 'summary' dict containing metric statistics (mean/std/sem).
    """
    summary_file = results_dir / "summary" / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            # Summary stats are nested under 'summary' key
            return data.get("summary", data)
    return None


def load_baseline_overall_summary(experiment_dir: Path) -> dict[str, Any] | None:
    """
    Load overall_summary/summary.json from a baseline experiment directory.

    Args:
        experiment_dir: Path to experiment directory (e.g., baselines_080126/<agent_config>_...)

    Returns:
        Dict with 'systems' list and 'summary' dict containing aggregated metrics.
    """
    summary_file = experiment_dir / "overall_summary" / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def load_baseline_per_system_summary(experiment_dir: Path) -> pd.DataFrame | None:
    """
    Load overall_summary/per_system_summary.csv from a baseline experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        DataFrame with per-system metrics breakdown.
    """
    csv_file = experiment_dir / "overall_summary" / "per_system_summary.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return None


def load_experiment_metadata(experiment_dir: Path) -> dict[str, Any] | None:
    """
    Load experiment_metadata.json from a baseline experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dict with experiment configuration (agent_config, systems_file, budget, etc.)
    """
    metadata_file = experiment_dir / "experiment_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return None


def load_experiment_progress(experiment_dir: Path) -> dict[str, Any] | None:
    """
    Load progress.json from a baseline experiment directory.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dict with status, completed_systems, total_systems, progress_percent.
    """
    progress_file = experiment_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return None


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_discovery_curve_metrics(
    metrics_histories: list[list[dict]],
    baseline_histories: list[list[dict]] | None = None,
    budget: int | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute discovery curve metrics (AF, EF, AUDC) from metrics histories.

    Args:
        metrics_histories: List of metrics_history lists (one per episode)
        baseline_histories: Optional baseline for AF/EF computation
        budget: Total query budget (auto-detected if None)

    Returns:
        Dict with mean/std for each metric
    """
    if not metrics_histories:
        return {}

    # Extract discovery curves (queries_used, num_discoveries)
    curves = []
    for history in metrics_histories:
        queries = [h.get("queries_used", i+1) for i, h in enumerate(history)]
        discoveries = [h.get("num_newly_discovered_stable", 0) for h in history]
        curves.append((queries, discoveries))

    if budget is None:
        budget = max(max(q) for q, _ in curves)

    # Compute metrics for each episode
    audc_values = []
    for queries, discoveries in curves:
        audc = DiscoveryCurveMetrics.area_under_discovery_curve(
            queries, discoveries, budget
        )
        audc_values.append(audc)

    n = len(audc_values)
    results = {
        "area_under_discovery_curve": {
            "mean": float(np.mean(audc_values)),
            "std": float(np.std(audc_values)),
            "sem": float(np.std(audc_values) / np.sqrt(n)) if n > 0 else 0.0,
        },
        "area_under_discovery_curve_normalized": {
            "mean": float(np.mean(audc_values)) / budget,
            "std": float(np.std(audc_values)) / budget,
            "sem": float(np.std(audc_values) / np.sqrt(n) / budget) if n > 0 else 0.0,
        },
    }

    # Compute AF/EF if baseline provided
    if baseline_histories:
        baseline_curves = []
        for history in baseline_histories:
            queries = [h.get("queries_used", i+1) for i, h in enumerate(history)]
            discoveries = [h.get("num_newly_discovered_stable", 0) for h in history]
            baseline_curves.append((queries, discoveries))

        # Average baseline curve
        baseline_discoveries = np.mean([d for _, d in baseline_curves], axis=0)

        af_values = []
        ef_values = []
        for queries, discoveries in curves:
            af = DiscoveryCurveMetrics.acceleration_factor(
                queries, discoveries, list(range(1, len(baseline_discoveries)+1)),
                list(baseline_discoveries), k=max(discoveries) // 2 or 1
            )
            ef = DiscoveryCurveMetrics.enhancement_factor(
                queries, discoveries, list(range(1, len(baseline_discoveries)+1)),
                list(baseline_discoveries), t=budget // 2
            )
            af_values.append(af)
            ef_values.append(ef)

        n_af = len(af_values)
        results["acceleration_factor"] = {
            "mean": float(np.mean(af_values)),
            "std": float(np.std(af_values)),
            "sem": float(np.std(af_values) / np.sqrt(n_af)) if n_af > 0 else 0.0,
        }
        results["enhancement_factor"] = {
            "mean": float(np.mean(ef_values)),
            "std": float(np.std(ef_values)),
            "sem": float(np.std(ef_values) / np.sqrt(n_af)) if n_af > 0 else 0.0,
        }

    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_discovery_curves(
    metrics_histories: list[list[dict]],
    label: str = "Agent",
    ax: plt.Axes | None = None,
    color: str | None = None,
    show_error: bool = True,
) -> plt.Axes:
    """
    Plot discovery curves with mean and standard error.

    Args:
        metrics_histories: List of metrics_history lists (one per episode)
        label: Legend label
        ax: Matplotlib axes (creates new figure if None)
        color: Line color
        show_error: Show standard error band

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if not metrics_histories:
        return ax

    # Extract curves
    max_len = max(len(h) for h in metrics_histories)
    n_episodes = len(metrics_histories)
    curves = np.full((n_episodes, max_len), np.nan)

    for i, history in enumerate(metrics_histories):
        discoveries = [h.get("num_newly_discovered_stable", 0) for h in history]
        curves[i, :len(discoveries)] = discoveries

    # Compute mean and SEM
    mean_curve = np.nanmean(curves, axis=0)
    sem_curve = np.nanstd(curves, axis=0) / np.sqrt(n_episodes)
    queries = np.arange(1, max_len + 1)

    # Plot
    line, = ax.plot(queries, mean_curve, label=label, color=color)
    if show_error:
        ax.fill_between(
            queries,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            alpha=0.2,
            color=line.get_color(),
        )

    ax.set_xlabel("Oracle Queries")
    ax.set_ylabel("Stable Discoveries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_final_metrics_comparison(
    final_metrics_list: list[dict[str, Any]],
    metrics: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot bar chart comparing final metrics across episodes.

    Args:
        final_metrics_list: List of final_metrics dicts (one per episode)
        metrics: List of metric names to plot (default: common metrics)
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    if not final_metrics_list:
        return ax

    if metrics is None:
        metrics = [
            "novelty_stable_unique_novel_count",
            "discovery_curve_area_under_discovery_curve_normalized",
            "acceleration_factor",
            "enhancement_factor",
        ]

    df = pd.DataFrame(final_metrics_list)
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    means = df[available_metrics].mean()
    sems = df[available_metrics].sem()  # Standard error of the mean

    x = np.arange(len(available_metrics))
    ax.bar(x, means, yerr=sems, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in available_metrics], rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return ax


def results_to_dataframe(
    final_metrics_by_system: dict[str, list[dict]],
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """
    Convert final metrics to a pandas DataFrame for analysis.

    Args:
        final_metrics_by_system: Dict mapping system_id -> list of final_metrics dicts
        strategy_name: Optional strategy name to add as column

    Returns:
        DataFrame with one row per (system, episode)
    """
    rows = []
    for system_id, metrics_list in final_metrics_by_system.items():
        for ep_idx, metrics in enumerate(metrics_list):
            row = {"system": system_id, "episode": ep_idx}
            if strategy_name:
                row["strategy"] = strategy_name
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)
