"""
Baseline experiment runner script.

Runs experiments for multiple agent configs across multiple datasets.
Supports running experiments sequentially or in parallel, with error handling.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import dotenv

logger = logging.getLogger(__name__)


def save_progress(
    output_dir: Path,
    completed_systems: list[str],
    total_systems: int,
    status: str = "in_progress",
) -> None:
    """Save progress tracking file."""
    progress_file = output_dir / "progress.json"
    progress = {
        "status": status,
        "completed_systems": completed_systems,
        "total_systems": total_systems,
        "progress_percent": len(completed_systems) / total_systems * 100 if total_systems > 0 else 0.0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def get_completed_systems(output_dir: Path) -> list[str]:
    """Get list of systems that have completed (have summary.json)."""
    systems_dir = output_dir / "systems"
    if not systems_dir.exists():
        return []

    completed = []
    for system_dir in systems_dir.iterdir():
        if system_dir.is_dir():
            summary_file = system_dir / "summary" / "summary.json"
            if summary_file.exists():
                completed.append(system_dir.name)
    return completed


def run_single_baseline_experiment(
    agent_config: str,
    systems_file: str,
    max_systems: int = 20,
    budget: int = 50,
    num_episodes: int = 1,
    infra: str = "local",
    output_base_dir: str = "./results/baselines",
    seed: int | None = None,
    max_stoichiometry: int = 20,
    stability_tolerance: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "made-baselines",
    wandb_tags: list[str] | None = None,
    resume: bool = True,
) -> Path:
    """
    Run a single baseline experiment.

    Args:
        agent_config: Name of agent config (e.g., "generative_baseline")
        systems_file: Path to systems JSON file
        max_systems: Number of systems to run (default: 20)
        budget: Queries per system (default: 50)
        num_episodes: Episodes per system (default: 1)
        infra: "local" or "modal"
        output_base_dir: Base directory for results
        seed: Random seed for reproducibility
        use_wandb: Whether to use wandb logging
        wandb_project: Wandb project name
        wandb_tags: List of wandb tags

    Returns:
        Path to output directory
    """
    # Create output directory (timestamp is already in output_base_dir)
    output_dir = Path(output_base_dir) / f"{agent_config}_{Path(systems_file).stem}_{max_systems}systems_{budget}queries_{int(stability_tolerance * 1000)}stabilitymeV"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running experiment: {agent_config}")
    logger.info(f"  Systems file: {systems_file}")
    logger.info(f"  Max systems: {max_systems}")
    logger.info(f"  Budget: {budget}")
    logger.info(f"  Stability tolerance: {stability_tolerance}")
    logger.info(f"  Output dir: {output_dir}")

    # Create a temporary config file for this experiment
    # We'll use Hydra's programmatic API to override config
    config_overrides = [
        f"agent={agent_config}",
        f"experiment.systems_file={systems_file}",
        f"experiment.max_systems={max_systems}",
        f"environment.budget={budget}",
        f"experiment.num_episodes={num_episodes}",
        f"experiment.infra={infra}",
        f"experiment.output_dir={output_dir}",
        f"logger.use_wandb={use_wandb}",
        f"logger.wandb_project={wandb_project}",
        # stability tolerance and max stoichiometry overrides
        f"environment.stability_tolerance={stability_tolerance}",
        f"++agent.max_stoichiometry={max_stoichiometry}",
        f"++agent.planner.max_stoichiometry={max_stoichiometry}",
        f"environment.max_stoichiometry={max_stoichiometry}",
    ]

    if seed is not None:
        config_overrides.append(f"experiment.seed={seed}")

    if wandb_tags:
        tags_str = ",".join(wandb_tags)
        config_overrides.append(f"logger.wandb_tags=[{tags_str}]")

    # Save experiment metadata
    metadata = {
        "agent_config": agent_config,
        "systems_file": str(systems_file),
        "max_systems": max_systems,
        "budget": budget,
        "num_episodes": num_episodes,
        "infra": infra,
        "seed": seed,
        "max_stoichiometry": max_stoichiometry,
        "config_overrides": config_overrides,
    }
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Check for existing progress
    completed_systems = []
    if resume:
        completed_systems = get_completed_systems(output_dir)
        if completed_systems:
            logger.info(f"Found {len(completed_systems)} completed systems, will resume from existing results")

    # Initialize progress tracking
    save_progress(output_dir, completed_systems, max_systems, status="starting")

    # Run the experiment using run_multi_systems via subprocess
    # Since run_multi_systems is a Hydra main, we call it via command line
    # Note: run_multi_systems already saves results incrementally per system
    try:
        script_path = Path(__file__).parent / "run_multi_systems.py"
        cmd = [
            sys.executable,
            str(script_path),
        ] + config_overrides

        logger.info(f"Running command: {' '.join(cmd)}")

        # Start subprocess and monitor progress
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Monitor output and update progress
        last_completed_count = len(completed_systems)
        last_progress_update = datetime.now()
        for line in process.stdout:
            print(line, end='')  # Print to console
            # Check for completed systems periodically (every 30 seconds or on system completion)
            current_time = datetime.now()
            time_since_update = (current_time - last_progress_update).total_seconds()
            if "Running system" in line or time_since_update > 30:
                current_completed = get_completed_systems(output_dir)
                if len(current_completed) > last_completed_count or time_since_update > 30:
                    last_completed_count = len(current_completed)
                    last_progress_update = current_time
                    save_progress(output_dir, current_completed, max_systems, status="in_progress")
                    logger.info(f"Progress: {len(current_completed)}/{max_systems} systems completed")

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        # Final progress update
        final_completed = get_completed_systems(output_dir)
        save_progress(output_dir, final_completed, max_systems, status="completed")
        logger.info(f"Experiment completed: {len(final_completed)}/{max_systems} systems finished")

    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        logger.warning("Experiment interrupted by user")
        current_completed = get_completed_systems(output_dir)
        save_progress(output_dir, current_completed, max_systems, status="interrupted")
        logger.info(f"Saved progress: {len(current_completed)}/{max_systems} systems completed")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running experiment {agent_config}: {e}", exc_info=True)
        # Save current progress even on error
        current_completed = get_completed_systems(output_dir)
        save_progress(output_dir, current_completed, max_systems, status="failed")
        # Save error info
        with open(output_dir / "error.log", "w") as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        logger.info(f"Saved partial results: {len(current_completed)}/{max_systems} systems completed")
        raise
    except Exception as e:
        logger.error(f"Error running experiment {agent_config}: {e}", exc_info=True)
        # Save current progress even on error
        current_completed = get_completed_systems(output_dir)
        save_progress(output_dir, current_completed, max_systems, status="failed")
        # Save error info
        with open(output_dir / "error.log", "w") as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        logger.info(f"Saved partial results: {len(current_completed)}/{max_systems} systems completed")
        raise

    logger.info(f"Completed experiment: {agent_config}")
    return output_dir


def run_multiple_baseline_experiments(
    agent_configs: list[str],
    systems_files: list[str] | None = None,
    max_systems: int = 10,
    budget: int = 50,
    num_episodes: int = 5,
    infra: str = "local",
    output_base_dir: str = "./results/baselines",
    seed: int | None = None,
    max_stoichiometry: int = 20,
    stability_tolerance: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "made-baselines",
    wandb_tags: list[str] | None = None,
    continue_on_error: bool = True,
    resume: bool = True,
) -> dict[str, Path | None]:
    """
    Run multiple baseline experiments.

    Args:
        agent_configs: List of agent config names
        systems_files: List of systems files (if None, uses default from config)
        max_systems: Number of systems per experiment
        budget: Queries per system
        num_episodes: Episodes per system
        infra: "local" or "modal"
        output_base_dir: Base directory for results
        seed: Random seed
        use_wandb: Whether to use wandb
        wandb_project: Wandb project name
        wandb_tags: List of wandb tags
        continue_on_error: If True, continue with remaining experiments on error

    Returns:
        Dictionary mapping agent_config -> output_dir (or None if failed)
    """
    results: dict[str, Path | None] = {}

    for agent_config in agent_configs:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment: {agent_config}")
        logger.info(f"{'='*80}")

        # Use first systems_file if provided, otherwise None (will use config default)
        systems_file = systems_files[0] if systems_files else None

        try:
            output_dir = run_single_baseline_experiment(
                agent_config=agent_config,
                systems_file=systems_file or "",  # Will use config default if empty
                max_systems=max_systems,
                budget=budget,
                num_episodes=num_episodes,
                infra=infra,
                output_base_dir=output_base_dir,  # Use provided directory (may already be timestamped)
                seed=seed,
                max_stoichiometry=max_stoichiometry,
                stability_tolerance=stability_tolerance,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_tags=wandb_tags,
                resume=resume,
            )
            results[agent_config] = output_dir
        except Exception as e:
            logger.error(f"Failed to run experiment {agent_config}: {e}")
            results[agent_config] = None
            if not continue_on_error:
                raise

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for materials discovery"
    )
    parser.add_argument(
        "--agent-configs",
        nargs="+",
        required=True,
        help="List of agent config names (e.g., generative_baseline mlip_baseline)",
    )
    parser.add_argument(
        "--systems-file",
        type=str,
        help="Path to systems JSON file (default: from config)",
    )
    parser.add_argument(
        "--max-systems",
        type=int,
        default=20,
        help="Number of systems to run (default: 20)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=50,
        help="Queries per system (default: 50)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Episodes per system (default: 1)",
    )
    parser.add_argument(
        "--infra",
        type=str,
        default="local",
        choices=["local", "modal"],
        help="Infrastructure to use (default: local)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/baselines",
        help="Base output directory (default: ./results/baselines)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-stoichiometry",
        type=int,
        default=20,
        help="Maximum stoichiometry for planner (default: 20)",
    )
    parser.add_argument(
        "--stability-tolerance",
        type=float,
        default=0.1,
        help="Stability tolerance in eV (default: 0.1)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="made-baselines",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=None,
        help="Wandb tags",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop on first error instead of continuing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing results if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from existing results",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dotenv.load_dotenv()

    # Create timestamped base directory for this run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamped_output_dir = f"{args.output_dir}/{timestamp}"
    Path(timestamped_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created timestamped output directory: {timestamped_output_dir}")

    # Run experiments with timestamped directory
    results = run_multiple_baseline_experiments(
        agent_configs=args.agent_configs,
        systems_files=[args.systems_file] if args.systems_file else None,
        max_systems=args.max_systems,
        budget=args.budget,
        num_episodes=args.num_episodes,
        infra=args.infra,
        output_base_dir=timestamped_output_dir,  # Use timestamped directory
        seed=args.seed,
        max_stoichiometry=args.max_stoichiometry,
        stability_tolerance=args.stability_tolerance,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        continue_on_error=not args.stop_on_error,
        resume=args.resume,
    )

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Experiment Summary")
    logger.info("="*80)
    for agent_config, output_dir in results.items():
        status = "SUCCESS" if output_dir else "FAILED"
        logger.info(f"  {agent_config}: {status}")
        if output_dir:
            logger.info(f"    Output: {output_dir}")
            # Check progress if available
            progress_file = output_dir / "progress.json"
            if progress_file.exists():
                with open(progress_file) as f:
                    progress = json.load(f)
                    logger.info(f"    Progress: {progress.get('status', 'unknown')} - {len(progress.get('completed_systems', []))}/{progress.get('total_systems', 0)} systems")

    # Save summary
    summary_file = Path(timestamped_output_dir) / "experiments_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "results": {
            k: str(v) if v else None for k, v in results.items()
        },
        "args": vars(args),
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
