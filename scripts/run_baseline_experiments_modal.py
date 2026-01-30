"""
Baseline experiment runner script for Modal.

Runs experiments for multiple agent configs across multiple datasets on Modal.
This allows you to close your laptop while experiments run.

Run on Modal:
    uv run modal run --detach scripts/run_baseline_experiments_modal.py \\
        --agent-configs "generative_baseline mlip_baseline" \\
        --systems-file ./data/systems_binary.json \\
        --max-systems 20 \\
        --budget 50 \\
        --max-stoichiometry 20

Results are stored in Modal volume 'baseline-results'. Download with:
    modal volume download baseline-results /results/baselines/...
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import dotenv
import modal

logger = logging.getLogger(__name__)

# Create Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",  # needed to install packages from GitHub
        "wget",
        "gnupg",
        "ca-certificates",
    )
    .run_commands(
        # Install Google Chrome for Kaleido (plotly image export)
        "wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update",
        "apt-get install -y google-chrome-stable",
    )
    .pip_install(
        "ase>=3.25.0",
        "ase-ga>=0.2.0",
        "average-minimum-distance>=1.6.0",
        "dspy>=3.0.3",
        "hydra-core>=1.3.1",
        "kaleido>=1.0.0",
        "mace-torch>=0.3.14",
        "modal>=1.1.1",
        "mp-api>=0.45.8",
        "orb-models>=0.5.4",
        "pydantic>=2.0.0",
        "pymatgen>=2025.6.14",
        "scipy>=1.16.1",
        "smact>=3.2.0",
        "wandb>=0.21.1",
        "uv",
    )
    .pip_install(
        "chemeleon-dng @ git+https://github.com/hspark1212/chemeleon-dng.git"
    )
    .run_commands(
        "ln -sf /ckpts/chemeleon/ckpts /root/ckpts"
    )
    .add_local_dir("./data", "/root/data")
    .add_local_dir("./src", "/root/src")
    .add_local_dir("./configs", "/root/configs")
    .add_local_dir("./scripts", "/root/scripts")
    .add_local_python_source("made")
)

# Create Modal app
app = modal.App("baseline-experiments-runner", image=image)

# Create a volume for persistent storage of results
results_volume = modal.Volume.from_name("made-benchmark-results", create_if_missing=True)

# Create a volume for model checkpoints (e.g., Chemeleon)
model_checkpoints_volume = modal.Volume.from_name("matopt-checkpoints", create_if_missing=True)

# Configure retries for handling timeouts and preemptions
retries = modal.Retries(initial_delay=0.0, max_retries=10)


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


def get_remaining_systems(
    systems_file: Path, completed_systems: list[str], max_systems: int | None
) -> list[list[str]]:
    """
    Get list of systems that still need to be evaluated.
    
    Args:
        systems_file: Path to JSON file containing list of element lists
        completed_systems: List of completed system IDs (e.g., ["Co-Nb-Sn", "Li-O"])
        max_systems: Optional maximum number of systems to process
    
    Returns:
        List of remaining systems as element lists (e.g., [["Co", "Nb", "Sn"], ["Li", "O"]])
    """
    # Load all systems from file
    with open(systems_file) as f:
        all_systems = [list(s) for s in json.load(f)]

    # Apply max_systems limit if specified
    if max_systems is not None:
        all_systems = all_systems[:max_systems]

    # Convert completed system IDs to element lists for comparison
    # Completed systems are stored as "Co-Nb-Sn" format
    completed_as_elements = []
    for completed_id in completed_systems:
        completed_as_elements.append(completed_id.split("-"))

    # Find remaining systems
    remaining = []
    for system in all_systems:
        system_id = "-".join(system)
        if system_id not in completed_systems:
            remaining.append(system)

    return remaining


@app.function(
    volumes={"/results": results_volume, "/ckpts": model_checkpoints_volume},
    secrets=[
        modal.Secret.from_name("materials-project-api-key"),
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("openai-api-key"),
    ],
    timeout=60 * 60 * 24,  # 24 hour timeout for long-running experiments
    cpu=2,
    memory=4096,
    max_inputs=1,  # Ensures retries kick off in a fresh container
    retries=retries,  # Handle timeouts and preemptions automatically
)
def run_single_baseline_experiment_modal_wrapper(
    args_tuple: tuple,
) -> dict[str, Any]:
    """Wrapper function to unpack tuple arguments for spawn_map."""
    (
        agent_config,
        systems_file,
        max_systems,
        budget,
        num_episodes,
        infra,
        output_base_dir,
        seed,
        use_wandb,
        wandb_project,
        wandb_tags,
        resume,
        max_stoichiometry,
        stability_tolerance,
    ) = args_tuple
    return run_single_baseline_experiment_modal(
        agent_config=agent_config,
        systems_file=systems_file,
        max_systems=max_systems,
        budget=budget,
        num_episodes=num_episodes,
        infra=infra,
        output_base_dir=output_base_dir,
        seed=seed,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
        resume=resume,
        max_stoichiometry=max_stoichiometry,
        stability_tolerance=stability_tolerance,
    )


def run_single_baseline_experiment_modal(
    agent_config: str,
    systems_file: str,
    max_systems: int,
    budget: int,
    num_episodes: int,
    infra: str,
    output_base_dir: str,
    seed: int | None,
    use_wandb: bool,
    wandb_project: str,
    wandb_tags: list[str] | None,
    resume: bool,
    max_stoichiometry: int = 20,
    stability_tolerance: float = 0.1,
) -> dict[str, Any]:
    """
    Run a single baseline experiment on Modal.
    
    Returns a dict with 'output_dir' (str) and 'success' (bool).
    """
    import sys
    from pathlib import Path

    import hydra

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Mount volume path
    volume_path = Path("/results")
    # output_base_dir already includes timestamp from main()
    output_dir = volume_path / output_base_dir.lstrip("./") / f"{agent_config}_{Path(systems_file).stem}_{max_systems}systems_{budget}queries_{int(stability_tolerance * 1000)}stabilitymeV"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running experiment on Modal: {agent_config}")
    logger.info(f"  Stability tolerance: {stability_tolerance}")
    logger.info(f"  Output dir: {output_dir}")

    # Convert systems_file path to Modal path if it's a local path
    # If it starts with ./data, convert to /root/data
    modal_systems_file = systems_file
    if systems_file.startswith("./data/"):
        modal_systems_file = systems_file.replace("./data/", "/root/data/")
    elif systems_file.startswith("data/"):
        modal_systems_file = f"/root/{systems_file}"
    elif not systems_file.startswith("/"):
        # Relative path, assume it's in data directory
        modal_systems_file = f"/root/data/{systems_file}"

    # Check for existing progress and determine remaining systems
    completed_systems = []
    remaining_systems = None
    if resume:
        completed_systems = get_completed_systems(output_dir)
        if completed_systems:
            logger.info(f"Found {len(completed_systems)} completed systems, will resume")

            # Load remaining systems that need to be evaluated
            systems_file_path = Path(modal_systems_file)
            if systems_file_path.exists():
                remaining_systems = get_remaining_systems(
                    systems_file_path, completed_systems, max_systems
                )
                if remaining_systems:
                    logger.info(f"Will evaluate {len(remaining_systems)} remaining systems")
                else:
                    logger.info("All systems already completed, nothing to do")
                    return {
                        "output_dir": str(output_dir),
                        "success": True,
                        "completed_systems": len(completed_systems),
                        "total_systems": max_systems or len(completed_systems),
                    }
            else:
                logger.warning(f"Systems file not found: {systems_file_path}, cannot resume")

    # Build config overrides list for Hydra compose
    config_overrides = [
        f"agent={agent_config}",
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


    # Handle systems: if resuming with remaining systems, set systems_file=null and systems explicitly
    if remaining_systems is not None:
        # Convert remaining systems to Hydra format: [[Co,Nb,Sn],[Li,O]]
        systems_str = ",".join([f"[{','.join(s)}]" for s in remaining_systems])
        config_overrides.append("experiment.systems_file=null")
        config_overrides.append(f"experiment.systems=[{systems_str}]")
        config_overrides.append(f"experiment.max_systems={len(remaining_systems)}")
        logger.info(f"Resuming with explicit systems: [{systems_str}]")
    else:
        # Normal case: use systems_file
        config_overrides.append(f"experiment.systems_file={modal_systems_file}")
        if max_systems is not None:
            config_overrides.append(f"experiment.max_systems={max_systems}")

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
        "completed_systems": completed_systems,
        "remaining_systems": remaining_systems if remaining_systems else None,
        "config_overrides": config_overrides,
    }
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Initialize progress tracking
    save_progress(output_dir, completed_systems, max_systems, status="starting")

    # Commit volume before starting
    results_volume.commit()

    # Run the experiment by calling run_multi_systems directly with Hydra compose
    try:
        logger.info(f"Running run_multi_systems with config overrides: {config_overrides}")

        # Set up a custom logging handler to track progress
        class ProgressTrackingHandler(logging.Handler):
            def __init__(self, output_dir, max_systems, volume):
                super().__init__()
                self.output_dir = output_dir
                self.max_systems = max_systems
                self.volume = volume
                self.last_completed_count = len(completed_systems)
                self.last_progress_update = datetime.now()

            def emit(self, record):
                msg = self.format(record)
                # Check for completed systems periodically
                current_time = datetime.now()
                time_since_update = (current_time - self.last_progress_update).total_seconds()
                if "Running system" in msg or time_since_update > 30:
                    current_completed = get_completed_systems(self.output_dir)
                    if len(current_completed) > self.last_completed_count or time_since_update > 30:
                        self.last_completed_count = len(current_completed)
                        self.last_progress_update = current_time
                        save_progress(self.output_dir, current_completed, self.max_systems, status="in_progress")
                        self.volume.commit()
                        logger.info(f"Progress: {len(current_completed)}/{self.max_systems} systems completed")

        # Add progress tracking handler
        progress_handler = ProgressTrackingHandler(output_dir, max_systems, results_volume)
        progress_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(progress_handler)

        try:
            # Import run_multi_systems inside the try block to ensure fresh import for each call
            # This is important for parallel execution via spawn_map
            sys.path.insert(0, "/root/scripts")
            from run_multi_systems import run_multi_systems

            # Use Hydra's compose API to create config and call run_multi_systems
            # Keep Hydra context active during the call
            with hydra.initialize(config_path="./configs", version_base=None):
                config = hydra.compose(config_name="config", overrides=config_overrides)
                run_multi_systems(config)
        finally:
            # Remove progress handler
            root_logger.removeHandler(progress_handler)

        # Final progress update
        final_completed = get_completed_systems(output_dir)
        save_progress(output_dir, final_completed, max_systems, status="completed")
        results_volume.commit()
        logger.info(f"Experiment completed: {len(final_completed)}/{max_systems} systems finished")

        return {
            "output_dir": str(output_dir),
            "success": True,
            "completed_systems": len(final_completed),
            "total_systems": max_systems,
        }

    except Exception as e:
        logger.error(f"Error running experiment {agent_config}: {e}", exc_info=True)
        # Save current progress even on error
        current_completed = get_completed_systems(output_dir)
        save_progress(output_dir, current_completed, max_systems, status="failed")
        # Save error info with full traceback
        with open(output_dir / "error.log", "w") as f:
            f.write(f"Error: {str(e)}\n\n")
            import traceback
            f.write("=" * 80 + "\n")
            f.write("Exception traceback:\n")
            f.write("=" * 80 + "\n")
            f.write(traceback.format_exc())
        results_volume.commit()
        logger.info(f"Saved partial results: {len(current_completed)}/{max_systems} systems completed")
        return {
            "output_dir": str(output_dir),
            "success": False,
            "error": str(e),
            "completed_systems": len(current_completed),
            "total_systems": max_systems,
        }


@app.local_entrypoint()
def main(
    agent_configs: str,
    systems_file: str | None = None,
    max_systems: int = 10,
    budget: int = 50,
    num_episodes: int = 5,
    infra: str = "modal",
    output_dir: str = "./results/baselines",
    seed: int | None = None,
    max_stoichiometry: int = 20,
    stability_tolerance: float = 0.1,
    use_wandb: bool = False,
    wandb_project: str = "made-baselines",
    wandb_tags: str | None = None,
    resume: bool = True,
    add_timestamp: str = "true",
):
    """
    CLI entry point for Modal.
    
    Usage:
        uv run modal run scripts/run_baseline_experiments_modal.py \\
            --agent-configs "generative_baseline mlip_baseline" \\
            --systems-file ./data/systems_binary.json \\
            --max-systems 20 \\
            --budget 50 \\
            --max-stoichiometry 20 \\
            --stability-tolerance 0.1
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dotenv.load_dotenv()

    # Parse agent_configs (space-separated string)
    agent_config_list = agent_configs.split()

    # Parse wandb_tags if provided
    wandb_tags_list = wandb_tags.split() if wandb_tags else None

    # Parse add_timestamp string to boolean
    add_timestamp_bool = add_timestamp.lower() in ("true", "1", "yes", "on")

    # Create timestamped base directory for this run (if enabled)
    if add_timestamp_bool:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        final_output_dir = f"{output_dir}/{timestamp}"
    else:
        final_output_dir = output_dir

    logger.info("Running experiments on Modal")
    logger.info(f"  Agent configs: {agent_config_list}")
    logger.info(f"  Systems file: {systems_file}")
    logger.info(f"  Max systems: {max_systems}")
    logger.info(f"  Budget: {budget}")
    logger.info(f"  Max stoichiometry: {max_stoichiometry}")
    logger.info(f"  Stability tolerance: {stability_tolerance} eV")
    logger.info(f"  Output dir: {final_output_dir} (stored in Modal volume)")
    if add_timestamp_bool:
        logger.info(f"  Timestamp: {timestamp}")

    # Prepare arguments for all experiments
    experiment_args = []
    for agent_config in agent_config_list:
        experiment_args.append((
            agent_config,
            systems_file or "",  # Will use config default if empty
            max_systems,
            budget,
            num_episodes,
            infra,
            final_output_dir,
            seed,
            use_wandb,
            wandb_project,
            wandb_tags_list,
            resume,
            max_stoichiometry,
            stability_tolerance,
        ))

    logger.info(f"\n{'='*80}")
    logger.info(f"Running {len(experiment_args)} experiments in parallel on Modal")
    logger.info(f"{'='*80}")

    run_single_baseline_experiment_modal_wrapper.spawn_map(experiment_args)
