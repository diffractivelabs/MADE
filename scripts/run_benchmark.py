"""
Basic benchmark runner.
"""

import collections
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import dotenv
import hydra
import modal
import numpy as np
import tqdm.auto as tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb

logger = logging.getLogger(__name__)

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core.structure import Structure

# install requirements and port local code to modal
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")  # needed to install packages from GitHub
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
    .add_local_python_source("made")
)

app = modal.App("benchmark-runner", image=image)

# Create a volume for checkpointing episodes
checkpoint_volume = modal.Volume.from_name("benchmark-checkpoints", create_if_missing=True)

model_checkpoints_volume = modal.Volume.from_name("matopt-checkpoints", create_if_missing=True)


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flattens a nested dictionary by concatenating keys (for wandb logging of config)
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_checkpoint_dir() -> Path:
    """Get the base checkpoint directory."""
    # Use /checkpoints on Modal, or local ./checkpoints for local execution
    if Path("/checkpoints").exists():
        checkpoint_dir = Path("/checkpoints")
    else:
        checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(
    episode_id: int,
    run_name: str,
    system_id: str | None = None
) -> Path:
    """Get the checkpoint file path for an episode.
    
    Structure: /checkpoints/{run_name}/{system_id}/episode_{episode_id:03d}/checkpoint.json
    
    Checkpoints are organized by run name (top-level), system_id (second level), 
    and episode ID (subfolder) to prevent conflicts and ensure correct config loading.
    """
    checkpoint_dir = get_checkpoint_dir()
    if system_id:
        episode_dir = checkpoint_dir / run_name / system_id / f"episode_{episode_id:03d}"
    else:
        episode_dir = checkpoint_dir / run_name / f"episode_{episode_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = episode_dir / "checkpoint.json"
    return checkpoint_file


def find_checkpoint_by_episode(
    episode_id: int,
    run_name: str,
    system_id: str | None = None
) -> tuple[Path | None, dict[str, Any] | None]:
    """Find checkpoint by episode_id in the specified run_name folder.
    
    Returns:
        Tuple of (checkpoint_path, checkpoint_data) or (None, None) if not found.
    """
    checkpoint_path = get_checkpoint_path(episode_id, run_name, system_id)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            return checkpoint_path, checkpoint
    return None, None


def save_checkpoint(
    checkpoint_path: Path,
    agent_state: dict[str, Any],
    env_state: dict[str, Any],
    trajectory: list[dict[str, Any]],
    query_count: int,
    wandb_run_id: str | None = None,
    metrics_history: list[dict[str, Any]] | None = None,
    config: DictConfig | None = None,
) -> None:
    """Save checkpoint with agent state, env state, trajectory, wandb run ID, metrics history, and config."""
    checkpoint = {
        "agent_state": agent_state,
        "env_state": env_state,
        "trajectory": trajectory,
        "query_count": query_count,
        "wandb_run_id": wandb_run_id,
        "metrics_history": metrics_history or [],
    }
    # Save config if provided (convert DictConfig to dict for JSON serialization)
    if config is not None:
        checkpoint["config"] = OmegaConf.to_container(config, resolve=False)
    # Write atomically by writing to temp file first, then renaming
    temp_path = checkpoint_path.with_suffix(".tmp.json")
    with open(temp_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    temp_path.replace(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path} (query_count={query_count})")


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load checkpoint if it exists."""
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} (query_count={checkpoint.get('query_count', 0)})"
        )
        return checkpoint
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        return None


def restore_environment_from_trajectory(
    env, trajectory: list[dict[str, Any]]
) -> None:
    """Restore environment state by replaying trajectory.
    
    Note: The environment's oracle will be called again during replay.
    If the oracle is deterministic and/or has caching, results should match.
    """
    logger.info(f"Restoring environment from trajectory with {len(trajectory)} steps")
    # Reset environment first
    env.reset()

    # Replay each step in the trajectory
    for obs_dict in trajectory:
        # Reconstruct Structure from dict
        proposal_dict = obs_dict["proposal"]
        if isinstance(proposal_dict, dict):
            proposal = Structure.from_dict(proposal_dict)
        else:
            proposal = proposal_dict

        # Step the environment with the proposal
        # The environment will call oracle.evaluate again, but if the oracle
        # is deterministic or has caching, results should match the original trajectory.
        env.step(proposal)

    logger.info(f"Environment restored: query_count={env.query_count}")


@app.function(
    gpu="T4",
    volumes={"/checkpoints": checkpoint_volume, "/ckpts": model_checkpoints_volume},
    secrets=[
        modal.Secret.from_name("materials-project-api-key"),
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("openai-api-key"),
    ],
    timeout=60 * 60 * 24,
)
def run_episode(
    config: DictConfig,
    episode_id: int = 0,
    wandb_run_name: str = "benchmark",
    system_id: str | None = None,
) -> None:
    """Run a single episode on a modal cluster.
    
    Args:
        config: Hydra config
        episode_id: Episode number
        wandb_run_name: Wandb run name (or timestamp when wandb is disabled)
        system_id: System identifier (e.g., "Co-Nb-Sn")
    """

    # Configure logging level from config
    log_level = config.experiment.get("logging_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing handlers
    )

    # Check for existing checkpoint first to get wandb run ID if resuming
    checkpoint_path, checkpoint = find_checkpoint_by_episode(episode_id, wandb_run_name, system_id)
    wandb_run_id = None

    if checkpoint is not None:
        # Found a checkpoint
        if checkpoint.get("wandb_run_id"):
            wandb_run_id = checkpoint["wandb_run_id"]

    # Initialize wandb for this episode if enabled
    if config.logger.get("use_wandb", False):
        # Finish any active run before starting/resuming
        if wandb.run is not None:
            wandb.finish()
        
        if wandb_run_id:
            # Resume existing wandb run
            logger.info(f"Resuming wandb run with ID: {wandb_run_id}")
            wandb.init(
                project=config.logger.get("wandb_project", "made-benchmark"),
                entity=config.logger.get("wandb_entity", None),
                resume="must",
                id=wandb_run_id,
                settings=wandb.Settings(code_dir="."),
            )
        else:
            # Start new wandb run
            wandb.init(
                project=config.logger.get("wandb_project", "made-benchmark"),
                entity=config.logger.get("wandb_entity", None),
                group=wandb_run_name,
                name=f"{wandb_run_name}_ep_{episode_id}"
                if system_id is None
                else f"{wandb_run_name}_{system_id}_ep_{episode_id}",
                tags=config.logger.get("wandb_tags", ["benchmark", "episode"]),
                reinit=True,
                settings=wandb.Settings(code_dir="."),
            )
            wandb_run_id = wandb.run.id
        wandb.config.update(flatten_dict(OmegaConf.to_container(config, resolve=False)))

    # Set checkpoint_path if not already set (for new episodes or when no checkpoint found)
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(episode_id, wandb_run_name, system_id)

    # Build components via Hydra
    dataset = instantiate(config.dataset)
    oracle = instantiate(config.oracle)
    env = instantiate(config.environment, dataset=dataset, oracle=oracle)
    agent = instantiate(config.agent)
    results = {
        "metrics": {},
        "trajectory": [],
        "phase_diagram_gt": [e.as_dict() for e in env.ground_truth_pd.all_entries],
    }

    if checkpoint is not None:
        # Resume from checkpoint
        logger.info(f"Resuming episode {episode_id} from checkpoint")
        trajectory = checkpoint["trajectory"]
        query_count = checkpoint["query_count"]

        # Restore environment by replaying trajectory
        restore_environment_from_trajectory(env, trajectory)

        # Restore agent state
        agent.load_state(checkpoint["agent_state"])
        logger.info(f"Restored agent state: last_step={agent.last_step}")

        # Note: Metrics don't need to be re-logged to wandb as they are already saved
        # when we resume the wandb run, it will have all previously logged metrics
    else:
        # Start fresh
        logger.info(f"Starting new episode {episode_id}")
        trajectory: list[dict[str, Any]] = []
        query_count = 0

    tmp_trajectory_file = Path(f"./trajectory_episode_{episode_id:03d}.json")
    try:
        while not env.is_done():
            state = env.get_state()
            _, struct = agent(state)
            obs, _ = env.step(struct)
            obs["proposal"] = obs["proposal"].as_dict()
            trajectory.append(obs)
            query_count += 1

            # Save checkpoint after each step
            agent_state = agent.get_state()
            env_state = env.get_state()
            # Get metrics history from environment if available
            metrics_history = None
            if hasattr(env, "get_metrics_history"):
                metrics_history = env.get_metrics_history()
            save_checkpoint(checkpoint_path, agent_state, env_state, trajectory, query_count, wandb_run_id, metrics_history, config)
            # Commit volume to persist checkpoint (only on Modal)
            try:
                checkpoint_volume.commit()
            except NameError:
                # checkpoint_volume not available (local execution)
                pass

            with open(tmp_trajectory_file, "w") as f:
                json.dump(trajectory, f, indent=2)
            if config.experiment.verbose:
                logger.info(f"Observation: {obs}")
                logger.info(
                    f"Metrics: {json.dumps(env.get_latest_metrics(), indent=2)}"
                )

            # Log step-level metrics to wandb
            if config.logger.get("use_wandb", False):
                step_metrics = {"episode_id": episode_id, "query_count": query_count}
                # Log any numeric values from the observation, convert bool to int
                for k, v in obs.items():
                    if isinstance(v, (int, float)):
                        step_metrics[f"obs/{k}"] = float(v)
                    elif isinstance(v, bool):
                        step_metrics[f"obs/{k}"] = int(v)
                # log overall metrics
                for k, v in env.get_latest_metrics().items():
                    if isinstance(v, (int, float)):
                        step_metrics[f"overall/{k}"] = float(v)
                    elif isinstance(v, bool):
                        step_metrics[f"overall/{k}"] = int(v)
                wandb.log(step_metrics)
                wandb.save(tmp_trajectory_file, policy="live")
    except KeyboardInterrupt:
        import traceback
        logger.warning(
            f"Keyboard interrupt, stopping episode part way through. Stack trace: {traceback.format_exc()}"
        )
        # Save checkpoint before exiting
        agent_state = agent.get_state()
        env_state = env.get_state()
        metrics_history = None
        if hasattr(env, "get_metrics_history"):
            metrics_history = env.get_metrics_history()
        save_checkpoint(checkpoint_path, agent_state, env_state, trajectory, query_count, wandb_run_id, metrics_history, config)
        try:
            checkpoint_volume.commit()
        except NameError:
            pass
        logger.info("Checkpoint saved before exit")

    final_metrics = {"episode_id": episode_id}
    final_metrics.update(
        {
            f"final/{k}": v
            for k, v in env.get_latest_metrics().items()
            if isinstance(v, (int, float))
        }
    )
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    results["metrics"] = final_metrics
    results["trajectory"] = trajectory
    results["final_env_state"] = (
        env.get_state()
    )  # final state is the final phase diagram
    # results["final_agent_state"] = agent.get_state()

    # Log final episode metrics to wandb
    if config.logger.get("use_wandb", False):
        wandb.log(final_metrics)
        wandb.finish()

    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Removed checkpoint file {checkpoint_path} after successful completion")
        try:
            checkpoint_volume.commit()
        except NameError:
            pass

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_benchmark(config: DictConfig) -> None:
    """Run benchmark defined by Hydra config.
    Args:
        config: Hydra DictConfig with groups: dataset, oracle, env, agent, experiment
    """
    dotenv.load_dotenv()

    # Configure logging level from config
    log_level = config.experiment.get("logging_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.experiment.get("seed", None):
        random.seed(int(config.experiment.seed))
        np.random.seed(int(config.experiment.seed))

    # Initialize main wandb run for the entire benchmark
    if config.logger.get("use_wandb", False):
        wandb.init(
            project=config.logger.get("wandb_project", "made-benchmark"),
            entity=config.logger.get("wandb_entity", None),
            tags=config.logger.get("wandb_tags", ["benchmark"]),
            name=config.logger.get("wandb_run_name", None),
        )
        wandb.config.update(flatten_dict(OmegaConf.to_container(config, resolve=False)))
        wandb_run_name = wandb.run.name
        wandb_run_id = wandb.run.id
    else:
        # Use timestamp as run name for checkpoint path when wandb is disabled
        wandb_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(config.experiment.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Run episodes
    num_episodes = int(config.experiment.get("num_episodes", 1))
    per_episode: list[dict[str, Any]] = []
    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    try:
        # run episodes in parallel
        if config.experiment.infra == "modal":
            with modal.enable_output():
                with app.run():
                    # Create arguments with episode IDs for each episode
                    episode_args = [
                        (config, ep, wandb_run_name, None) for ep in range(num_episodes)
                    ]
                    for ep, result in enumerate(run_episode.starmap(episode_args)):
                        per_episode.append(result.get("metrics", {}))
                        # Save trajectory and phase diagram per episode
                        with open(
                            trajectories_dir / f"episode_{ep:03d}.json", "w"
                        ) as f_traj:
                            json.dump(result, f_traj, indent=2)
                        # can load phase diagram from the final state
                        phase_diagram = PhaseDiagram(
                            [
                                PDEntry.from_dict(e)
                                for e in result["final_env_state"][
                                    "phase_diagram_all_entries"
                                ]
                            ]
                        )
                        # Only plot phase diagrams with 4 or fewer elements
                        num_elements = len(result["final_env_state"].get("elements", []))
                        if num_elements <= 4:
                            fig = phase_diagram.get_plot(
                                backend="plotly", show_unstable=1.0
                            )
                            fig.write_image(
                                trajectories_dir / f"phase_diagram_episode_{ep:03d}.png"
                            )
        else:
            for ep in tqdm.trange(num_episodes, desc="Running episodes"):
                result = run_episode.local(
                    config, ep, wandb_run_name=wandb_run_name, system_id=None
                )
                per_episode.append(result.get("metrics", {}))
                # Save trajectory and phase diagram per episode
                with open(trajectories_dir / f"episode_{ep:03d}.json", "w") as f_traj:
                    json.dump(result, f_traj, indent=2)
                # can load phase diagram from the final state
                phase_diagram = PhaseDiagram(
                    [
                        PDEntry.from_dict(e)
                        for e in result["final_env_state"]["phase_diagram_all_entries"]
                    ]
                )
                # Only plot phase diagrams with 4 or fewer elements
                num_elements = len(result["final_env_state"].get("elements", []))
                if num_elements <= 4:
                    fig = phase_diagram.get_plot(backend="plotly", show_unstable=1.0)
                    fig.write_image(
                        trajectories_dir / f"phase_diagram_episode_{ep:03d}.png"
                    )
    except KeyboardInterrupt:
        import traceback

        logger.warning(
            f"Keyboard interrupt, stopping benchmark part way through. Stack trace: {traceback.format_exc()}"
        )

    # save ground truth phase diagram too (only for systems with 4 or fewer elements)
    num_elements = len(result["final_env_state"].get("elements", []))
    if num_elements <= 4:
        phase_diagram_gt = PhaseDiagram(
            [PDEntry.from_dict(e) for e in result["phase_diagram_gt"]]
        )
        fig = phase_diagram_gt.get_plot(backend="plotly", show_unstable=1.0)
        fig.write_image(summary_dir / "phase_diagram_gt.png")

    # Write episodes metrics to JSON (array)
    with open(summary_dir / "episodes.json", "w") as f_json:
        json.dump(per_episode, f_json, indent=2)

    # Also write episodes metrics to CSV
    # Determine all keys present across episodes for consistent columns
    all_keys: list[str] = sorted({k for m in per_episode for k in m.keys()})
    with open(summary_dir / "episodes.csv", "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=all_keys)
        writer.writeheader()
        for row in per_episode:
            writer.writerow(row)

    # Aggregate metrics: automatically include all numeric final/* keys across episodes
    def collect(name: str) -> list[float]:
        vals: list[float] = []
        for m in per_episode:
            if name in m and isinstance(m[name], (int, float)):
                vals.append(float(m[name]))
        return vals

    all_final_keys = sorted(
        {k for m in per_episode for k in m.keys() if k.startswith("final/")}
    )
    summary: dict[str, dict[str, float]] = {}
    for k in all_final_keys:
        values = collect(k)
        if values:
            arr = np.array(values, dtype=float)
            n = len(arr)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "sem": float(np.std(arr) / np.sqrt(n)) if n > 0 else 0.0,
            }

    with open(summary_dir / "summary.json", "w") as f:
        json.dump({"episodes": len(per_episode), "summary": summary}, f, indent=2)

    # Log summary statistics and phase diagram images to wandb
    if config.logger.get("use_wandb", False):
        # Finish any active run before resuming the overall run
        if wandb.run is not None:
            wandb.finish()
        
        # resume the overall run
        wandb.init(
            project=config.logger.get("wandb_project", "made-benchmark"),
            entity=config.logger.get("wandb_entity", None),
            resume="must",
            id=wandb_run_id,
        )
        wandb.log(
            {
                f"phase_diagram/episode_{ep:03d}": wandb.Image(
                    str(trajectories_dir / f"phase_diagram_episode_{ep:03d}.png")
                )
                for ep in range(num_episodes)
                if os.path.exists(
                    trajectories_dir / f"phase_diagram_episode_{ep:03d}.png"
                )
            }
        )
        if os.path.exists(summary_dir / "phase_diagram_gt.png"):
            wandb.log(
                {
                    "phase_diagram/ground_truth": wandb.Image(
                        str(summary_dir / "phase_diagram_gt.png")
                    )
                }
            )

        # Create summary table for wandb
        summary_table = wandb.Table(
            columns=["metric", "mean", "std", "sem"],
            data=[[k, v["mean"], v["std"], v["sem"]] for k, v in summary.items()],
        )
        wandb.log({"summary_table": summary_table})
        # save all files in output directory to wandb
        wandb.save(f"{str(trajectories_dir)}/*", base_path=str(out_dir), policy="live")
        wandb.save(f"{str(summary_dir)}/*", base_path=str(out_dir), policy="live")

        wandb.finish()

    logger.info(f"Summary results: {json.dumps(summary, indent=2)}")
    logger.info(
        f"Saved results to {config.experiment.output_dir} (episodes.json, episodes.csv, summary.json, trajectories/*.json)"
    )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    run_benchmark()
