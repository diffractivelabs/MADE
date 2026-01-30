"""
Multi-system benchmark runner.

Reads a list of chemical systems from config.experiment.systems (list of lists of elements),
runs the existing benchmark episode loop per system via the run_benchmark.run_episode function
on local or Modal infra, saves per-system outputs, and aggregates metrics across systems.
"""

import csv
import importlib.util
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import dotenv
import hydra
import modal
import numpy as np
import tqdm.auto as tqdm
from omegaconf import DictConfig, OmegaConf
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

import wandb

logger = logging.getLogger(__name__)


def _import_run_benchmark():
    """Dynamically import run_benchmark.py so script can be invoked directly."""
    script_path = Path(__file__).resolve().parent / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load run_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def aggregate_metrics(per_episode: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
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
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_multi_systems(config: DictConfig) -> None:
    dotenv.load_dotenv()

    # Configure logging level from config
    log_level = config.experiment.get("logging_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # clear elements from config
    config.dataset.elements = []

    rb = _import_run_benchmark()

    if config.logger.get("use_wandb", False):
        wandb.init(
            project=config.logger.get("wandb_project", "made-benchmark"),
            entity=config.logger.get("wandb_entity", None),
            tags=config.logger.get("wandb_tags", ["benchmark", "multi-system"]),
        )
        wandb.config.update(flatten_dict(OmegaConf.to_container(config, resolve=False)))
        wandb_run_name = wandb.run.name
        wandb_run_id = wandb.run.id
    else:
        wandb_run_id = None
        # Use timestamp as run name for checkpoint path when wandb is disabled
        wandb_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = Path(config.experiment.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load systems either directly from config or from systems_file (JSON)
    systems: list[list[str]] = []
    if config.experiment.get("systems_file", None):
        systems_file = Path(config.experiment.systems_file)
        with open(systems_file) as f:
            systems = [list(s) for s in json.load(f)]
    else:
        systems = [list(s) for s in config.experiment.get("systems", [])]

    # Optional cap on number of systems
    max_systems = config.experiment.get("max_systems", None)
    if max_systems is not None:
        systems = systems[: int(max_systems)]
    if not systems:
        raise ValueError(
            "experiment.systems must be a non-empty list of element lists, e.g., [[Co, Nb, Sn], [Li, O]]"
        )

    systems_dir = out_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    all_episodes_across_systems: list[dict[str, Any]] = []
    per_system_summaries: dict[str, dict[str, dict[str, float]]] = {}

    # Iterate systems and run episodes per system using run_episode
    try:
        for sys_idx, elements in enumerate(systems):
            system_id = "-".join(elements)
            logger.info(f"Running system {sys_idx + 1}/{len(systems)}: {system_id}")

            # Duplicate config and override dataset elements
            cfg_this = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
            cfg_this.dataset.elements = list(elements)

            # Prepare directories
            system_dir = systems_dir / system_id
            trajectories_dir = system_dir / "trajectories"
            summary_dir = system_dir / "summary"
            trajectories_dir.mkdir(parents=True, exist_ok=True)
            summary_dir.mkdir(parents=True, exist_ok=True)

            num_episodes = int(cfg_this.experiment.get("num_episodes", 1))
            per_episode: list[dict[str, Any]] = []

            try:
                if cfg_this.experiment.infra == "modal":
                    with modal.enable_output():
                        with rb.app.run():
                            episode_args = [
                                (cfg_this, ep, wandb_run_name, system_id)
                                for ep in range(num_episodes)
                            ]
                            for ep, result in enumerate(
                                rb.run_episode.starmap(episode_args)
                            ):
                                per_episode.append(result.get("metrics", {}))
                                # Save trajectory and per-episode PD image
                                with open(
                                    trajectories_dir / f"episode_{ep:03d}.json", "w"
                                ) as f_traj:
                                    json.dump(result, f_traj, indent=2)
                                # Only plot phase diagrams with 4 or fewer elements
                                num_elements = len(result["final_env_state"].get("elements", []))
                                if num_elements <= 4:
                                    phase_diagram = PhaseDiagram(
                                        [
                                            PDEntry.from_dict(e)
                                            for e in result["final_env_state"][
                                                "phase_diagram_all_entries"
                                            ]
                                        ]
                                    )
                                    fig = phase_diagram.get_plot(
                                        backend="plotly", show_unstable=1.0
                                    )
                                    fig.write_image(
                                        trajectories_dir
                                        / f"phase_diagram_episode_{ep:03d}.png"
                                    )
                else:
                    for ep in tqdm.trange(num_episodes, desc=f"Episodes ({system_id})"):
                        result = rb.run_episode.local(
                            cfg_this,
                            ep,
                            wandb_run_name=wandb_run_name,
                            system_id=system_id,
                        )
                        per_episode.append(result.get("metrics", {}))
                        with open(
                            trajectories_dir / f"episode_{ep:03d}.json", "w"
                        ) as f_traj:
                            json.dump(result, f_traj, indent=2)
                        # Only plot phase diagrams with 4 or fewer elements
                        num_elements = len(result["final_env_state"].get("elements", []))
                        if num_elements <= 4:
                            phase_diagram = PhaseDiagram(
                                [
                                    PDEntry.from_dict(e)
                                    for e in result["final_env_state"][
                                        "phase_diagram_all_entries"
                                    ]
                                ]
                            )
                            fig = phase_diagram.get_plot(
                                backend="plotly", show_unstable=1.0
                            )
                            fig.write_image(
                                trajectories_dir / f"phase_diagram_episode_{ep:03d}.png"
                            )
            except KeyboardInterrupt:
                import traceback

                logger.warning(
                    f"Keyboard interrupt, stopping benchmark part way through. Stack trace: {traceback.format_exc()}"
                )

            # Save ground truth phase diagram image from any episode's stored GT entries
            # Only plot phase diagrams with 4 or fewer elements
            if len(elements) <= 4:
                with open(trajectories_dir / f"episode_{0:03d}.json") as f_first:
                    first_result = json.load(f_first)
                phase_diagram_gt = PhaseDiagram(
                    [PDEntry.from_dict(e) for e in first_result["phase_diagram_gt"]]
                )
                fig_gt = phase_diagram_gt.get_plot(backend="plotly", show_unstable=1.0)
                fig_gt.write_image(summary_dir / "phase_diagram_gt.png")

            # Write per-system episodes metrics
            with open(summary_dir / "episodes.json", "w") as f_json:
                json.dump(per_episode, f_json, indent=2)
            all_keys: list[str] = sorted({k for m in per_episode for k in m.keys()})
            with open(summary_dir / "episodes.csv", "w", newline="") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=all_keys)
                writer.writeheader()
                for row in per_episode:
                    writer.writerow(row)

            # Per-system summary
            system_summary = aggregate_metrics(per_episode)
            with open(summary_dir / "summary.json", "w") as f_sum:
                json.dump(
                    {"episodes": len(per_episode), "summary": system_summary},
                    f_sum,
                    indent=2,
                )

            # Track for overall aggregation
            all_episodes_across_systems.extend(per_episode)
            per_system_summaries[system_id] = system_summary

            # Minimal wandb logging
            if config.logger.get("use_wandb", False):
                if os.path.exists(summary_dir / "phase_diagram_gt.png"):
                    wandb.log(
                        {
                            f"{system_id}/phase_diagram/ground_truth": wandb.Image(
                                str(summary_dir / "phase_diagram_gt.png")
                            )
                        }
                    )
    except KeyboardInterrupt:
        import traceback

        logger.warning(
            f"Keyboard interrupt, stopping benchmark part way through. Stack trace: {traceback.format_exc()}"
        )

    # Overall aggregation across systems
    overall_summary = aggregate_metrics(all_episodes_across_systems)
    overall_dir = out_dir / "overall_summary"
    overall_dir.mkdir(parents=True, exist_ok=True)
    with open(overall_dir / "summary.json", "w") as f_overall:
        json.dump(
            {
                "systems": ["-".join(s) for s in systems],
                "episodes_total": len(all_episodes_across_systems),
                "summary": overall_summary,
            },
            f_overall,
            indent=2,
        )

    # Per-system summary CSV
    per_system_csv = overall_dir / "per_system_summary.csv"
    metric_keys = sorted({k for s in per_system_summaries.values() for k in s.keys()})
    with open(per_system_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["system", "metric", "mean", "std", "sem"])
        for sys_id, metrics in per_system_summaries.items():
            for k in metric_keys:
                if k in metrics:
                    v = metrics[k]
                    writer.writerow(
                        [
                            sys_id,
                            k,
                            v.get("mean", ""),
                            v.get("std", ""),
                            v.get("sem", ""),
                        ]
                    )

    if config.logger.get("use_wandb", False):
        # resume overall run to add an overall table
        wandb.init(
            project=config.logger.get("wandb_project", "made-benchmark"),
            entity=config.logger.get("wandb_entity", None),
            resume="must",
            id=wandb_run_id,
        )
        overall_table = wandb.Table(
            columns=["metric", "mean", "std", "sem"],
            data=[
                [k, v["mean"], v["std"], v["sem"]] for k, v in overall_summary.items()
            ],
        )
        wandb.log({"overall_summary_table": overall_table})
        # save recursively
        wandb.save(f"{str(systems_dir)}/*/*/*", base_path=str(out_dir), policy="live")
        wandb.save(f"{str(overall_dir)}/*", base_path=str(out_dir), policy="live")
        wandb.finish()

    logger.info(f"Overall summary: {json.dumps(overall_summary, indent=2)}")
    logger.info(f"Saved per-system and overall results under {str(out_dir)}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    run_multi_systems()
