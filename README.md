# MADE: MAterials Discovery Environments

This is the official code repository for the paper:

> **MADE: Benchmark Environments for Closed-Loop Materials Discovery**
> [Arxiv Preprint](https://arxiv.org/abs/2601.20996)
> [NeurIPS AI4Mat 2025 Workshop Paper](https://openreview.net/pdf?id=Cfj7uBu5dy)

## Overview

MADE (MAterials Discovery Environments) provides dynamic benchmark environments
for evaluating end-to-end autonomous materials discovery pipelines. MADE simulates closed-loop
discovery campaigns where agents propose, evaluate, and refine candidate materials
under constrained oracle budgets.

## Key Features

- **Closed-loop evaluation**: Agents iteratively propose structures and receive feedback
- **Modular agents**: Compose generators, planners, scorers, and filters
- **Flexible environments**: Define your own convex hull discovery tasks with any oracle.
- **Discovery metrics**: AF, EF, AUDC, mSUN for comparing strategies

## Installation

Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Set up API keys or add to .env file:

```bash
export MATERIALS_PROJECT_API_KEY=your_key
export WANDB_API_KEY=your_key  # optional
export ANTHROPIC_API_KEY=your_key  # for Anthropic agents
export OPENAI_API_KEY=your_key  # for OpenAI agents
```

Add these API keys to Modal secrets too if running on Modal.

Update the wandb entity in `configs/logger/wandb.yaml` to your wandb username/team if you want to save progress to wandb. Otherwise, set `use_wandb: false`.

## Quick Start

```bash
# Run a single benchmark locally (Li-O system, random agent, ORB oracle, 3 episodes, 5 queries)
uv run scripts/run_benchmark.py

# Run on Modal (parallel episodes)
uv run scripts/run_benchmark.py experiment.infra=modal

# Run with custom config
uv run scripts/run_benchmark.py dataset.elements='[Fe,O]' experiment.num_episodes=5
```

## Architecture

- **Environment** (`ConvexHullEnvironment`): Defines discovery task on a phase diagram
- **Oracle** (ORB, MACE, Analytic): Evaluates formation energy of proposed structures
- **Agent**: Pipeline or Orchestrator for that proposes structures for evaluation. See `src/made/agents/README.md` for more details on available agents and extending.

All environment, oracle, and agent components are defined via Hydra config files in `configs/`. These can be combined in a variety of ways to create different agents and environments.

## Running Baseline Experiments

We provide the configs used to run the baseline experiments in the paper in the agents config folder. These can be run using the scripts in `scripts/`. For example:

```bash
# Local (sequential)
uv run scripts/run_baseline_experiments.py \
    --agent-configs "random_generator_baseline chemeleon_generative_baseline" \
    --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json

# Modal (parallel)
uv run modal run --detach scripts/run_baseline_experiments_modal.py \
    --agent-configs "random_generator_baseline chemeleon_generative_baseline" \
    --systems-file ./data/systems_10_mp_20/systems_ternary_n10_maxatoms20_intermetallic_smact.json
```

to run the random generator and chemeleon generative baseline on ternary intermetallic systems.

This will save results to `./results/baselines/`, or on a Modal volume if running on Modal.

## Results Format

### Single Run (`run_benchmark.py`)

```text
results/<timestamp>-<oracle>-<agent>/
├── .hydra/                      # Hydra config files
│   ├── config.yaml              # Full resolved config
│   └── overrides.yaml           # CLI overrides used
├── trajectories/
│   ├── episode_000.json         # Full trajectory for episode 0
│   ├── episode_001.json         # Full trajectory for episode 1
│   └── phase_diagram_*.png      # Phase diagram visualizations
├── summary/
│   ├── summary.json             # Aggregated metrics (mean/std across episodes)
│   ├── episodes.json            # Per-episode metrics
│   ├── episodes.csv             # Per-episode metrics (CSV format)
│   └── phase_diagram_gt.png     # Ground truth phase diagram
└── run_benchmark.log            # Execution log
```

### Baseline Experiments (`run_baseline_experiments.py`)

```text
results/baselines_<date>/
└── <agent_config>_<systems_file>_<N>systems_<B>queries_<T>stabilitymeV/
    ├── experiment_metadata.json   # Experiment configuration
    ├── progress.json              # Progress tracking (status, completed systems)
    ├── overall_summary/           # Aggregated metrics across all systems
    │   ├── summary.json           # Summary statistics (mean/std/sem)
    │   └── per_system_summary.csv # Per-system breakdown
    └── systems/
        └── <system_id>/           # e.g., Co-Mg-Na
            ├── trajectories/
            │   ├── episode_000.json
            │   ├── episode_001.json
            │   └── phase_diagram_episode_*.png
            └── summary/
                ├── summary.json
                ├── episodes.json
                ├── episodes.csv
                └── phase_diagram_gt.png
```

### Analyzing Results

See `notebooks/basic_analysis.ipynb` for a basic example of loading and analyzing results from a single benchmark run. `notebooks/results_analysis_utils.py` contains utility functions for loading and analyzing results from a single benchmark run and comparing baseline experiments.

## Generating Systems

We provide the scripts to generate the systems used in the baseline experiments in `scripts/generate_systems.py`. For example:

```bash
uv run scripts/generate_systems.py --output-dir ./data/systems_10_mp_20 --filter-by-smact --system-sizes [3,4,5] --only-intermetallics
```

to generate the systems used in the baseline experiments.

## Extending MADE

MADE is designed to be extensible. You can create custom components by subclassing the base classes and adding Hydra configs:

- **Oracles**: Subclass `Oracle` from `made.oracles.base`, implement `evaluate(structure) -> dict`
- **Environments**: Subclass `Environment` from `made.envs.base`, implement `reset()`, `step()`, `get_state()`
- **Agents**: See `src/made/agents/README.md` for detailed documentation on creating new agents and components (planners, generators, filters, scorers)

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use MADE in your research, please cite our paper:

```bibtex
@misc{malik2026made,
      title={MADE: Benchmark Environments for Closed-Loop Materials Discovery}, 
      author={Shreshth A Malik and Tiarnan Doherty and Panagiotis Tigas and Muhammed Razzak and Stephen J. Roberts and Aron Walsh and Yarin Gal},
      year={2026},
      eprint={2601.20996},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.20996}, 
}
```
