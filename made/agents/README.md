# Agents Overview

Agents are organized into two main categories:

1. **Workflow Agents**: Use a standard planner → generator → filter → scorer workflow
   - These agents orchestrate four components in a loop:
     1. **Planner** proposes a `Plan` (compositions to explore and number of candidates).
     2. **Generator** produces candidate `Structure`s for those compositions.
     3. **Filter** validates and filters candidate structures (optional, defaults to NoOpFilter).
     4. **Scorer** selects one winning structure from the candidates for evaluation.

2. **Orchestrator/Specialized Agents**: Use custom logic to orchestrate components or perform specialized tasks
   - These agents may use components (generators, filters, scorers) but with custom orchestration logic
   - Examples: `LLMReActOrchestratorAgent`

## Core interfaces

Defined in `made.agents.base`:

- `Plan` – contains `compositions: list[Composition]`, `num_candidates: int`, and optional `constraints`.
- `Planner.propose(state, previous=None) -> Plan`.
- `Generator.generate(plan, state) -> list[Structure]`.
- `Filter.filter(candidates, state) -> list[Structure]`.
- `Scorer.select(candidates, state) -> Structure`.

## Available implementations

- Planners: `GridSearchPlanner`, `LLMPlanner`
- Generators: `RandomGenerator`, `ChemeleonGenerator`, `CachedGenerator`.
- Filters: `NoOpFilter`, `SMACTValidityFilter`, `MinDistanceFilter`, `UniquenessFilter`, `FilterChain`.
- Scorers: `RandomSelector`, `CompositionDiversity`, `LLMScorer`, `OracleScorer`, `ScorerChain`.

## Component details

### Planners

Planners decide compositions and candidate counts:

- LLMPlanner
  - Uses an LLM to propose the next compositions given experiment context.
  - Key params: `max_stoichiometry`, `num_compositions`, `num_candidates`, plus `llm_config` and `context_config`.
  - `llm_config` fields:
    - `model` (e.g. `anthropic/claude-sonnet-4-20250514`), `max_output_tokens`, `temperature`, `cache` (caching via dspy).
  - `context_config` fields:
    - `objective_prompt` (system instructions), toggles for `include_structure_info`, `include_composition_counter`, `include_recent_trial`, and `previous_observations` window.
  - Requires: `state['elements']`, convex-hull related context is summarized automatically from state.

- GridSearchPlanner (recommended)
  - Enumerates all stoichiometric compositions up to `max_stoichiometry` over `state['elements']`, and selects a batch each step using a pluggable selection strategy.
  - Out-of-the-box strategies include: `random`, `ucb`, `unexplored`, `iterative`, and `diversity` (vectorized, distance- and weight-based with reduced-formula masking and elemental references).
  - Key params: `max_stoichiometry`, `num_compositions`, `num_candidates`, `seed`, `selection_strategy`, `selection_strategy_kwargs`, `score_function`, `filter_by_smact_validity`, `return_all_compositions`.
  - Selection strategy params (via `selection_strategy_kwargs`):
    - UCB: `exploration_factor`, `ucb_reward` ("success_rate", "failure_rate", "best_score")
    - Random/Iterative: `max_attempts_per_composition`, `move_on_success`
    - Diversity: `distance_metric`, `unattempted_weight`, `attempt_weight_factor`, `failure_weight_factor`, `mask_same_reduced_formula`
  - `score_function`: One of "energy_per_atom", "formation_energy_per_atom", "e_above_hull" (used for tracking composition statistics).
  - `filter_by_smact_validity`: If True, only enumerate compositions that pass SMACT validity checks.
  - `return_all_compositions`: If True, return all valid compositions in plan (useful with cached generator, ignores selection strategy).
  - Requires: `state['elements']`. Centralizes composition enumeration and selection logic with a single consistent API.

### Generators

Generators produce structures for the planned compositions:

- RandomGenerator
  - Creates random lattices and places atoms per composition. Falls back to one-atom-per-element from `state['elements']` if `plan.compositions` empty.
  - Params: `seed`, `lattice_length_range`, `lattice_angle_range`.
  - Requires: `state['elements']` when `plan.compositions` is empty.

- ChemeleonGenerator
  - Uses Chemeleon-DNG library directly for crystal structure generation.
  - Primarily uses CSP (Crystal Structure Prediction) mode to generate structures for given compositions.
  - Params: `task` ("csp" or "dng"), `batch_size`, `device` ("cpu" or "cuda"), `num_atom_distribution` (for DNG only, e.g., "mp-20"), `output_dir`.
  - CSP mode (default): Uses `plan.compositions` as formulas to predict stable structures for given compositions.
  - DNG mode (optional): Generates novel crystal structures without predefined compositions based on atom number distribution.
  - Installation: Installed directly from GitHub (`chemeleon-dng @ git+https://github.com/hspark1212/chemeleon-dng.git`).
  - Checkpoints are automatically downloaded from Figshare on first use via the `download_util` module.
  - Notes:
    - Structures are generated in CIF format and converted to pymatgen Structure objects.
    - Model is lazy-loaded in `setup()` method to avoid initialization overhead.
    - Currently generates structures in P1 space group (symmetry detection may be needed for other space groups).

- CachedGenerator
  - Generates candidates once and caches them for reuse, avoiding redundant generation.
  - Generates candidates only on the first call per composition (or globally), then returns cached candidates on subsequent calls (excluding already-selected ones).
  - Params: `base_generator` (generator to use for creating candidates), `num_initial_candidates` (number to generate and cache initially), `cache_by_composition` (if True, maintain separate cache per composition; if False, single global cache).
  - Useful when: Using expensive generators (e.g., Chemeleon) with planners that revisit compositions, or when using `GridSearchPlanner` with `return_all_compositions=True`.

### Scorers

Scorers pick one structure from candidates:

- LLMScorer
  - Uses an LLM to select one candidate structure to evaluate next, balancing likelihood of stability (low E_hull) and diversity.
  - Key params: `llm_config`, `context_config` with an `objective_prompt` and the same context toggles as above.
  - Input/Output: returns a single `Structure` from a list of candidates; falls back to the first candidate if the LLM returns an invalid index.

- RandomSelector
  - Uniform random choice.
  - Params: `seed`.

- CompositionDiversity
  - Chooses candidate farthest in composition space from observed.
  - Params: `distance_metric` (euclidean|manhattan|cosine), `seed`.
  - Requires: `state['phase_diagram_all_entries']`.

- OracleScorer
  - Wraps any `Oracle` to score candidate structures using oracle evaluation (relaxations, calculators, etc.).
  - Supports caching and reranking when state changes (for e_above_hull scoring).
  - Params: `oracle` (Oracle instance), `score_function` ("energy_per_atom", "formation_energy_per_atom", "e_above_hull"), `enable_cache` (default: True), `rerank_on_state_change` (default: False, only relevant for e_above_hull).
  - Batching is handled internally by the oracle. Caching avoids redundant oracle evaluations for identical structures.
  - Requires: `state['phase_diagram_all_entries']` for formation_energy_per_atom or e_above_hull scoring.

- ScorerChain
  - Chains multiple scorers together for multiobjective optimization.
  - Combines scores from multiple scorers using weighted sum or Pareto dominance.
  - Params: `scorers` (list of scorers), `weights` (optional list of weights, default: equal weights), `combination_method` ("weighted_sum" or "pareto"), `normalize_scores` (default: True, normalizes each scorer's scores to [0, 1] before combination).
  - Useful for: Combining multiple objectives (e.g., stability + diversity, energy + properties).

### Filters

Filters validate candidate structures between generation and scoring, removing invalid structures:

- NoOpFilter
  - Pass-through filter that returns all candidates unchanged.
  - Use this when no filtering is desired (default).

- SMACTValidityFilter
  - Filters structures based on SMACT chemical validity checks.
  - Removes structures that fail charge balance and electronegativity constraints.
  - Uses the same validity logic as the evaluation metrics.

- MinDistanceFilter
  - Filters structures based on minimum interatomic distance.
  - Removes structures where atoms are too close together (below threshold).
  - Params: `min_distance_threshold` (default: 0.5 Angstroms).

- UniquenessFilter
  - Filters structures that are unique compared to already attempted structures.
  - Removes structures that match (via StructureMatcher) any structure that has already been attempted.
  - Params: `ltol` (fractional length tolerance, default: 0.2), `stol` (site tolerance, default: 0.3), `angle_tol` (angle tolerance in degrees, default: 5.0), `primitive_cell` (use primitive cell for matching, default: True).
  - Requires: `state['phase_diagram_all_entries']` to extract reference structures from attempted entries.
  - Useful for: Avoiding redundant evaluation of structures that are structurally equivalent to previously attempted ones.

- FilterChain
  - Chains multiple filters together, applying them sequentially.
  - Each filter receives the output of the previous filter.
  - Params: `filters` (list of Filter instances).
  - Example: Apply min-distance filtering first, then SMACT validity.

**Filter Chaining:**

Filters can be chained together to apply multiple validation criteria. You can either:

1. Use a pre-made chain config (e.g., `chain` with custom filters)
2. Use FilterChain directly in your config:

```yaml
filter:
  _target_: made.agents.filters.FilterChain
  filters:
    - _target_: made.agents.filters.MinDistanceFilter
      min_distance_threshold: 0.5
    - _target_: made.agents.filters.SMACTValidityFilter
```

When using defaults, the WorkflowAgent constructor accepts either a single Filter or a list of Filters, automatically wrapping lists in a FilterChain.

## Configuration

Hydra-style configs live under `configs/agent/`. For example, `random_generator_baseline.yaml`:

```yaml
_target_: made.agents.OneShotWorkflowAgent

defaults:
  - planner: random
  - generator: cached
  - filter: noop  # Options: noop, smact, min_distance, chain (or custom FilterChain)
  - scorer: random
```

Switch components by changing the defaults, e.g. use `planner: diversity` and `scorer: diversity` with `generator: chemeleon` and `filter: smact` to explore diverse compositions, validate with SMACT, and select diverse candidates using Chemeleon for structure generation.

## Agent Hierarchy

- **Agent** (base): Minimal abstract base class defining the interface for all agents

### Workflow Agents

- **WorkflowAgent**: Base class for planner → generator → filter → scorer workflow agents
  - **OneShotWorkflowAgent**: One-shot strategy (plan once, generate once, select once)
  - Uses the standard workflow with pluggable components (planners, generators, filters, scorers)

### Orchestrator/Specialized Agents

- **LLMReActOrchestratorAgent**: LLM ReAct agent that orchestrates generators/filters/scorers using LLM decision-making
  - Uses DSPy ReAct to decide actions (generate, score, query, select)
  - Maintains a buffer of pre-validated structures
  - Can use multiple generators and scorers as tools

## Extending

### Creating a New Agent

To create a completely new agent with custom logic:

1. Subclass `Agent` from `made.agents.base`:

```python
from made.agents.base import Agent
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

class MyCustomAgent(Agent):
    def __init__(self, my_param: str):
        super().__init__()
        self.my_param = my_param

    def propose_composition_and_structure(self, state: dict) -> tuple[Composition, Structure]:
        """Propose a structure for oracle evaluation."""
        # Your custom logic here
        composition = Composition("Li2O")
        structure = ...  # Generate or select a structure
        return composition, structure

    def get_state(self) -> dict:
        """Return agent state for checkpointing."""
        return {"my_param": self.my_param, "last_step": self.last_step}

    def load_state(self, state: dict) -> None:
        """Load agent state from checkpoint."""
        self.my_param = state.get("my_param", self.my_param)
        self.last_step = state.get("last_step", 0)

    def update_state(self, state: dict) -> None:
        """Update agent based on environment feedback."""
        action = self._update_last_step(state)
        if action == "skip":
            return
        # Process new observations from state["last_observation"]
```

2. Create a Hydra config in `configs/agent/my_agent.yaml`:

```yaml
_target_: my_module.MyCustomAgent
my_param: "value"
```

3. Use it: `uv run scripts/run_benchmark.py agent=my_agent`

### Creating a Workflow Agent

For agents that follow the planner → generator → filter → scorer workflow:

```python
from made.agents.base import WorkflowAgent

class MyWorkflowAgent(WorkflowAgent):
    def propose_composition_and_structure(self, state: dict) -> tuple[Composition, Structure]:
        # 1. Plan: decide which compositions to explore
        plan = self.planner.propose(state)

        # 2. Generate: create candidate structures
        candidates = self.generator.generate(plan, state)

        # 3. Filter: remove invalid candidates
        candidates = self.filter.filter(candidates, state)

        # 4. Score: select the best candidate
        selected = self.scorer.select(candidates, state)

        return selected.composition, selected
```

### Creating New Components

Each component type has an abstract base class in `made.agents.base`. Implement the required methods and document any state requirements.

#### New Planner

```python
from made.agents.base import Planner, Plan

class MyPlanner(Planner):
    def propose(self, state: dict, previous: dict | None = None) -> Plan:
        """Return compositions to explore and number of candidates."""
        compositions = [Composition("Li2O"), Composition("LiO2")]
        return Plan(compositions=compositions, num_candidates=10)

    def get_state(self) -> dict:
        return {}

    def update_state(self, state: dict) -> None:
        pass
```

#### New Generator

```python
from made.agents.base import Generator, Plan
from pymatgen.core.structure import Structure

class MyGenerator(Generator):
    def generate(self, plan: Plan, state: dict) -> list[Structure]:
        """Generate candidate structures for the given plan."""
        structures = []
        for composition in plan.compositions:
            for _ in range(plan.num_candidates):
                structure = ...  # Your generation logic
                structures.append(structure)
        return structures

    def get_state(self) -> dict:
        return {}

    def update_state(self, state: dict) -> None:
        pass
```

#### New Filter

```python
from made.agents.base import Filter, FilterResult
from pymatgen.core.structure import Structure

class MyFilter(Filter):
    filter_name = "MyFilter"

    def _filter_with_results(
        self, candidates: list[Structure], state: dict
    ) -> tuple[list[Structure], list[FilterResult]]:
        """Filter candidates and return detailed results."""
        passed = []
        results = []
        for structure in candidates:
            is_valid = ...  # Your validation logic
            results.append(FilterResult(
                passed=is_valid,
                filter_name=self.filter_name,
                rejection_reason=None if is_valid else "Reason for rejection"
            ))
            if is_valid:
                passed.append(structure)
        return passed, results

    def get_state(self) -> dict:
        return {}

    def update_state(self, state: dict) -> None:
        pass
```

#### New Scorer

```python
from made.agents.base import Scorer, ScoreResult
from pymatgen.core.structure import Structure

class MyScorer(Scorer):
    scorer_name = "MyScorer"

    def _score_with_results(
        self, candidates: list[Structure], state: dict
    ) -> tuple[list[float], list[ScoreResult]]:
        """Score candidates (higher is better)."""
        scores = []
        results = []
        for structure in candidates:
            score = ...  # Your scoring logic (higher = better)
            scores.append(score)
            results.append(ScoreResult(
                score=score,
                scorer_name=self.scorer_name,
                details={"additional_info": "..."}
            ))
        return scores, results

    def get_state(self) -> dict:
        return {}

    def update_state(self, state: dict) -> None:
        pass
```

### Registering Components

After implementing a component:

1. Add it to the appropriate `__init__.py` (e.g., `made/agents/planners/__init__.py`)
2. Create a Hydra config under `configs/agent/<component_type>/my_component.yaml`
3. Reference it in your agent config using Hydra defaults
