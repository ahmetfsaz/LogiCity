# Goal-Oriented Semantic Communication in LogiCity

Experiment code for:

> **Goal-Oriented Semantic Communication for Logical Decision Making**
> Ahmet Faruk Saz, Faramarz Fekri — *IEEE GLOBECOM 2026*
> [arXiv:2604.19614](https://arxiv.org/abs/2604.19614)

This is a fork of [LogiCity](https://github.com/Jaraxxus-Me/LogiCity) (Li et al., NeurIPS 2024
D&B). The simulator is forked; this fork adds a semantic communication layer on top of it.

## What this adds

Agents in LogiCity act on first-order logic traffic rules evaluated by a Z3 solver, but each
sees only a limited field of view. A Global Navigation Assistant (GNA) can send an agent a
small number of additional FOL expressions about its surroundings. Which ones should it send?

This fork implements the goal-oriented selection rule from the paper. Task rules partition
world states into **goal states** — logical equivalence classes that lead to the same action —
and transmitted evidence is chosen to most sharply resolve which goal state holds. The
objective is optimized by a polynomial-time lexicographic comparison over small integers,
avoiding direct construction of the doubly-exponential constituent space. The baseline selects
the same number of expressions uniformly at random.

## Implementation

Three files hold the method; everything else in `logicity/` is upstream.

**[`logicity/utils/semantic_selection.py`](logicity/utils/semantic_selection.py)** — the
selection objective. Assigns inductive probabilities by uniform weight over constituents, the
maximally specific world descriptions of a dyadic first-order language over 11 monadic and 6
dyadic predicates, then chooses the evidence subset minimizing the summed Bernoulli variance
of the traffic-rule hypotheses:

```
min_{Ê ⊆ E}  Σ_i  p(Ê) · p(Γ_i | Ê) · (1 − p(Γ_i | Ê))
```

The direct form underflows in floating point, since the exponents involved have billions of
digits, so `compute_objective_simplified` evaluates the equivalent lexicographic key over small
integers instead. `select_optimal_subset` is the entry point; the module docstring derives the
probability model in full.

**[`logicity/agents/gna.py`](logicity/agents/gna.py)** — the Global Navigation Assistant.
Collects agent state before Z3 reasoning, then broadcasts a filtered top-k subset to each
agent, giving it awareness beyond its own field of view.

**[`logicity/agents/lna.py`](logicity/agents/lna.py)** — the Local Navigation Assistants, a
grid of zone-level assistants placed at intersections. Each covers a non-overlapping
rectangular zone: cars uplink their top-`k1` field-of-view entities, and the assistant
aggregates, filters, re-grounds, and relays top-`k2` entities back to each ego car. This is
what the `semantic_lna*` selection modes exercise.

The remaining files in `logicity/agents/` (`basic.py`, `car.py`, `pedestrian.py`, `bus.py`)
are upstream's agent classes.

## Setup

Install as original branch describes, then activate the environment:

```bash
conda activate logicity
```

**Before running anything**, open the scripts in `scripts/sim/` and fix the conda path. They
currently hardcode:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
```

A commented alternative (`/opt/conda/...`) sits directly above it. Adjust to your install.

## Running experiments

All experiments run through `main.py` against `config/tasks/sim/expert.yaml`. The scripts in
`scripts/sim/` sweep that config and collect metrics; there is no separate entry point.

| Script | What it does |
| --- | --- |
| `run_semantic_quick_test.sh` | A few `gna_top_k` values, semantic vs. random. Start here to check the pipeline. |
| `run_semantic_experiments.sh` | Main sweep: `gna_top_k` 0–5 across all six selection modes, 3 trials each. |
| `run_gna_top_k_experiments.sh` | `gna_top_k` 0–10, 5 trials each, reporting rule observability and informativeness. |
| `run_gna_top_k_quick_test.sh` | Shortened version of the above. |
| `run_full_experiments.sh` | Master orchestrator: sweeps field of view, agent density, vicinity radius, and rule set. |
| `run_experiment_worker.sh` | Runs one configuration through all modes; called by the orchestrator. |
| `run_metrics_verification.sh` | Sanity check that metrics vary across configurations. |
| `generate_agents.py` | Builds an agent YAML for a given car and pedestrian count. |
| `run_sim_easy.sh`, `run_sim_expert.sh` | Upstream's plain simulation runs, no communication. |

A typical first run:

```bash
bash scripts/sim/run_semantic_quick_test.sh     # validate
bash scripts/sim/run_semantic_experiments.sh    # full sweep
```

To sweep a different rule set, set `RULE_FILE`:

```bash
RULE_FILE=config/rules/sim/expert/expert_rule_spatial.yaml \
  bash scripts/sim/run_semantic_experiments.sh
```

`run_full_experiments.sh` takes no arguments — experiments are declared inside it with
`add_experiment`, labelled `fov{N}_{cars}c{peds}p_r{region}_{ruleset}`. Most are commented out
because they have already been run; uncomment the ones you want. Note that it edits
`AGENT_FOV` in `logicity/core/config.py` between groups and restores it at the end, so let it
finish cleanly. It runs up to 5 experiments in parallel (`MAX_PARALLEL`).

## Selection modes

Set by `gna_selection_mode` in the config, or swept by the scripts.

| Mode | Behavior |
| --- | --- |
| `semantic` | Goal-oriented selection (the proposed method) |
| `semantic_random` | Uniform random selection at the same budget (baseline) |
| `semantic_lna` | Local assistant variant; uplink sends the full field of view, downlink budget is `gna_top_k2` |
| `semantic_lna_random` | Random counterpart to the above |
| `semantic_lna_single` | Single-zone variant |
| `semantic_lna_random_single` | Random counterpart to the single-zone variant |

## Rule sets

Four rule files under `config/rules/sim/expert/`:

| File | Description |
| --- | --- |
| `expert_rule.yaml` | Upstream's original rule set |
| `expert_rule_extended.yaml` | More hypotheses |
| `expert_rule_spatial.yaml` | Emphasizes spatially-dependent rules |
| `expert_rule_discriminative.yaml` | Built to maximize entity diversity among hypotheses (default) |

## Configuration keys

The scripts patch these in `config/tasks/sim/expert.yaml`:

| Key | Meaning |
| --- | --- |
| `enable_gna` | Turns the Global Navigation Assistant on |
| `gna_selection_mode` | Selection strategy (table above) |
| `gna_top_k` | Number of FOL expressions transmitted |
| `gna_top_k1`, `gna_top_k2` | Uplink and downlink budgets for the LNA modes |
| `agent_region` | Vicinity radius |
| `agent_yaml_file` | Agent composition |
| `rule_yaml_file` | Rule set |
| `enable_sim_metrics` | Enables metric collection |

`AGENT_FOV` lives in `logicity/core/config.py`, not in the YAML.

> The sweep scripts modify `config/tasks/sim/expert.yaml` in place, keeping a `.backup`
> alongside it. If a run is interrupted, check that the config was restored before starting
> another.

## Outputs

Each script writes to its own timestamped directory: `semantic_experiments_<timestamp>/`,
`gna_top_k_results_<timestamp>/`, or `full_experiments_<timestamp>/` (which holds `configs/`,
`agents/`, and `logs/`). Raw simulation logs go to `log_sim/`.

Metrics are reported as mean ± standard deviation across trials:

- **Decision Success Rate**, at subrule and action level — agreement with a full-information
  baseline in which the agent observes its entire vicinity
- **Trajectory Success Rate**
- **Per-agent rule observability** — the share of rules fully, partially, and unobserved
- **Normalized informativeness** — how well an agent's information explicates the true world state

Results from the runs reported in the paper are in `full_experiments_merged/`.

## Citation

Please cite both the paper and the simulator it builds on:

```bibtex
@inproceedings{saz2026goaloriented,
  title     = {Goal-Oriented Semantic Communication for Logical Decision Making},
  author    = {Saz, Ahmet Faruk and Fekri, Faramarz},
  booktitle = {Proceedings of the IEEE Global Communications Conference (GLOBECOM)},
  year      = {2026}
}

@inproceedings{li2024logicity,
  title     = {{LogiCity}: Advancing Neuro-Symbolic AI with Abstract Urban Simulation},
  author    = {Li, Bowen and Li, Zhaoyu and Du, Qiwei and Luo, Jinqi and Wang, Wenshan
               and Xie, Yaqi and Stepputtis, Simon and Wang, Chen and Sycara, Katia P.
               and Ravikumar, Pradeep Kumar and Gray, Alex and Si, Xujie and Scherer, Sebastian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

## License

Inherited from upstream LogiCity — see [LICENSE](LICENSE).
