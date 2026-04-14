# Full Experiment Results (Merged)

## Overview

This directory contains merged results from 17 out of 18 planned experiment configurations, evaluating the impact of different GNA/LNA entity selection modes on agent decision-making in the LogiCity simulator. Results were collected across three separate runs (April 1–4, 2026) and consolidated here.

The missing experiment is `fov7_30c8p_r70_discriminative` (30 cars, FOV=7), which was excluded due to prohibitive runtime (~6 hours per trial in semantic mode with 30 cars).

## Simulation Parameters (Common)

| Parameter | Value |
|-----------|-------|
| Max steps per simulation | 100 |
| Trials per (mode, k) | 3 (seeds 0, 1, 2) |
| AGENT_VICINITY | 25 |
| WORLD_SIZE | 241 x 241 |
| k values swept | 0, 1, 2, 3, 4, 5 |

## Selection Modes (6 total)

| Mode | Description |
|------|-------------|
| `semantic` | GNA selects top-k entities from vicinity using inductive probability optimization |
| `semantic_random` | GNA selects k random entities from vicinity (baseline for semantic) |
| `semantic_lna` | Multi-zone LNA: cars uplink all FOV entities, LNA pools/re-grounds, semantic downlink top-k2 |
| `semantic_lna_random` | Same as semantic_lna but with random downlink selection (baseline for LNA) |
| `semantic_lna_single` | Single-zone LNA: same pipeline as semantic_lna but one global LNA covers the entire map |
| `semantic_lna_random_single` | Same as semantic_lna_single but with random downlink selection |

For `semantic` and `semantic_random`, the swept parameter is `gna_top_k` (k=0..5).
For all LNA modes, uplink transmits all FOV entities (k1 unused); the swept parameter is `gna_top_k2` (k2=0..5).

## Metrics

- **Subrule DSR (Decision Success Rate)**: Fraction of ego-gate-active subrule evaluations that match between baseline (full information) and test (limited by selection mode). Reported as mean +/- std across trials, plus raw correct/incorrect/total counts.
- **Action DSR**: Fraction of steps where the derived action (Stop/Slow/Fast/Normal) matches between baseline and test.
- **Trajectory SR (Success Rate)**: Fraction of finished trajectories where the agent reached its goal without any action-level rule violations. No step horizon is imposed.

For LNA modes, only car agents are evaluated (pedestrians do not participate in LNA communication).

## The 18 Experiments

### FOV=5 Group (4 experiments, all complete)

All use 20 cars, 8 pedestrians, agent_region=70.

| Experiment | Rule File |
|-----------|-----------|
| `fov5_20c8p_r70_original` | expert_rule.yaml (12 rules) |
| `fov5_20c8p_r70_extended` | expert_rule_extended.yaml (48 rules) |
| `fov5_20c8p_r70_spatial` | expert_rule_spatial.yaml (30 rules) |
| `fov5_20c8p_r70_discriminative` | expert_rule_discriminative.yaml (30 rules) |

### FOV=7 Group (10 experiments, 9 complete)

Default: 20 cars, 8 pedestrians, agent_region=70, discriminative rules — unless noted.

| Experiment | Variation | Status |
|-----------|-----------|--------|
| `fov7_20c8p_r70_original` | original rules | Complete |
| `fov7_20c8p_r70_extended` | extended rules | Complete |
| `fov7_20c8p_r70_spatial` | spatial rules | Complete |
| `fov7_20c8p_r70_discriminative` | discriminative rules | Complete |
| `fov7_20c4p_r70_discriminative` | 4 pedestrians | Complete |
| `fov7_20c12p_r70_discriminative` | 12 pedestrians | Complete |
| `fov7_10c8p_r70_discriminative` | 10 cars | Complete |
| `fov7_30c8p_r70_discriminative` | 30 cars | **Missing** (runtime prohibitive) |
| `fov7_20c8p_r50_discriminative` | agent_region=50 | Complete |
| `fov7_20c8p_r120_discriminative` | agent_region=120 | Complete |

### FOV=10 Group (4 experiments, all complete)

All use 20 cars, 8 pedestrians, agent_region=70.

| Experiment | Rule File |
|-----------|-----------|
| `fov10_20c8p_r70_original` | expert_rule.yaml (12 rules) |
| `fov10_20c8p_r70_extended` | expert_rule_extended.yaml (48 rules) |
| `fov10_20c8p_r70_spatial` | expert_rule_spatial.yaml (30 rules) |
| `fov10_20c8p_r70_discriminative` | expert_rule_discriminative.yaml (30 rules) |

## File Structure

```
full_experiments_merged/
├── README.md                              # This file
├── summary.csv                            # Machine-readable: 612 rows (17 experiments x 36 mode-k combos)
├── summary.txt                            # Formatted tables for all experiments
├── fov5_20c8p_r70_original_results.txt    # Per-experiment result files (one line per mode-k combo)
├── fov5_20c8p_r70_extended_results.txt
├── ...
└── fov10_20c8p_r70_discriminative_results.txt
```

### Result file format (per-experiment `*_results.txt`)

Each line: `label mode k sr_mean sr_std sr_correct sr_incorrect sr_total act_mean act_std act_correct act_incorrect act_total traj_mean traj_std traj_total traj_success traj_violated traj_incomplete`

### CSV columns

`label, mode, k, sr_mean, sr_std, sr_correct, sr_incorrect, sr_total, act_mean, act_std, act_correct, act_incorrect, act_total, traj_mean, traj_std, traj_total, traj_success, traj_violated, traj_incomplete`

## Rule Files

| File | Rules | Description |
|------|-------|-------------|
| `expert_rule.yaml` | 12 (4S/4Sl/4F) | Original rules, many ego-gated |
| `expert_rule_extended.yaml` | 48 (16S/16Sl/16F) | Extended with more diverse subrules |
| `expert_rule_spatial.yaml` | 30 (10S/10Sl/10F) | Emphasis on spatial predicates, fewer ego gates |
| `expert_rule_discriminative.yaml` | 30 (10S/10Sl/10F) | Non-ego-gated, type-discriminative, designed to maximize semantic vs. random differentiation |

## Source Runs

| Directory | Date | Experiments completed |
|-----------|------|----------------------|
| `full_experiments_20260401_210122` | Apr 1–2 | 7 (all FOV=5 + 3 FOV=7) |
| `full_experiments_20260403_013449` | Apr 3 | 6 complete + 1 partial (remaining FOV=7) |
| `full_experiments_20260403_225550` | Apr 3–4 | 4 (all FOV=10) |
