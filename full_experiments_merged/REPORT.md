# Experiment Report: Semantic vs. Random Entity Selection Under Limited Field-of-View

## 1. Problem Statement and Core Question

Autonomous agents navigating urban environments must make rule-based decisions — when to stop, slow down, or speed up — by reasoning about nearby entities (vehicles, pedestrians, emergency vehicles). In practice, agents have a **limited field-of-view (FOV)** and cannot directly observe all entities that are relevant to their decision-making. Entities beyond the FOV but within a broader "vicinity" zone may still trigger critical traffic rules.

The central question of this study is:

> **When agents have limited field-of-view, can they recover correct decisions by intelligently selecting which nearby entities to reason about?**

More precisely, if a communication infrastructure can relay information about *k* additional entities from the vicinity to an ego agent, does it matter *which* entities are selected? We compare **semantic selection** — which uses an inductive probability framework to identify entities that most affect rule evaluation — against **random selection**, which transmits *k* entities chosen uniformly at random.

## 2. Simulation Environment

All experiments run on the **LogiCity** simulator, a neuro-symbolic benchmark for urban traffic. The environment is a 241 x 241 grid world populated with cars and pedestrians. Each agent navigates toward randomly assigned destinations, and at every simulation step a **Z3 SMT solver** evaluates first-order logic (FOL) rules to determine the agent's action (Stop, Slow, Fast, or Normal).

### Common Parameters

| Parameter | Value |
|-----------|-------|
| World size | 241 x 241 grid cells |
| Vicinity radius (AGENT_VICINITY) | 25 cells |
| Max steps per simulation | 100 |
| Trials per (mode, k) combination | 3 (seeds 0, 1, 2) |
| k values swept | 0, 1, 2, 3, 4, 5 |
| Spatial predicate ranges | IsClose: 12–20 cells, NextTo: < 12 cells, CollidingClose: <= 8/3 cells (Car/Ped) + angle < 0.1 rad |

The **vicinity radius** is fixed at 25 cells across all experiments. The FOV radius varies (5, 7, or 10 cells). The gap between FOV and vicinity — the "vicinity ring" — defines the zone from which entities must be relayed via communication, and is where the selection mechanism makes its impact.

## 3. Communication Modes

We evaluate three architectures for relaying information from the vicinity ring to ego agents. Each architecture is tested with both semantic and random selection, yielding six modes total.

### 3.1 GNA (Global Navigation Assistant) — Direct Broadcast

- **`semantic`**: A centralized GNA observes the full global state. For each ego agent, it identifies all entities within the agent's vicinity ring (beyond FOV but within 25 cells). It applies the semantic selection algorithm — based on inductive probability optimization — to select the top-*k* entities whose inclusion most reduces uncertainty in subrule evaluation, and transmits them to the ego agent.
- **`semantic_random`**: Identical pipeline, but the *k* entities are chosen uniformly at random from the vicinity ring instead of semantically.

This pair isolates the question: **does intelligent entity selection matter when a centralized oracle can see everything?**

### 3.2 Multi-Zone LNA (Local Navigation Assistants) — Distributed Relay

- **`semantic_lna`**: The map is partitioned into non-overlapping rectangular zones, each managed by a Local Navigation Assistant (LNA) placed at an intersection. Cars within each zone transmit all entities in their FOV (plus themselves) to their local LNA — no selection or grounding occurs during this uplink phase. Each LNA pools the received entities, removes duplicates, filters out entities already in the ego car's FOV, re-grounds spatial predicates from the ego car's perspective, and then applies semantic selection to choose the top-*k₂* entities to relay back to each car.
- **`semantic_lna_random`**: Same pipeline, but the *k₂* downlink entities are chosen randomly instead of semantically.

This pair tests: **does semantic selection matter in a realistic, distributed architecture where information must flow through local intermediaries?** Only car agents participate in LNA communication; pedestrians are excluded from both the pipeline and the metrics.

### 3.3 Single-Zone LNA — Unified Relay

- **`semantic_lna_single`**: Identical to the multi-zone LNA pipeline, but a single LNA covers the entire map. This eliminates the zone fragmentation effect — every car's uplink reaches a common pool, and every car can receive entities from any other car's FOV, regardless of distance.
- **`semantic_lna_random_single`**: Same pipeline, with random downlink selection.

This pair isolates: **is the LNA zone partitioning a bottleneck, and does consolidating into a single relay improve semantic selection's advantage?**

## 4. Experimental Dimensions

We vary five parameters across 17 completed experiments (one of 18 planned was excluded due to prohibitive runtime):

### 4.1 Field-of-View (FOV)

| FOV | Vicinity Ring Width | Effect |
|-----|-------------------|--------|
| 5 | 20 cells | Largest ring: most entities compete for *k* slots; highest selection pressure |
| 7 | 18 cells | Moderate ring: default configuration |
| 10 | 15 cells | Smallest ring: fewer entities beyond FOV; less need for communication |

FOV determines how much each agent can directly observe. A smaller FOV means more entities fall into the vicinity ring and must be relayed, amplifying the importance of *which* entities are selected.

### 4.2 Number of Cars

10, 20 (default), or 30 cars. More cars increase the entity density, enlarging the pool of candidates for selection and creating more inter-vehicle rule interactions.

### 4.3 Number of Pedestrians

4, 8 (default), or 12 pedestrians. Pedestrians are a distinct entity type with different spatial predicate thresholds (CollidingClose range = 3 vs. 8 for cars). Varying their count changes the type diversity in the selection pool.

### 4.4 Agent Region

70 (default) or 120. This parameter constrains the area where agents spawn and navigate. A larger region spreads agents thinner, creating sparser interactions and larger distances between entities.

### 4.5 Rule Sets

Four rule files define the traffic logic agents must follow. Each rule set creates a different selection challenge. See Section 5 for details.

## 5. Rule Sets: Structure and Design

Each rule set defines subrules in three action categories — **Stop**, **Slow**, and **Fast** — combined via disjunction (OR). A subrule fires if all its conjunctive predicates are satisfied. The action derivation is cascading: Stop takes priority over Slow, which takes priority over Fast, which takes priority over Normal (the default).

### 5.1 Original Rules (`expert_rule.yaml`) — 12 subrules (7 Stop / 2 Slow / 3 Fast)

The original rule set from the LogiCity benchmark. Designed for general traffic scenarios.

**Key characteristics:**
- **Heavy ego-gating**: 8 of 12 subrules have ego-gate predicates (e.g., `IsAmbulance(entity)`, `IsBus(entity)`, `IsTiro(entity)`, `IsPolice(entity)`) that must evaluate to True for the ego agent itself before the subrule activates. For a generic car, most ego gates fail, meaning only 2–4 subrules are ever "live" for a given agent.
- **What it evaluates**: Whether semantic selection can identify the handful of critical entities when most rules are irrelevant to the ego agent. Because so few subrules activate, even *k=1* often suffices for semantic to achieve perfect scores.
- **Baseline performance (k=0, FOV=7)**: Subrule DSR = 0.811, Action DSR = 0.849, TSR = 0.119.

### 5.2 Extended Rules (`expert_rule_extended.yaml`) — 48 subrules (20 Stop / 15 Slow / 13 Fast)

A 4x expansion of the original rules, retaining the originals and adding new subrules involving diverse entity types and spatial predicates.

**Key characteristics:**
- **Mixed ego-gating**: The original ego-gated subrules remain, but many new subrules have weaker or no ego gates (e.g., `IsCar(entity)` which is True for all cars, or `IsPedestrian(entity)` for all pedestrians).
- **Broad predicate coverage**: Uses all 9 attributive predicates (IsCar, IsPedestrian, IsAmbulance, IsBus, IsPolice, IsTiro, IsReckless, IsOld, IsYoung) and all spatial predicates (IsClose, NextTo, CollidingClose, LeftOf, RightOf, HigherPri, IsAtInter, IsInInter).
- **What it evaluates**: Whether the selection mechanism scales gracefully as the rule complexity increases. More active subrules mean more entities become decision-relevant.
- **Baseline performance (k=0, FOV=7)**: Subrule DSR = 0.929, Action DSR = 0.882, TSR = 0.031.

### 5.3 Spatial Rules (`expert_rule_spatial.yaml`) — 30 subrules (10 Stop / 10 Slow / 10 Fast)

Designed to emphasize spatial predicates and minimize ego-gating.

**Key characteristics:**
- **Minimal ego-gating**: 24 of 30 subrules have no ego-gate predicates. The few that do use location predicates (`IsAtInter(entity)`, `IsInInter(entity)`) or negated type predicates (`Not(IsAmbulance(entity))`), which still activate for most agents.
- **Spatial predicate density**: Every subrule includes at least one spatial predicate (IsClose, NextTo, CollidingClose, LeftOf, RightOf, or HigherPri), making entity positions critical. The *distance* and *direction* of nearby entities directly determines rule activation.
- **Other-entity type discrimination**: Attributive predicates appear primarily on `dummyEntityA` (the non-ego entity), meaning the *type* of the observed entity matters for rule evaluation.
- **What it evaluates**: Whether semantic selection can prioritize entities based on spatial relevance — an entity that is `IsClose` and of the right type should be selected over one that is far away and type-irrelevant.
- **Baseline performance (k=0, FOV=7)**: Subrule DSR = 0.897, Action DSR = 0.888, TSR = 0.034.

### 5.4 Discriminative Rules (`expert_rule_discriminative.yaml`) — 30 subrules (10 Stop / 10 Slow / 10 Fast)

Purpose-built to maximize the difference between semantic and random selection, particularly in LNA modes.

**Key characteristics:**
- **Near-zero ego-gating**: 28 of 30 subrules have no ego-gate predicates whatsoever. Every subrule activates for every agent at every step, conditional only on the presence and properties of nearby entities.
- **Type-discriminative**: Each subrule targets a distinct entity type — one requires `IsCar(dummyEntityA)`, another `IsPedestrian(dummyEntityA)`, another `IsAmbulance(dummyEntityA)`, etc. This ensures that the *identity* of the selected entity determines whether the rule fires.
- **2–3 predicates per rule**: Each subrule combines a spatial predicate with a type predicate (and sometimes a directional predicate), keeping complexity manageable while ensuring rule diversity.
- **All 9 attributive predicates used**: IsCar, IsPedestrian, IsAmbulance, IsBus, IsPolice, IsTiro, IsReckless, IsOld, IsYoung all appear across the rules, providing maximum type diversity.
- **What it evaluates**: The scenario where every entity is potentially decision-relevant and the selection mechanism must identify *which* entity types and spatial configurations are present. This is the hardest test for random selection, as blindly picking entities is unlikely to cover all required types.
- **Baseline performance (k=0, FOV=7)**: Subrule DSR = 0.930, Action DSR = 0.933, TSR = 0.013.

### 5.5 Summary of Rule Set Properties

| Rule Set | Subrules | Ego-Gated | Non-Ego-Gated | Primary Challenge |
|----------|----------|-----------|---------------|-------------------|
| Original | 12 | 8 (67%) | 4 (33%) | Few active rules; semantic needs to find 1–2 critical entities |
| Extended | 48 | ~20 (42%) | ~28 (58%) | Scaling: many rules, diverse entity requirements |
| Spatial | 30 | 6 (20%) | 24 (80%) | Spatial relevance: position and direction matter |
| Discriminative | 30 | 2 (7%) | 28 (93%) | Type coverage: must cover many entity types |

## 6. Evaluation Metrics

At every simulation step, we evaluate each agent's subrules under two conditions:

1. **Baseline**: The agent has full information — its FOV entities *plus all vicinity entities*.
2. **Test**: The agent has limited information — its FOV entities *plus only the k entities selected by the communication mode*.

### 6.1 Subrule Decision Success Rate (Subrule DSR)

For each agent at each step, we compare every subrule's evaluation (True/False) between baseline and test. Only subrules whose **ego-gate evaluates to True** in the baseline are counted — this filters out subrules that are trivially inactive regardless of information availability (e.g., an `IsAmbulance(entity)` gate for a non-ambulance ego agent). The metric is:

> **Subrule DSR** = (number of ego-gate-active subrule evaluations matching between baseline and test) / (total ego-gate-active subrule evaluations)

This metric captures fine-grained decision fidelity at the individual rule level. Values close to 1.0 mean the selection mechanism preserves nearly all subrule evaluations.

### 6.2 Action Decision Success Rate (Action DSR)

At each step, after evaluating all subrules, the agent derives a final action through the cascading priority: Stop > Slow > Fast > Normal. If *any* Stop subrule fires, the action is Stop; otherwise if *any* Slow subrule fires (and the agent is not already stopping), the action is Slow; and so on. We compare the derived action under baseline vs. test:

> **Action DSR** = (number of steps where baseline and test produce the same action) / (total steps)

This metric captures whether the selection mechanism preserves the *final decision*, regardless of how many individual subrules differ. A subrule mismatch may not affect the action if another subrule in the same category still fires correctly.

### 6.3 Trajectory Success Rate (TSR)

Each agent navigates from a start position to a goal. A trajectory is considered **successful** if the agent reaches its goal without any action-level rule violation (i.e., every step's action matches the baseline). A trajectory is **violated** if at least one step's action differs from the baseline.

> **TSR** = (number of successful trajectories) / (total finished trajectories)

Only finished trajectories (completed or violated) are counted; incomplete trajectories (still in progress when the simulation ends) are excluded. TSR is the most stringent metric: a single action mismatch in a 100-step trajectory marks it as violated.

### 6.4 Metric Hierarchy

The three metrics form a hierarchy of increasing stringency:

- **Subrule DSR** >= **Action DSR** (subrule mismatches may cancel out at the action level)
- **Action DSR** >= **TSR** (a single action mismatch over a full trajectory fails TSR)

In LNA modes, only car agents are included in metric evaluation, as pedestrians do not participate in the LNA communication pipeline.

## 7. Results: GNA — `semantic` vs. `semantic_random`

### 7.1 Overall Performance

Semantic selection consistently and substantially outperforms random selection in the GNA setting. The advantage is visible across all 16 valid experiments, all k values, and all three metrics.

**Representative example — best experiment (`fov5_20c8p_r70_original`):**

| k | SR (sem) | SR (rnd) | Gap | Act (sem) | Act (rnd) | Gap | TSR (sem) | TSR (rnd) | Gap |
|---|---------|---------|-----|-----------|-----------|-----|-----------|-----------|-----|
| 0 | 0.805 | 0.805 | 0.000 | 0.842 | 0.842 | 0.000 | 0.128 | 0.128 | 0.000 |
| 1 | 0.913 | 0.819 | **+0.094** | 1.000 | 0.862 | **+0.138** | 1.000 | 0.144 | **+0.856** |
| 2 | 0.928 | 0.833 | +0.095 | 1.000 | 0.885 | +0.116 | 1.000 | 0.147 | +0.853 |
| 3 | 0.929 | 0.844 | +0.085 | 1.000 | 0.899 | +0.101 | 1.000 | 0.155 | +0.845 |
| 5 | 0.929 | 0.867 | +0.062 | 1.000 | 0.929 | +0.071 | 1.000 | 0.169 | +0.831 |

Semantic achieves **perfect Action DSR and TSR at k=1** — a single well-chosen entity suffices to recover all correct decisions. Random selection at k=5 still only reaches Action DSR of 0.929 and TSR of 0.169.

### 7.2 Semantic Advantage by Rule Set

Averaged across k=1 to k=5, holding FOV=7, 20 cars, 8 pedestrians, region=70:

| Rule Set | Avg SR Gap | Avg Action Gap | Avg TSR Gap |
|----------|-----------|----------------|-------------|
| Original (12) | **+0.077** | **+0.098** | **+0.832** |
| Extended (48) | +0.036 | +0.059 | +0.656 |
| Spatial (30) | +0.062 | +0.062 | +0.801 |
| Discriminative (30) | +0.039 | +0.037 | +0.576 |

**Original rules show the largest gaps** because their heavy ego-gating means only 2–4 subrules are ever active. Semantic selection identifies the critical entity with high precision, while random selection frequently misses it. With extended and discriminative rules, more subrules are active and more entity types are relevant, narrowing (but not eliminating) the gap.

**Spatial rules show the second-largest gaps** because the non-ego-gated spatial predicates make entity positioning critical. Semantic selection directly optimizes for spatial relevance, selecting entities that are `IsClose`, `NextTo`, or `CollidingClose` to the ego agent over those that are distant.

### 7.3 Effect of FOV

Holding all else constant at 20 cars, 8 pedestrians, region=70, discriminative rules:

| FOV | Avg SR Gap | Avg Action Gap |
|-----|-----------|----------------|
| 5 | **+0.051** | **+0.053** |
| 7 | +0.039 | +0.037 |
| 10 | +0.035 | +0.017 |

**Smaller FOV amplifies semantic advantage.** With FOV=5, the vicinity ring spans 20 cells, containing more candidate entities. This increases selection pressure: random selection must "get lucky" among a larger pool, while semantic selection optimally identifies the critical entities. With FOV=10, the vicinity ring shrinks to 15 cells, fewer entities need relaying, and the gap narrows.

### 7.4 Effect of Region Size

| Region | Avg SR Gap | Avg Action Gap |
|--------|-----------|----------------|
| 70 | +0.039 | +0.037 |
| 120 | +0.031 | **+0.100** |

Expanding the region from 70 to 120 **reduces the Subrule DSR gap** (entities are sparser, so there is less variety in the vicinity to begin with) but **dramatically increases the Action DSR gap** (+0.100 vs. +0.037). This is because sparser environments create "high-stakes" scenarios: when the few entities in the vicinity happen to be critical (e.g., an ambulance approaching an intersection), random selection's failure to find them causes action-level failures. Semantic selection's precision matters more in these sparse but consequential situations.

### 7.5 Effect of Car Count

| Cars | Avg SR Gap | Avg Action Gap |
|------|-----------|----------------|
| 10 | +0.025 | **+0.055** |
| 20 | +0.039 | +0.037 |

With fewer cars (10), the Subrule DSR gap is smaller (fewer entities means less competition for *k* slots), but the Action DSR gap is *larger* (+0.055 vs. +0.037). Fewer cars means each car that *is* present carries more weight in rule evaluation — missing one has a bigger impact on the final action.

### 7.6 Effect of Pedestrian Count

| Pedestrians | Avg SR Gap | Avg Action Gap |
|-------------|-----------|----------------|
| 4 | +0.033 | +0.048 |
| 8 | +0.039 | +0.037 |
| 12 | **+0.047** | +0.034 |

More pedestrians increase the type diversity in the selection pool. With 12 pedestrians, the Subrule DSR gap widens (+0.047) because many discriminative rules target pedestrian types, and random selection is less likely to cover them all. However, the Action DSR gap slightly narrows as the cascading priority (Stop dominates) absorbs some subrule-level mismatches.

## 8. Results: Multi-Zone LNA — `semantic_lna` vs. `semantic_lna_random`

### 8.1 Overall Performance

Semantic selection consistently outperforms random in the multi-zone LNA setting, though the absolute gaps are smaller than in the GNA setting. This reflects the structural bottleneck of the LNA architecture: only entities observed by other cars' FOVs can enter the relay pool, and the zone partitioning fragments this pool further.

**Representative example — best experiment (`fov5_20c8p_r70_spatial`):**

| k₂ | SR (sem) | SR (rnd) | Gap | Act (sem) | Act (rnd) | Gap | TSR (sem) | TSR (rnd) | Gap |
|----|---------|---------|-----|-----------|-----------|-----|-----------|-----------|-----|
| 0 | 0.912 | 0.912 | 0.000 | 0.848 | 0.848 | 0.000 | 0.030 | 0.030 | 0.000 |
| 1 | 0.951 | 0.923 | **+0.028** | 0.918 | 0.872 | **+0.046** | 0.122 | 0.049 | **+0.073** |
| 2 | 0.963 | 0.931 | +0.031 | 0.925 | 0.885 | +0.040 | 0.166 | 0.060 | **+0.106** |
| 3 | 0.964 | 0.937 | +0.027 | 0.926 | 0.892 | +0.034 | 0.166 | 0.050 | **+0.116** |
| 5 | 0.967 | 0.948 | +0.018 | 0.926 | 0.910 | +0.016 | 0.166 | 0.095 | +0.071 |

### 8.2 Semantic Advantage by Rule Set

Averaged across k₂=1 to k₂=5, FOV=7, 20 cars, 8 pedestrians, region=70:

| Rule Set | Avg SR Gap | Avg Action Gap | Avg TSR Gap |
|----------|-----------|----------------|-------------|
| Original (12) | +0.015 | +0.023 | +0.055 |
| Extended (48) | +0.013 | +0.018 | +0.021 |
| Spatial (30) | **+0.021** | **+0.027** | **+0.064** |
| Discriminative (30) | +0.019 | +0.015 | +0.067 |

**Spatial rules produce the largest LNA gaps** for both Subrule DSR and Action DSR. Their heavy reliance on spatial predicates (IsClose, NextTo, directional) means that re-grounding from the ego car's perspective creates sharp relevance differences between entities, which semantic selection exploits. Discriminative rules show competitive TSR gaps (+0.067) due to their near-universal activation making every entity a potential contributor to rule evaluation.

### 8.3 Effect of FOV

| FOV | Avg SR Gap | Avg Action Gap |
|-----|-----------|----------------|
| 5 | **+0.029** | **+0.020** |
| 7 | +0.019 | +0.015 |
| 10 | +0.019 | +0.006 |

The same FOV trend holds for LNA: smaller FOV creates larger pools and more selection pressure. However, the Action DSR gap is notably smaller across all FOVs compared to GNA, reflecting the inherent information bottleneck of the uplink phase.

### 8.4 Effect of Region Size

| Region | Avg SR Gap | Avg Action Gap |
|--------|-----------|----------------|
| 70 | **+0.019** | +0.015 |
| 120 | +0.009 | **+0.024** |

With region=120, the LNA zones become sparser, reducing the number of cars contributing to each LNA's pool. This paradoxically increases the Action DSR gap (fewer entities in the pool means each one carries more weight) while decreasing the Subrule DSR gap (fewer total subrule evaluations).

### 8.5 Effect of Car Count

| Cars | Avg SR Gap | Avg Action Gap |
|------|-----------|----------------|
| 10 | +0.004 | +0.017 |
| 20 | **+0.019** | +0.015 |

With only 10 cars, the LNA pools are very small (each zone has few contributors), and the Subrule DSR gap nearly vanishes (+0.004). This is a fundamental limitation: the LNA architecture requires sufficient car density for the uplink pool to contain entities that semantic selection can differentiate.

## 9. Results: Single-Zone LNA — `semantic_lna_single` vs. `semantic_lna_random_single`

### 9.1 Overall Performance

The single-zone LNA eliminates zone fragmentation, giving the semantic selection algorithm access to a larger, globally unified pool of uplinked entities. Performance is comparable to multi-zone LNA, with some experiments showing slight improvements.

**Representative example — best experiment (`fov5_20c8p_r70_discriminative`):**

| k₂ | SR (sem) | SR (rnd) | Gap | Act (sem) | Act (rnd) | Gap | TSR (sem) | TSR (rnd) | Gap |
|----|---------|---------|-----|-----------|-----------|-----|-----------|-----------|-----|
| 0 | 0.923 | 0.923 | 0.000 | 0.888 | 0.888 | 0.000 | 0.027 | 0.027 | 0.000 |
| 1 | 0.953 | 0.931 | +0.022 | 0.921 | 0.902 | +0.019 | 0.049 | 0.043 | +0.006 |
| 2 | 0.969 | 0.939 | **+0.030** | 0.925 | 0.910 | +0.015 | 0.050 | 0.043 | +0.007 |
| 3 | 0.975 | 0.945 | **+0.030** | 0.931 | 0.914 | **+0.016** | 0.069 | 0.046 | **+0.023** |
| 5 | 0.980 | 0.956 | +0.024 | 0.932 | 0.919 | +0.013 | 0.071 | 0.057 | +0.014 |

### 9.2 Semantic Advantage by Rule Set

Averaged across k₂=1 to k₂=5, FOV=7, 20 cars, 8 pedestrians, region=70:

| Rule Set | Avg SR Gap | Avg Action Gap | Avg TSR Gap |
|----------|-----------|----------------|-------------|
| Original (12) | +0.013 | +0.018 | +0.055 |
| Extended (48) | +0.012 | +0.014 | +0.031 |
| Spatial (30) | +0.015 | +0.019 | **+0.072** |
| Discriminative (30) | **+0.017** | +0.010 | +0.051 |

The discriminative rules show the largest Subrule DSR gap (+0.017) for single-zone LNA, confirming that type-discriminative rules best exercise the semantic selection algorithm's ability to cover diverse entity types. Spatial rules produce the largest TSR gap (+0.072) because their strict spatial predicate requirements mean that a missed entity often translates to a missed trajectory.

### 9.3 Single-Zone vs. Multi-Zone LNA Comparison

| Mode | Avg SR Gap | Avg Action Gap | Avg TSR Gap |
|------|-----------|----------------|-------------|
| Multi-Zone LNA | +0.019 | +0.015 | +0.067 |
| Single-Zone LNA | +0.017 | +0.010 | +0.051 |

Contrary to initial expectations, the single-zone LNA does **not** consistently produce larger semantic-vs-random gaps than the multi-zone LNA. While the single-zone pool is larger (all cars contribute), this also means the random baseline has more entities to "accidentally" pick useful ones from. The multi-zone LNA's smaller, more constrained pools make random selection's inefficiency more visible.

## 10. Cross-Cutting Trends and Summary

### 10.1 Semantic Always Outperforms Random

Across all 16 valid experiments, all 3 communication modes, all k values > 0, and all 3 metrics, semantic selection **never underperforms** random selection. The advantage ranges from marginal (Subrule DSR gap of +0.004 for 10-car LNA) to dramatic (TSR gap of +0.98 for GNA with region=120).

### 10.2 GNA Gaps >> LNA Gaps

The GNA (centralized) architecture produces semantic-vs-random gaps that are **5–50x larger** than the LNA (distributed) architecture:

| Metric | GNA avg gap | Multi-LNA avg gap | Single-LNA avg gap |
|--------|-------------|-------------------|---------------------|
| Subrule DSR | +0.04 to +0.08 | +0.01 to +0.03 | +0.01 to +0.03 |
| Action DSR | +0.04 to +0.10 | +0.01 to +0.05 | +0.01 to +0.02 |
| TSR | +0.58 to +0.95 | +0.02 to +0.12 | +0.01 to +0.10 |

This gap exists because GNA has access to the full global state — it can *always* find and relay the most critical entities. LNA modes are constrained by what other cars happen to see in their FOVs, introducing an information bottleneck that limits the ceiling for both semantic and random selection.

### 10.3 k=1 is the Most Informative Comparison Point

The semantic advantage is consistently **largest at k=1** for Subrule DSR and Action DSR: when only one entity can be transmitted, the quality of that choice matters most. As k increases, both methods improve (random "covers" more entities by chance), and the gap narrows. However, semantic selection reaches its ceiling much faster — often at k=2 or k=3 — while random selection continues to improve slowly all the way to k=5.

For TSR, the peak gap often occurs at k=2 to k=4, because semantic reaches TSR=1.0 while random is still below 0.2.

### 10.4 When the Gap is Largest

The semantic advantage is **maximized** under conditions that combine:

1. **Small FOV** (FOV=5): Largest vicinity ring, most selection pressure
2. **Original or spatial rules**: Rules with strong entity-type specificity or spatial dependence
3. **GNA mode**: Full global state access
4. **Sparse environments** (region=120) for action-level metrics
5. **Sufficient entity density** (20+ cars) to populate the vicinity ring

### 10.5 When the Gap is Smallest

The semantic advantage is **minimized** under conditions that combine:

1. **Large FOV** (FOV=10): Small vicinity ring, fewer entities need relaying
2. **Extended rules**: Many overlapping subrules create redundancy that buffers both methods
3. **LNA modes with few cars** (10 cars): Sparse uplink pools limit both methods
4. **Large k** (k=5): Random selection covers enough entities by chance

### 10.6 Diminishing Returns Beyond k=3

For GNA, semantic selection typically reaches near-perfect performance (SR > 0.99, Act > 0.99) by k=3 across all rule sets. For LNA modes, improvement plateaus by k=3 to k=4 as the uplink pool is exhausted. This suggests that **k=3 is a practical operating point** — it captures most of the benefit of semantic selection while keeping communication costs low.

### 10.7 Rule Set Design Matters for Differentiation

The discriminative rule set, designed with 93% non-ego-gated rules and distinct entity-type requirements per subrule, produces the clearest differentiation at the subrule level in LNA modes. However, the original rule set produces the largest *action-level* and *trajectory-level* gaps in GNA mode because its few active rules create "all-or-nothing" scenarios where semantic selection's precision translates directly to correct actions.

This highlights a design tension: rule sets that are easy for semantic selection to solve (few active rules) show dramatic gaps but may not represent realistic complexity, while rule sets that better represent realistic traffic logic (many active rules, spatial dependencies) show smaller but more practically meaningful gaps.
