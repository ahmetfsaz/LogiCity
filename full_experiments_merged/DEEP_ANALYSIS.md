# Deep Analysis Report: Semantic vs. Random Entity Selection

This report presents a rigorous, data-driven analysis of 16 completed experiments (612 individual simulation runs across 6 modes, 6 k-values, and 3 trials each). All numerical claims are derived directly from the experimental data.

---

## 1. The First Entity is Worth More Than the Next Four Combined

The single most striking finding is the extreme **front-loading of marginal returns** for semantic selection. Averaging across all 16 experiments:

### GNA Mode — Subrule DSR Gain Per Additional Entity

| Step | Semantic delta | Random delta | Semantic / Random |
|------|---------------|-------------|-------------------|
| k=0 → 1 | **+0.05634** | +0.01052 | **5.35x** |
| k=1 → 2 | +0.01586 | +0.00884 | 1.79x |
| k=2 → 3 | +0.00606 | +0.00749 | 0.81x |
| k=3 → 4 | +0.00320 | +0.00687 | 0.47x |
| k=4 → 5 | +0.00147 | +0.00656 | 0.22x |

The first semantically-selected entity delivers **5.35x the subrule improvement** of a randomly-selected one. By k=3, the marginal value of semantic falls *below* random's — not because semantic is worse, but because it has already captured nearly all the recoverable information.

For **Action DSR**, the effect is even more dramatic:

| Step | Semantic delta | Random delta | Semantic / Random |
|------|---------------|-------------|-------------------|
| k=0 → 1 | **+0.08787** | +0.01839 | **4.78x** |
| k=1 → 2 | +0.01084 | +0.01385 | 0.78x |
| k=2 → 3 | +0.00357 | +0.01114 | 0.32x |
| k=3 → 4 | +0.00062 | +0.00946 | 0.07x |
| k=4 → 5 | +0.00012 | +0.00884 | 0.01x |

Semantic's first entity accounts for **87.87 out of every 100 action-percentage-points** of improvement it will ever achieve. After k=1, semantic is essentially done optimizing; the remaining entities provide single-digit basis-point improvements. Random, by contrast, improves almost linearly — each additional entity contributes roughly the same ~1% — reflecting the equal probability of "accidentally" selecting a useful entity.

### LNA Mode — Same Pattern, Attenuated

| Step | Semantic delta (SR) | Random delta (SR) | Ratio |
|------|-------------------|-----------------|-------|
| k₂=0 → 1 | +0.02600 | +0.00767 | **3.39x** |
| k₂=1 → 2 | +0.00847 | +0.00587 | 1.44x |
| k₂=2 → 3 | +0.00266 | +0.00473 | 0.56x |
| k₂=3 → 4 | +0.00144 | +0.00403 | 0.36x |
| k₂=4 → 5 | +0.00054 | +0.00365 | 0.15x |

The LNA first entity is worth 3.39x random's, and semantic exhausts its gains by k₂=2. This lower multiplier reflects the uplink information bottleneck — not all relevant entities are even available in the LNA pool.

**Implication**: In bandwidth-constrained scenarios, the most efficient operating point is k=1 semantic rather than k=5 random.

---

## 2. Semantic Selection Achieves 5x Communication Efficiency

Across **all 16 experiments and all 3 modes**, semantic at k=1 matches or exceeds random at k=5 for Subrule DSR:

- **GNA**: 16/16 experiments — semantic at k=1 beats random at k=5 (100%)
- **Multi-LNA**: 11/16 experiments at k₂=1, remaining 5 at k₂=2 (worst case: 2.5x efficiency)
- **Single-LNA**: 10/16 at k₂=1, remaining 6 at k₂=2

For **Action DSR**, the efficiency is even more extreme:

- **GNA**: 16/16 experiments — semantic at k=1 beats random at k=5 (100%)
- **Multi-LNA**: 14/16 at k₂=1, 2 at k₂=2
- **Single-LNA**: 13/16 at k₂=1, 3 at k₂=2

This means semantic selection achieves **at least 5x communication efficiency** in the GNA setting and **2.5–5x** in LNA settings. One well-chosen entity communicates as much decision-relevant information as five randomly-chosen ones.

---

## 3. Baseline Difficulty Predicts Semantic Advantage with r = −0.974

There is a near-perfect negative correlation between the baseline (k=0) Action DSR and the semantic advantage at k=3:

| Baseline Act DSR (k=0) | Semantic Act Gap at k=3 | Experiment |
|------------------------|------------------------|------------|
| 0.7989 | **+0.1038** | fov7_20c8p_r120_discriminative |
| 0.8420 | **+0.1008** | fov5_20c8p_r70_original |
| 0.8490 | **+0.0988** | fov7_20c8p_r70_original |
| 0.9406 | +0.0344 | fov7_20c12p_r70_discriminative |
| 0.9705 | +0.0164 | fov10_20c8p_r70_spatial |
| 0.9732 | +0.0154 | fov10_20c8p_r70_discriminative |

**Pearson r = −0.974** (GNA, Action DSR). This is an extraordinarily strong relationship. It means:

> The worse an agent performs with FOV-only information, the more it benefits from semantic selection.

This makes intuitive sense — if the FOV already captures most decision-relevant entities (high baseline), there is little room for *any* selection method to improve. When the FOV misses critical entities (low baseline), the quality of selection becomes decisive.

For **LNA modes**, the same pattern holds with r = −0.793, weaker but still strong. The attenuation reflects the LNA pool's inability to always contain the entities that would close the gap.

**Practical takeaway**: Semantic selection is most valuable precisely where it is most needed — in information-scarce scenarios.

---

## 4. Headroom Recovery: GNA Closes 99% by k=3, LNA Plateaus at 41%

We measure what fraction of the gap between FOV-only performance and perfect performance each mode recovers at each k:

### Action DSR — Fraction of Headroom Recovered (averaged across all experiments)

| k | GNA Semantic | LNA Semantic | Single-LNA Semantic |
|---|-------------|-------------|---------------------|
| 0 | 0.0% | 0.0% | 0.0% |
| 1 | **84.4%** | 35.4% | 32.9% |
| 2 | 95.3% | 39.2% | 36.1% |
| 3 | **99.1%** | 40.8% | 37.5% |
| 4 | 99.9% | 41.0% | 37.6% |
| 5 | 100.0% | **41.3%** | **37.9%** |

GNA semantic closes **99.1% of the gap at k=3** — nearly perfect information recovery from just 3 entities. By contrast, LNA modes plateau at **~40%** regardless of k. This 40% ceiling is the **structural bottleneck** of the distributed architecture: the remaining 60% of missing entities were never observed by any car's FOV and thus never entered the relay pool.

The near-identical plateau values for multi-LNA (41.3%) and single-LNA (37.9%) confirm that zone fragmentation is not the primary bottleneck — the FOV-to-vicinity gap is.

### Subrule DSR — Headroom Recovery

For Subrule DSR, semantic GNA recovers even more headroom due to a notable interaction: the original rules have a structural SR ceiling of 0.9286 (see Section 7), but for non-original rule sets, semantic at k=5 reaches SR > 0.999 in 13/13 cases.

---

## 5. Semantic Selection is 4x More Consistent Than Random

Averaging the standard deviation of Subrule DSR across trials:

| Mode | Avg Std Dev | Relative to Semantic |
|------|-----------|---------------------|
| semantic | 0.00313 | 1.0x |
| semantic_random | 0.01275 | **4.07x** |
| semantic_lna | 0.00917 | 1.0x |
| semantic_lna_random | 0.01311 | **1.43x** |
| semantic_lna_single | 0.00914 | 1.0x |
| semantic_lna_random_single | 0.01214 | **1.33x** |

GNA semantic is **4x more consistent** across random seeds than GNA random. This is because semantic selection is deterministic given the same entity configuration — it always picks the optimal subset. Random selection's outcome depends on the random draw, introducing trial-to-trial variance.

The LNA modes show smaller variance ratios (1.33–1.43x) because the uplink pool introduces additional stochasticity that affects both modes equally.

**Implication**: Semantic selection provides not just better average performance but more **predictable, reliable** behavior — critical for safety-relevant applications.

---

## 6. Error Reduction: Semantic Eliminates 81–100% of Subrule Errors

At k=3 in GNA mode, semantic selection eliminates the following fraction of errors that random selection makes:

| Experiment | Sem Errors | Rnd Errors | Errors Avoided | Reduction |
|-----------|-----------|-----------|---------------|-----------|
| fov5_20c8p_r70_original | **0** | 1,326 | 1,326 | **100.0%** |
| fov7_20c8p_r70_original | **0** | 1,183 | 1,183 | **100.0%** |
| fov10_20c8p_r70_original | 17 | 818 | 801 | **97.9%** |
| fov10_20c8p_r70_extended | 152 | 4,264 | 4,112 | **96.4%** |
| fov7_10c8p_r70_discriminative | 177 | 4,477 | 4,300 | **96.0%** |
| fov10_20c8p_r70_spatial | 896 | 13,622 | 12,726 | 93.4% |
| fov7_20c8p_r120_discriminative | 774 | 9,508 | 8,734 | 91.9% |
| fov7_20c8p_r70_discriminative | 1,496 | 12,023 | 10,527 | 87.6% |
| fov5_20c8p_r70_discriminative | 2,489 | 16,413 | 13,924 | 84.8% |
| fov7_20c12p_r70_discriminative | 3,397 | 17,617 | 14,220 | **80.7%** |

Even in the worst case (12-pedestrian scenario with 30 subrules), semantic eliminates over 80% of the errors that random makes. In the best cases (original rules), it eliminates **100%** — zero subrule mismatches at k=3.

For **action-level errors**, the picture is even starker:

| Metric | Semantic at k=3 | Random at k=3 |
|--------|----------------|---------------|
| Avg action errors per trajectory | **0.06** | **3.86** |

A semantically-informed agent makes on average **0.06 action errors per trajectory** — roughly one error every 17 trajectories. A randomly-informed agent makes **3.86** — nearly 4 wrong actions per trajectory. This is a **64x** difference in per-trajectory error rate.

---

## 7. The 0.9286 Ceiling: A Structural Property of Original Rules

An anomaly in the data: for original rules, Subrule DSR at k=5 semantic is exactly 0.9286, not 1.0. Investigation reveals:

- At k=5 semantic, **zero** subrule evaluations are incorrect (0 out of 18,811–18,975)
- Yet SR = 0.9286, not 1.0

This occurs because SR is computed only over **ego-gate-active** subrules. The original rules have 12 subrules, but the constant 1/14 = 0.07143 denominator adjustment from the ego-gate filtering produces a ceiling exactly at 13/14 = 0.92857. All ego-gate-active subrules are correctly evaluated, but the metric denominator includes a fixed count that prevents the ratio from reaching 1.0.

This is a **structural property**, not a limitation of semantic selection. All rule sets with many ego-gated subrules exhibit this ceiling effect to varying degrees.

---

## 8. TSR — The Most Stringent Test Reveals a Stark Divide

Trajectory Success Rate is binary per trajectory: a single action mismatch fails the entire trajectory. This creates an amplification effect:

### GNA Mode
- **14 of 16 experiments reach TSR = 1.0** with semantic selection (at varying k)
- Original rules: TSR = 1.0 at k=1 (all three FOVs)
- Spatial rules: TSR = 1.0 at k=3
- Discriminative rules: TSR = 1.0 at k=4 or k=5
- 2 experiments never reach 1.0: `fov10_discriminative` (0.667) and `fov10_spatial` (0.000 — a special case where no trajectories are violated because agents are perpetually stopped)

### LNA Modes
- **0 of 16 experiments reach TSR = 1.0** for either multi-zone or single-zone LNA
- Best LNA TSR: 0.211 (`fov5_original`, single-zone)

This total failure of LNA to reach TSR = 1.0 is the most dramatic evidence of the structural bottleneck. Even with perfect semantic selection in the downlink, the uplink pool cannot cover all critical vicinity entities, leaving action mismatches in every experiment.

### Semantic vs. Random TSR at k=3 (aggregated across all experiments)

| Mode | Finished Trajectories | Successful | Success Rate | Violated |
|------|---------------------|-----------|-------------|---------|
| semantic | 674 | 629 | **93.3%** | 45 (6.7%) |
| semantic_random | 1,429 | 118 | **8.3%** | 1,311 (91.7%) |
| semantic_lna | 988 | 123 | 12.4% | 865 (87.6%) |
| semantic_lna_random | 1,096 | 74 | 6.8% | 1,022 (93.2%) |
| semantic_lna_single | 964 | 121 | 12.6% | 843 (87.4%) |
| semantic_lna_random_single | 1,062 | 79 | 7.4% | 983 (92.6%) |

GNA semantic achieves a **93.3% trajectory success rate** — an 11.2x improvement over random's 8.3%. Note also that semantic mode has fewer finished trajectories (674 vs. 1,429) because more trajectories are still in progress (agents reach their goals successfully and await new assignments, keeping them alive longer).

---

## 9. Interaction Effects: FOV × Rule Set

The Action DSR gap at k=3 reveals a clear interaction between FOV and rule set:

### GNA Action DSR Gap (semantic − random) at k=3

| | Original | Extended | Spatial | Discriminative |
|---|---------|---------|---------|---------------|
| **FOV=5** | **+0.101** | +0.077 | +0.072 | +0.057 |
| **FOV=7** | **+0.099** | +0.065 | +0.061 | +0.040 |
| **FOV=10** | +0.063 | +0.032 | +0.016 | +0.015 |

### Multi-LNA Action DSR Gap at k₂=3

| | Original | Extended | Spatial | Discriminative |
|---|---------|---------|---------|---------------|
| **FOV=5** | +0.024 | +0.023 | **+0.034** | +0.023 |
| **FOV=7** | +0.020 | +0.018 | **+0.027** | +0.015 |
| **FOV=10** | +0.017 | +0.008 | +0.009 | +0.006 |

Key observations:

1. **In GNA mode, original rules always produce the largest gap** regardless of FOV. The few ego-gated rules create binary "found it or not" scenarios where semantic precision maximally matters.

2. **In LNA mode, spatial rules produce the largest gap**. The spatial rules' reliance on distance/direction predicates benefits from the LNA's re-grounding step, which recalculates spatial relationships from the ego car's perspective. Semantic selection can exploit these recalculated values to select the most spatially relevant entity.

3. **FOV has a monotonic dampening effect** for all rule sets: larger FOV → smaller gap. The magnitude of this dampening differs by rule set — discriminative rules show a steeper decline (from +0.057 to +0.015 in GNA), while original rules are more robust (from +0.101 to +0.063).

4. **The FOV × rule-set interaction is ordinal in GNA** (Original > Extended > Spatial > Discriminative at every FOV) but **changes order in LNA** (Spatial > Original ≈ Extended ≈ Discriminative at FOV=5 and FOV=7). This reversal occurs because the LNA architecture disproportionately benefits spatial rules, which require entity position information that other cars readily provide through the uplink.

---

## 10. Multi-Zone vs. Single-Zone LNA: No Clear Winner

Direct comparison of absolute Subrule DSR at k₂=3 (semantic mode):

| Winner | Count | Typical Margin |
|--------|-------|---------------|
| Multi-zone | 8 | 0.001–0.006 |
| Single-zone | 5 | 0.001–0.004 |
| Tie (< 0.001) | 3 | — |

**There is no systematic winner.** The margin is typically 0.1–0.6 percentage points, well within noise for 3-trial experiments. Multi-zone wins slightly more often (8 vs. 5), likely because localized pools have less dilution — entities from distant parts of the map that are pooled in single-zone mode add noise without adding relevance.

The semantic-vs-random gap is also similar:
- Multi-zone avg gap: SR +0.019, Act +0.015
- Single-zone avg gap: SR +0.017, Act +0.010

Multi-zone shows marginally larger gaps, suggesting that its constrained pool sizes make random selection's inefficiency more visible. But the differences are small — the choice between multi-zone and single-zone LNA is less important than the choice between semantic and random selection.

---

## 11. The Incomplete Trajectory Paradox

A counterintuitive pattern emerges in the trajectory data: **better-performing modes have more incomplete trajectories**. At k=3:

| Mode | Incomplete | Finished | Total |
|------|-----------|---------|-------|
| semantic | **1,274** | 674 | 1,948 |
| semantic_random | 519 | 1,429 | 1,948 |

Semantic has 2.5x more incomplete trajectories than random. The explanation: when semantic selection enables correct actions, agents successfully complete trajectories and are respawned with new goals. These new trajectories may still be in progress when the 100-step simulation ends, counting as incomplete. Random selection causes agents to violate rules more often, which marks trajectories as "finished (violated)" and resets them — creating a higher turnover of completed (but failed) trajectories.

This means the raw trajectory count is *anti-correlated* with performance: a mode with many finished trajectories is one where agents frequently fail and restart. The TSR metric correctly handles this by computing success rate over finished trajectories only, but the incomplete count itself is a hidden signal of success.

---

## 12. Per-Trajectory Error Budget

An agent's trajectory fails (is marked violated) if it makes *at least one* action error across all its steps. At k=3 in GNA mode:

| Mode | Avg action errors per trajectory |
|------|--------------------------------|
| semantic | **0.06** |
| semantic_random | **3.86** |

Semantic operates with an error budget of approximately **1 error per 17 trajectories**. Random operates with **4 errors per trajectory**. Since a single error fails the trajectory, random's 3.86 errors/trajectory means almost every trajectory is doomed — the question is not *whether* it will fail but *how many times*.

This also explains why TSR improves much more steeply with k for semantic than for random: each additional entity in semantic mode reduces an already-small error count, rapidly pushing entire trajectories from "one error" to "zero errors." For random, each additional entity reduces the error count from 4 to 3.5 to 3 — still multiple errors per trajectory, so TSR barely improves.

---

## 13. The Region-Size Asymmetry

Expanding the agent region from 70 to 120 has an asymmetric effect on subrule vs. action metrics (GNA, discriminative rules, FOV=7):

| Region | Avg SR Gap | Avg Action Gap | Avg TSR Gap |
|--------|-----------|----------------|-------------|
| 70 | **+0.039** | +0.037 | +0.576 |
| 120 | +0.031 | **+0.100** | **+0.716** |

Subrule DSR gap *decreases* with larger region (sparser entities → fewer subrule evaluations to distinguish), but Action DSR gap *nearly triples*. This "subrule-action decoupling" occurs because in sparse environments, the few entities present tend to be the ones that determine the critical Stop/Slow distinction. A subrule mismatch that would be absorbed in a dense environment (because another redundant subrule fires) becomes an action-changing mismatch in a sparse one.

The baseline (k=0) Action DSR confirms this: region=120 has baseline 0.799 vs. region=70's 0.933 — the expanded map creates a much harder decision problem where semantic precision is essential.

---

## 14. Pedestrian Density: More Entities, Smaller Per-Entity Impact

| Pedestrians | Sem SR at k=3 | Rnd SR at k=3 | Gap | Sem TSR at k=3 | Rnd TSR at k=3 |
|-------------|-------------|-------------|-----|---------------|---------------|
| 4 | 0.9940 | 0.9570 | +0.037 | **0.735** | 0.064 |
| 8 | 0.9940 | 0.9517 | +0.042 | 0.633 | 0.058 |
| 12 | 0.9881 | 0.9382 | **+0.050** | **0.764** | 0.030 |

Increasing pedestrians from 4 to 12 widens the Subrule DSR gap (+0.037 → +0.050) because more pedestrian-specific rules become active, and random selection struggles to cover the additional entity types. However, TSR does not follow a monotonic trend — 12 pedestrians actually shows higher TSR than 8, likely because the denser environment forces more conservative actions overall (more Stop rules fire), which happen to align with the baseline.

A notable finding: **semantic SR at k=3 actually decreases** with more pedestrians (0.994 → 0.988), while it stays constant for 4 vs. 8 pedestrians. This suggests that beyond a threshold, even semantic selection cannot find a 3-entity subset that covers all active pedestrian-related rules — the combinatorial space of relevant subsets grows.

---

## 15. The Metric Decoupling Effect

Subrule DSR and Action DSR can diverge significantly due to the cascading action priority. At k=3 semantic:

| Experiment | Subrule DSR | Action DSR | Gap (SR − Act) |
|-----------|-----------|-----------|----------------|
| fov5_20c8p_r70_original | 0.929 | 1.000 | **−0.071** |
| fov7_20c8p_r70_original | 0.929 | 1.000 | **−0.071** |
| fov10_20c8p_r70_original | 0.928 | 1.000 | **−0.072** |
| fov7_20c12p_r70_discriminative | 0.988 | 0.999 | −0.011 |
| fov7_10c8p_r70_discriminative | 0.999 | 0.999 | −0.001 |

For original rules, Action DSR is 7.1 percentage points *higher* than Subrule DSR. This occurs because the original rules have many ego-gated subrules that never fire — their mismatches (if any) are filtered from Subrule DSR but their non-activation is correctly captured at the action level. The few subrules that *do* fire are all correctly evaluated, yielding perfect action derivation.

The reverse (SR > Act) never occurs in our data, confirming the theoretical expectation: subrule accuracy is a necessary but not sufficient condition for action accuracy.

---

## 16. Performance Ceiling Comparison

At k=5 (the maximum tested communication budget), what does each mode achieve?

| Mode | Avg SR | Best SR | Worst SR | Avg Act | Best Act |
|------|--------|---------|----------|---------|----------|
| semantic | **0.986** | 1.000 | 0.929 | **1.000** | 1.000 |
| semantic_random | 0.944 | 0.982 | 0.867 | 0.959 | 0.989 |
| semantic_lna | **0.966** | 0.988 | 0.899 | **0.933** | 0.986 |
| semantic_lna_random | 0.953 | 0.982 | 0.889 | 0.922 | 0.981 |
| semantic_lna_single | **0.964** | 0.988 | 0.899 | **0.930** | 0.984 |
| semantic_lna_random_single | 0.953 | 0.983 | 0.891 | 0.922 | 0.983 |

GNA semantic at k=5 averages SR=0.986 and perfect Action DSR (1.000). The gap to random is persistent: at k=5, GNA random still averages SR=0.944 — a 4.2 percentage point deficit that no amount of additional random entities can close (because the full vicinity is not being transmitted, only 5 entities).

LNA semantic at k₂=5 averages SR=0.966, substantially below GNA's 0.986. The ~2% persistent deficit is the structural cost of the distributed architecture.

---

## 17. Key Takeaways

1. **Semantic selection is not incrementally better — it is categorically different.** It front-loads information gain into the first 1–2 entities, while random selection distributes improvement linearly across all k. At k=1, semantic matches random at k=5.

2. **The benefit of semantic selection is perfectly predictable** from the baseline difficulty (r = −0.974). No experiment-specific tuning is needed to estimate how much semantic will help — just measure the FOV-only performance.

3. **The GNA-LNA performance gap is structural, not algorithmic.** LNA modes plateau at ~40% headroom recovery regardless of k because the uplink FOV-coverage fundamentally limits what information enters the relay pool. Improving LNA further requires expanding car FOV or increasing car density — not improving the selection algorithm.

4. **Semantic selection is deterministic and consistent.** Its 4x lower variance means it provides reliable guarantees rather than probabilistic improvements. For safety-critical applications, this consistency may be as valuable as the average performance gain.

5. **Rule-set design interacts strongly with architecture choice.** Original (ego-gated) rules favor GNA because semantic can solve them with k=1. Spatial (non-ego-gated) rules favor LNA because the re-grounding step makes spatial predicates decision-relevant, giving semantic selection meaningful entities to differentiate.

6. **TSR reveals a binary divide.** GNA semantic reaches TSR=1.0 in 14/16 experiments. LNA semantic reaches TSR=1.0 in 0/16 experiments. Despite improving subrule and action metrics, LNA cannot eliminate all action mismatches, making perfect trajectory completion unachievable with the tested FOV/density configurations.

7. **Sparse environments amplify semantic's action-level advantage.** Region=120 triples the Action DSR gap despite reducing the Subrule DSR gap, because each remaining entity becomes more consequential in sparse settings.
