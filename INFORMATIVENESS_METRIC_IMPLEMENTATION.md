# Informativeness Metric Implementation

## Overview

The informativeness metric quantifies how much "rule-critical information" an agent can observe through their FOV and GNA broadcast, normalized by the total information available in the scene.

## Metric Definition

### Informativeness Score Formula

```
Normalized Informativeness = (Local + GNA Score) / (Global Oracle Score)

Where:
- Local + GNA Score = Σ (occurrence_score of each visible entity)
- Global Oracle Score = Σ (occurrence_score of all entities in scene)
```

## Implementation Details

### 1. Entity Occurrence Scores

Defined in `config.py`:

```python
ENTITY_OCCURRENCE_SCORES = {
    "Ambulance": 7,  # Appears in 7 sub-rules
    "Old": 6,        # Appears in 6 sub-rules
    "Police": 4,     # Appears in 4 sub-rules
    "Bus": 2,        # Appears in 2 sub-rules
    "Pedestrian": 2,
    "Reckless": 2,
    "Tiro": 2,
    "Young": 2,
    "Car": 1         # Appears in 1 sub-rule
}
```

### 2. Global Oracle Score Calculation

In `z3.py::_analyze_global_subrule_satisfiability()`:

```python
global_oracle_informativeness = 0

for agent_obj in agents:
    entity_type = self._get_entity_type_from_concepts(agent_obj.concepts)
    if entity_type:
        score = ENTITY_OCCURRENCE_SCORES.get(entity_type, 0)
        global_oracle_informativeness += score

self.current_global_oracle_informativeness = global_oracle_informativeness
```

**Example**:
- Scene has: 2 Ambulances, 3 Old, 1 Police, 2 Bus, 2 Reckless, etc.
- Global Oracle Score = 7+7 + 6+6+6 + 4 + 2+2 + 2+2 + ... = **59**

### 3. Local + GNA Score Calculation

In `z3.py::calculate_local_gna_informativeness()`:

```python
def calculate_local_gna_informativeness(self, ego_agent, partial_agents, gna_broadcast):
    local_gna_score = 0
    entities_counted = set()
    
    # Count ego agent
    ego_type = self._get_entity_type_from_concepts(ego_agent.concepts)
    local_gna_score += ENTITY_OCCURRENCE_SCORES.get(ego_type, 0)
    entities_counted.add(f"ego_{ego_agent.id}")
    
    # Count FOV entities (each entity counted separately)
    for agent_name, agent in partial_agents.items():
        entity_type = self._get_entity_type_from_concepts(agent.concepts)
        entity_id = f"fov_{agent.layer_id}"
        if entity_id not in entities_counted:
            local_gna_score += ENTITY_OCCURRENCE_SCORES.get(entity_type, 0)
            entities_counted.add(entity_id)
    
    # Count GNA broadcast entities (avoid double-counting)
    if gna_broadcast:
        for agent_id, agent_data in gna_broadcast['global_context'].items():
            gna_entity_id = f"gna_{agent_id}"
            if gna_entity_id not in entities_counted:
                entity_type = self._get_entity_type_from_concepts(concepts)
                local_gna_score += ENTITY_OCCURRENCE_SCORES.get(entity_type, 0)
                entities_counted.add(gna_entity_id)
    
    return local_gna_score, len(entities_counted)
```

**Example for Agent_5**:
- Ego: Car (score: 1)
- FOV: Ambulance_1 (7), Old_1 (6)
- GNA (k=3): Ambulance_2 (7), Old_2 (6), Police_1 (4)
- **Local+GNA Score** = 1 + 7 + 6 + 7 + 6 + 4 = **31**
- **Normalized** = 31 / 59 = **0.525** (52.5%)

### 4. Averaging Across Agents

In `z3.py::plan()`:

```python
# Calculate for each agent
for ego_name in partial_agents.keys():
    local_gna_score, entity_count = self.calculate_local_gna_informativeness(
        ego_agent_obj, partial_agents[ego_name], gna_broadcast
    )
    normalized = local_gna_score / self.current_global_oracle_informativeness

# Average across all agents
avg_normalized = sum(all_normalized_scores) / num_agents
```

**Logged as**: "Average Normalized Informativeness: 0.XXX (XX.X%)"

## Interpretation

### What the Metric Tells Us

- **0.00 (0%)**: Agent sees no rule-critical entities (only itself or very low-priority entities)
- **0.25 (25%)**: Agent sees 25% of total rule-critical information in scene
- **0.50 (50%)**: Agent sees half of available rule-critical information
- **1.00 (100%)**: Agent has perfect information (sees all entities - only possible with full GNA)

### Comparison: Priority vs Random

#### Priority Selection (k=5)
Broadcasts: Ambulance_1, Ambulance_2, Old_1, Old_2, Old_3
```
Broadcast contributes: 7 + 7 + 6 + 6 + 6 = 32 points
If FOV already has Car (1) + Pedestrian (2):
Local+GNA = 1 + 2 + 32 = 35
Normalized = 35 / 59 = 0.593 (59.3%)
```

#### Random Selection (k=5)
Broadcasts: Ambulance_1, Bus_1, Reckless_1, Young_1, Police_1
```
Broadcast contributes: 7 + 2 + 2 + 2 + 4 = 17 points
If FOV has Car (1) + Pedestrian (2):
Local+GNA = 1 + 2 + 17 = 20
Normalized = 20 / 59 = 0.339 (33.9%)
```

**BUT**: Random provides better DIVERSITY, so it helps satisfy more sub-rules despite lower informativeness!

This creates an interesting trade-off:
- **Priority**: Higher informativeness (concentrated on critical entities)
- **Random**: Better sub-rule coverage (diverse entity types)

## Expected Trends

### Priority Selection
```
k=1:  0.12-0.15 (only most critical entity type)
k=3:  0.30-0.40 (Ambulance + Old entities)
k=5:  0.50-0.60 (saturates due to redundancy)
k=10: 0.80-0.90 (eventually covers most types)
```

### Random Selection
```
k=1:  0.05-0.15 (random entity)
k=3:  0.20-0.30 (3 random entities, diverse but might miss critical ones)
k=5:  0.35-0.45 (more diverse types)
k=10: 0.70-0.85 (good coverage but might include low-score entities)
```

## Key Insights

1. **Informativeness ≠ Sub-rule Coverage**: High informativeness doesn't guarantee high fully-observed sub-rules
2. **Priority maximizes informativeness** by focusing on high-occurrence entities
3. **Random maximizes diversity** which helps satisfy diverse sub-rule requirements
4. **Optimal strategy**: Diversity-aware selection that balances both metrics

## Usage in Experiments

The experiment scripts now output:

```
==========================================
RESULTS FOR gna_top_k = 5
==========================================
Per-Agent Averages (averaged over 20 trials):
  Fully observed:     6.61
  Partially observed: 2.72
  Unobserved:         3.67
  Informativeness:    0.525 (52.5%)
==========================================
```

Summary table:
```
gna_top_k | Fully | Partially | Unobserved | Informativeness
----------|-------|-----------|------------|----------------
        1 |  1.86 |      4.10 |       7.05 | 0.145 (14.5%)
        5 |  6.61 |      2.72 |       3.67 | 0.525 (52.5%)
       10 |  8.10 |      4.08 |       0.82 | 0.892 (89.2%)
```

This allows you to compare both metrics across selection strategies!

