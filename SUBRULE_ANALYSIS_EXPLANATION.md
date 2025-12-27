# Sub-rule Observability Analysis - Complete Explanation

## Overview

This document explains how sub-rule observability analysis works and why certain counterintuitive behaviors occur.

## What Are Sub-rules?

In expert mode, there are **13 sub-rules** total:
- **Stop**: 7 sub-rules
- **Slow**: 3 sub-rules
- **Fast**: 3 sub-rules

Each sub-rule requires specific entity types to be "fully observed". For example:
- Stop sub-rule 1: requires `['Ambulance', 'Old']`
- Stop sub-rule 4: requires `['Ambulance', 'Police', 'Car']`
- Slow sub-rule 1: requires `['Tiro', 'Pedestrian']`
- Fast sub-rule 2: requires `['Bus']`

## How Observability is Determined

For each agent at each timestep, the system checks **13 sub-rules**:

### Step 1: Collect Present Entity Types

The system gathers entity types from THREE sources:

1. **Ego agent**: The agent's own entity type (e.g., "Old", "Ambulance", "Car")
2. **FOV (Field of View)**: Entity types of agents visible in the agent's local field of view
3. **GNA Broadcast**: Entity types from the Global Navigation Assistant's top-K broadcast

```python
def _get_present_entity_types(self, ego_agent, partial_agents, gna_broadcast=None):
    present_types = set()
    
    # Check ego agent
    if hasattr(ego_agent, 'concepts') and ego_agent.concepts:
        entity_type = self._get_entity_type_from_concepts(ego_agent.concepts)
        if entity_type:
            present_types.add(entity_type)
    
    # Check agents in FOV
    for agent_name, agent in partial_agents.items():
        if hasattr(agent, 'concepts') and agent.concepts:
            entity_type = self._get_entity_type_from_concepts(agent.concepts)
            if entity_type:
                present_types.add(entity_type)
    
    # Check GNA broadcast
    if gna_broadcast and 'global_context' in gna_broadcast:
        for agent_name, agent_data in gna_broadcast['global_context'].items():
            if 'agent_properties' in agent_data and 'concepts' in agent_data['agent_properties']:
                concepts = agent_data['agent_properties']['concepts']
                entity_type = self._get_entity_type_from_concepts(concepts)
                if entity_type:
                    present_types.add(entity_type)
    
    return present_types
```

### Step 2: Classify Each Sub-rule

For each of the 13 sub-rules:

```python
required_entity_types = sub_rule['required_entity_types']  # e.g., ['Ambulance', 'Old']

observed = [et for et in required_entity_types if et in present_entity_types]
missing = [et for et in required_entity_types if et not in present_entity_types]

if len(observed) == len(required_entity_types):
    # ALL required types present
    grounding_status = 'fully_observed'
elif len(observed) > 0:
    # SOME but not all required types present
    grounding_status = 'partially_observed'
else:
    # NO required types present
    grounding_status = 'fully_unobserved'
```

## What is `agent_region`?

The `agent_region` parameter **controls the spatial area where agents can spawn** their start and goal positions.

```python
# In Car.get_start() and Pedestrian.get_start():
desired_locations[self.region:, :] = False  # Set positions beyond region to invalid
desired_locations[:, self.region:] = False  # Set positions beyond region to invalid
```

**Effect**:
- `agent_region=70`: Agents spawn in a **70×70 pixel area** → DENSE clustering
- `agent_region=200`: Agents spawn in a **200×200 pixel area** → SPARSE distribution

## Why Larger agent_region → LESS Fully Observed

This is the KEY insight:

### Scenario 1: agent_region=70 (Small, Dense)

```
World: 70×70 area
Agents: 18 agents (8 pedestrians, 10 cars) packed in 70×70 space

Agent A's perspective:
├─ FOV: Sees 5 nearby agents (diverse types: Old, Young, Ambulance, Car, Tiro)
├─ GNA: Broadcasts top-3 (Bus, Ambulance, Old) - all nearby
└─ Present types: {Old, Young, Ambulance, Car, Tiro, Bus, Pedestrian} = 7 types

Sub-rule check:
├─ Stop sub-rule 1: requires [Ambulance, Old] → ✅ FULLY OBSERVED
├─ Stop sub-rule 4: requires [Ambulance, Police, Car] → ⚠️ PARTIALLY (no Police nearby)
├─ Slow sub-rule 1: requires [Tiro, Pedestrian] → ✅ FULLY OBSERVED
└─ Fast sub-rule 2: requires [Bus] → ✅ FULLY OBSERVED

Result: Many sub-rules FULLY OBSERVED because agents are close → diverse entities visible
```

### Scenario 2: agent_region=200 (Large, Sparse)

```
World: 200×200 area
Agents: 18 agents spread across 200×200 space

Agent A's perspective:
├─ FOV: Sees 1-2 nearby agents (limited types: Car, Pedestrian)
├─ GNA: Broadcasts top-3 but they're far away and maybe not in FOV
└─ Present types: {Car, Pedestrian, Ambulance} = 3 types (via GNA)

Sub-rule check:
├─ Stop sub-rule 1: requires [Ambulance, Old] → ⚠️ PARTIALLY (Ambulance via GNA, but no Old)
├─ Stop sub-rule 4: requires [Ambulance, Police, Car] → ⚠️ PARTIALLY (no Police)
├─ Slow sub-rule 1: requires [Tiro, Pedestrian] → ⚠️ PARTIALLY (no Tiro visible)
└─ Fast sub-rule 2: requires [Bus] → ❌ UNOBSERVED (no Bus visible)

Result: Many sub-rules PARTIALLY/UNOBSERVED because agents are sparse → limited entity diversity
```

**Mathematical relationship**:
- Density ∝ 1 / (agent_region²)
- Expected agents in FOV ∝ Density
- Entity type diversity ∝ Expected agents in FOV
- Fully observed sub-rules ∝ Entity type diversity

Therefore: **↑ agent_region → ↓ Density → ↓ Diversity → ↓ Fully Observed**

## Local vs Global Oracle Comparison

### CRITICAL BUG IDENTIFIED

There is a **fundamental flaw** in how global and local statistics are compared:

#### Local Analysis (Per-Agent)
```python
for ego_name in partial_agents.keys():  # Iterates over ALL 18 agents
    ego_agent_obj = ego_agent[ego_name]
    analysis = self.analyze_subrule_satisfiability(
        ego_agent_obj, partial_agents[ego_name],
        partial_world[ego_name], partial_intersections[ego_name],
        gna_broadcast, ego_name
    )
    # Counts 13 sub-rules for THIS agent
    # Total per timestep: 13 × 18 = 234 sub-rule evaluations
```

**Local "Per-Agent Averages"** = Total counts / Number of agent analyses

- If 100 timesteps with 18 agents:
  - total_agent_analyses = 100 × 18 = 1800
  - If 5494 total fully observed across all evaluations
  - Per-Agent Average = 5494 / 1800 = **3.05 per agent**

#### Global Oracle Analysis
```python
def _analyze_global_subrule_satisfiability(...):
    # Creates ONE mock ego agent with perfect vision
    mock_ego_agent = agents[0]
    global_agents = {all agents}  # All 18 agents
    mock_gna_broadcast = {all 18 agents}  # Perfect knowledge
    
    # Calls analyze_subrule_satisfiability ONCE
    global_analysis = self.analyze_subrule_satisfiability(
        mock_ego_agent, global_agents, global_world, global_intersections,
        mock_gna_broadcast, "global_oracle"
    )
    # Counts 13 sub-rules for ONE mock agent
    # Total per timestep: 13 × 1 = 13 sub-rule evaluations
```

**Global Oracle "Averages"** = Total counts / Number of oracle analyses (= number of timesteps)

- If 100 timesteps:
  - global_oracle_analyses = 100
  - If ALL entity types present in world, maybe 10 sub-rules fully observed per timestep
  - Global Oracle Average = 1000 / 100 = **10.0 per oracle**

### Why Local Can EXCEED Global (THE BUG)

This should be **impossible** if implemented correctly, but it happens because:

**BUG 1: Global oracle doesn't actually see all entity types**

The global oracle analysis uses:
```python
present_types = self._get_present_entity_types(
    ego_agent=agents[0],  # ONE specific agent
    partial_agents=global_agents,  # All agents
    gna_broadcast=mock_gna_broadcast  # All agents
)
```

However, `partial_agents` is keyed by **string IDs** like `"1"`, `"2"`, etc., which are agent IDs, NOT properly formatted agent layer IDs.

When `_get_present_entity_types()` iterates:
```python
for agent_name, agent in partial_agents.items():
    # agent_name = "1", "2", "3", etc.
    # But these need to be actual agent objects with .concepts attribute
```

**If the agents dict doesn't properly expose `.concepts`, entity types won't be detected!**

**BUG 2: World state variability**

At any given timestep, not all 18 agent types may be present:
- Some agents may have reached their goals and respawned
- Agents spawn with random concepts from the configuration
- The actual world state varies

So global oracle at timestep T might only see:
- 2 Ambulances, 1 Old, 3 Young, 2 Buses, etc.
- Missing: Police, Tiro, Reckless, etc.

Meanwhile, local agents with dense spawning (small agent_region) see:
- High local diversity due to clustering
- GNA broadcasts high-priority entities
- Each agent's local view might be MORE diverse than one global snapshot

**BUG 3: Conceptual mismatch**

- **Local**: "What can individual agents observe with limited FOV + GNA?"
  - Averaged across all agents
  - Each agent has different local context
  - Dense spawning = high local diversity

- **Global**: "What can one omniscient agent observe?"
  - Single perspective
  - But doesn't account for the fact that with dense spawning, LOCAL diversity can be higher than GLOBAL diversity at a single point in time

## Agents in Expert Mode

Total: **18 agents**

### Pedestrians (8 total)
- 3 × Old Pedestrian
- 3 × Young Pedestrian  
- 2 × Regular Pedestrian

### Cars (10 total)
- 2 × Ambulance
- 2 × Bus
- 2 × Tiro
- 2 × Reckless
- 1 × Police
- 1 × Regular Car

### Entity Types Present
Possible types: `{Old, Young, Pedestrian, Ambulance, Bus, Tiro, Reckless, Police, Car}`

However, entity type detection from concepts uses a priority order:
```python
concept_priority_order = [
    ('bus', 'Bus'),
    ('ambulance', 'Ambulance'),
    ('old', 'Old'),
    ('tiro', 'Tiro'),
    ('police', 'Police'),
    ('young', 'Young'),
    ('reckless', 'Reckless'),
    ('pedestrian', 'Pedestrian'),
    ('car', 'Car')
]
```

So a car with `bus=1.0` is detected as "Bus", not "Car".

Actual detected types in the scene:
1. Bus (2)
2. Ambulance (2)
3. Old (3)
4. Tiro (2)
5. Police (1)
6. Young (3)
7. Reckless (2)
8. Pedestrian (2 - those without old/young concepts)

**Missing from some sub-rules**: Depending on spawning, some entity types may cluster or be isolated.

## ✅ FIXED: Global Oracle Implementation

The global oracle has been **fixed** to properly detect all entity types in the scene and evaluate sub-rules correctly.

### What Was Fixed

**Before (Buggy)**:
- Tried to simulate an agent's perspective with `analyze_subrule_satisfiability()`
- Used improper dict structures that didn't expose agent concepts
- Could miss entity types due to implementation issues

**After (Fixed)**:
- Directly collects ALL entity types from all agents in the scene
- Evaluates each sub-rule by checking: `required_types.issubset(all_entity_types)`
- No longer relies on mock agents or complex dict structures
- Provides detailed logging of entity types present and missing

### The Fixed Implementation

```python
def _analyze_global_subrule_satisfiability(self, world_matrix, agents, intersect_matrix):
    """
    Analyze sub-rule satisfiability for the global (oracle) world state.
    
    This checks which sub-rules would be fully observable if an agent had
    perfect knowledge of all entity types present in the entire world.
    """
    # Collect all entity types actually present in the scene
    all_entity_types = set()
    for agent_obj in agents:
        if hasattr(agent_obj, 'concepts') and agent_obj.concepts:
            entity_type = self._get_entity_type_from_concepts(agent_obj.concepts)
            if entity_type:
                all_entity_types.add(entity_type)
    
    # Evaluate each sub-rule with perfect knowledge
    for rule_name, sub_rules in self.sub_rules.items():
        for sub_rule in sub_rules:
            required_types = set(sub_rule['required_entity_types'])
            
            if not required_types:  # Spatial-only rules
                global_fully_observed += 1
            elif required_types.issubset(all_entity_types):  # All required types present
                global_fully_observed += 1
            elif required_types.intersection(all_entity_types):  # Some types present
                global_partially_observed += 1
            else:  # No types present
                global_fully_unobserved += 1
    
    # Log and update counters...
```

### Expected Behavior After Fix

With the fixed implementation:
- **Global oracle SHOULD always be ≥ local averages** for "fully observed"
- Global oracle shows the theoretical maximum observability (all entity types visible)
- Local averages show practical observability (limited FOV + GNA)
- The gap between global and local indicates the value of better perception/communication

### Interpretation of Current Results

Given the bugs identified:

**When agent_region increases:**
- Agents spread out → lower local density
- Fewer agents in FOV → fewer entity types visible locally
- GNA still broadcasts top-K, but with sparse spawning, top-K entities may be far away and not useful
- Result: Fewer fully observed sub-rules

**Why local can exceed global:**
- Dense spawning (small region) creates high LOCAL diversity within FOVs
- Each agent sees many nearby diverse agents
- Global oracle may not properly detect all entities due to implementation bugs
- Global oracle analyzes from ONE perspective while local averages across ALL agent perspectives

## Summary

1. **Sub-rule logic**: Checks if required entity types are present (ego + FOV + GNA)
2. **agent_region effect**: Controls spawn density; smaller = denser = more diversity in FOV
3. **Local vs Global bug**: Implementation issues cause global oracle to potentially miss entity types
4. **Why trends occur**: Dense spawning → high local diversity → more fully observed sub-rules

The counterintuitive result (local > global) is due to implementation bugs in the global oracle analysis, not a fundamental logical impossibility.

