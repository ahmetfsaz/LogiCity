# Global Entity Integration with Coordinate System - Implementation Summary

## Overview

Implemented a **hybrid local-global reasoning system** that allows RL agents to reason about both:
- **Local entities** within the 25x25 FOV
- **Global high-priority entities** anywhere in the environment (via GNA broadcast)

## Architecture

### Data Flow

```
World State
    ↓
GNA Collects All Agents → Computes Intersection States → Broadcasts Top-K
    ↓                                                           ↓
Local FOV Extraction                                  Global Entities (with abs. positions)
    ↓                                                           ↓
Entity Selection: m local + (5-m) global
    ↓
Z3 Predicate Evaluation (uses global coordinates for spatial predicates)
    ↓
Logical Grounding Vector (205 dim)
    ↓
Neural Network (NLM-DQN)
```

## Implementation Details

### 1. **Enhanced PseudoAgent Class** (`z3.py`)

Added global coordinate support:
```python
class PesudoAgent:
    def __init__(self, type, layer_id, concepts, moving_direction, 
                 global_pos=None, in_fov_matrix=True):
        # ... existing fields ...
        self.global_pos = global_pos  # Absolute world position
        self.in_fov_matrix = in_fov_matrix  # Whether in local FOV matrix
```

### 2. **GNA Spatial Context Enhancement** (`gna.py`)

GNA now computes and broadcasts:
- **Absolute positions** of all agents
- **Intersection states** (`is_at_intersection`, `is_in_intersection`)
- Pre-computed for all agents in the environment

```python
agent_properties = {
    'position': agent_pos,  # Absolute coordinates
    'is_at_intersection': bool(...),  # Pre-computed
    'is_in_intersection': bool(...),  # Pre-computed
    # ... other properties ...
}
```

### 3. **Global Pseudo Agent** (`basic.py`)

`GlobalPseudoAgent` carries complete spatial information:
```python
class GlobalPseudoAgent:
    def __init__(self, ...):
        self.pos = position  # Absolute position
        self.global_pos = position  # Duplicate for clarity
        self.in_fov_matrix = False  # Mark as global
        self.is_at_intersection = ...  # From GNA
        self.is_in_intersection = ...  # From GNA
```

### 4. **Entity Selection** (`z3_rl.py`)

Lines 191-246: Smart entity selection:
```python
# Configuration
max_local = 3   # From config: max_local_entities
max_global = 2  # From config: max_global_entities

# Selection Process:
1. Ego agent (always included in local)
2. Up to (m-1) nearest local FOV entities
3. Up to (5-m) highest priority global entities from GNA
4. Pad with placeholders to reach 5 total
```

**Key Feature:** Global entities are added WITHOUT requiring them to be in the FOV matrix!

### 5. **Predicate System Overhaul** (`pred_converter/z3.py`)

Created dual-mode predicate evaluation:

**Helper Function** (lines 157-187):
```python
def _get_entity_position(world_matrix, agents, entity_name, agent_type, layer_id):
    # Check if global entity
    if not agent_obj.in_fov_matrix:
        return agent_obj.global_pos  # Use absolute coordinates
    else:
        return world_matrix[layer_id]  # Use FOV-relative coordinates
```

**Updated Predicates:**
- `IsAtInter`: Uses `agent.is_at_intersection` for global entities
- `IsInInter`: Uses `agent.is_in_intersection` for global entities
- `IsClose`, `CollidingClose`, `LeftOf`, `RightOf`, `NextTo`: Use `global_pos` for distance/direction calculations

## How It Works

### Example Scenario

**Setup:**
- Ego agent (Car_1) at position [46, 25] with FOV 25x25
- FOV contains: 2 nearby cars
- Global broadcast contains: Ambulance at [92, 86] (far outside FOV), Bus at [62, 46]

**With Config:**
```yaml
fov_entities:
  Entity: 5
  max_local_entities: 2  # Ego + 1 local car
  max_global_entities: 3  # Ambulance, Bus, + 1 more
```

**Processing:**

1. **Local Entity Collection:**
   - Ego (Car_1) at [46, 25]
   - Nearest car in FOV

2. **Global Entity Integration:**
   - Ambulance at [92, 86] (priority 2) - **ADDED**
   - Bus at [62, 46] (priority 1) - **ADDED**
   - Police at [75, 50] (priority 5) - **ADDED**

3. **Predicate Evaluation:**
   ```python
   IsAmbulance(Entity_3) → TRUE (checks concepts)
   IsAtInter(Entity_3) → Uses ambulance.is_at_intersection from GNA
   IsClose(Entity_0, Entity_3) → Calculates distance using:
       - Entity_0: FOV-relative position from world_matrix
       - Entity_3: Absolute position [92, 86] from global_pos
       - Result: FALSE (too far)
   HigherPri(Entity_3, Entity_0) → TRUE (ambulance priority 2 < car priority 9)
   ```

4. **Result:**
   - Agent knows there's a high-priority ambulance far away
   - Can reason about priority relationships globally
   - Makes informed decisions even about distant entities

## Files Modified

1. **`logicity/planners/local/z3.py`**
   - Enhanced `PesudoAgent` class with global position support

2. **`logicity/planners/local/z3_rl.py`**
   - Entity splitting logic (local vs global)
   - Global entity integration from GNA

3. **`logicity/agents/gna.py`**
   - Added intersection state computation
   - Broadcasts spatial context globally

4. **`logicity/agents/basic.py`**
   - Enhanced `GlobalPseudoAgent` with spatial info

5. **`logicity/utils/pred_converter/z3.py`**
   - Added `_get_entity_position` helper
   - Updated all spatial predicates to handle global coordinates

6. **Config files** (easy/medium/hard/expert):
   - Added `max_local_entities` and `max_global_entities` parameters

## Benefits

### 1. **True Global Awareness**
- Agent can reason about important entities anywhere in the 100x100 world
- Not limited to 25x25 FOV for entity detection

### 2. **Flexible Entity Mix**
Adjust the balance between local and global:
```yaml
# Focus on immediate surroundings
max_local_entities: 4
max_global_entities: 1

# Balance local and global
max_local_entities: 3
max_global_entities: 2

# Maximum global awareness
max_local_entities: 2
max_global_entities: 3
```

### 3. **Priority-Based Selection**
Global entities sorted by:
1. Entity type priority (Bus > Ambulance > Old Person > ...)
2. Can be extended with distance-based ranking

### 4. **Complete Spatial Reasoning**
All predicates work with global entities:
- Type predicates: `IsAmbulance`, `IsCar`, etc.
- Intersection predicates: `IsAtInter`, `IsInInter` (from GNA pre-computation)
- Spatial relationships: `IsClose`, `LeftOf`, `RightOf`, `NextTo`, `CollidingClose` (using global coordinates)
- Priority: `HigherPri` (works as before)

### 5. **Backward Compatible**
- Set `max_global_entities: 0` → Original behavior
- Disable GNA → No global entities
- Old configs still work

## Configuration

### Basic Setup
```yaml
simulation:
  enable_gna: true
  gna_top_k: 5  # Broadcast top-5 priority entities

rl_agent:
  fov_entities:
    Entity: 5
    max_local_entities: 3
    max_global_entities: 2
```

### Advanced: Maximize Global Awareness
```yaml
simulation:
  enable_gna: true
  gna_top_k: 10  # Larger pool for selection

rl_agent:
  fov_entities:
    Entity: 7  # More total entities
    max_local_entities: 3  # Keep local small
    max_global_entities: 4  # Most entities from global
```

## Testing

The system will log entity selection:
```
Agent Car_1: Added 2 local entities (max: 2)
Agent Car_1: Added 3 global entities from GNA broadcast (max: 3)
Agent Car_1: Total entities = 5 (local: 2, global: 3, placeholders: 0)
```

## Use Cases

### 1. **Emergency Vehicle Awareness**
Ego agent (regular car) can detect ambulance blocks away and prepare to yield even before it's visible.

### 2. **Traffic Flow Coordination**
Know about buses at distant intersections affecting traffic patterns.

### 3. **Priority-Based Navigation**
Reason about all high-priority entities globally, not just nearby ones.

### 4. **Strategic Planning**
Make decisions based on global traffic state, not just immediate surroundings.

## Performance Considerations

- **Memory:** Slightly increased (storing global positions)
- **Computation:** Minimal overhead (GNA already computes positions)
- **Observation Dimension:** **Unchanged** (still 205 or 87)
- **Predicate Evaluation:** Similar speed (helper function adds minimal overhead)

## Future Enhancements

1. Add time-to-collision prediction for global entities
2. Implement trajectory prediction using global movement data
3. Add configurable distance thresholds for global entity relevance
4. Support different entity quotas per type (e.g., max 2 cars, max 1 pedestrian from global)
5. Add visualization showing which entities are local vs global

## Key Innovation

This implementation creates a **two-tier observation system**:
- **Tier 1 (Local):** High-resolution spatial info for nearby entities
- **Tier 2 (Global):** Coarse-grained info for critical distant entities

The agent gets "detailed nearby + critical distant" instead of just "detailed nearby", enabling better long-term planning and safety-aware navigation.


