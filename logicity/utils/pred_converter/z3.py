"""
Predicate Converter for Z3 Reasoning

[MODIFIED] Key changes for GNA integration and coordinate system consistency:

1. ATTRIBUTE PREDICATES (IsAmb, IsBus, IsPolice, etc.):
   - Now check for "global_{layer_id}" keys in addition to regular and ego keys
   - This allows GNA broadcast entities to be evaluated by attribute predicates

2. NEW HELPER FUNCTION (_get_entity_position):
   - Unified position retrieval for ALL entities (local FOV and GNA global)
   - Returns world_pos from PesudoAgent.world_pos (absolute world coordinates)
   - Ensures consistent coordinate system for all spatial predicates

3. SPATIAL PREDICATES (IsClose, CollidingClose, LeftOf, RightOf, NextTo):
   - Now use _get_entity_position() for position retrieval
   - All distance calculations use absolute world coordinates
   - Consistent comparisons between local and global entities

4. INTERSECTION PREDICATES (IsAtInter, IsInInter):
   - For GNA entities: Use pre-computed is_at_intersection/is_in_intersection
   - For local entities: Continue using intersection_matrix lookup
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ...core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def IsCar(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    if "Car" in entity:
        return 1
    else:
        return 0
    
def IsPed(world_matrix, intersect_matrix, agents, entity):
    if "PH" in entity:
        return 0
    if "Pedestrian" in entity:
        return 1
    else:
        return 0

def IsAmb(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "ambulance" in agent_concept:
        return 1
    else:
        return 0

def IsBus(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "bus" in agent_concept:
        return 1
    else:
        return 0
    
def IsPolice(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "police" in agent_concept:
        return 1
    else:
        return 0

def IsTiro(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "tiro" in agent_concept:
        return 1
    else:
        return 0

def IsReckless(world_matrix, intersect_matrix, agents, entity):
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "reckless" in agent_concept:
        return 1
    else:
        return 0
    
def IsOld(world_matrix, intersect_matrix, agents, entity):
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "old" in agent_concept:
        return 1
    else:
        return 0
    
def IsYoung(world_matrix, intersect_matrix, agents, entity):
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    elif "ego_{}".format(layer_id) in agents.keys():
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    elif "global_{}".format(layer_id) in agents.keys():
        agent_concept = agents["global_{}".format(layer_id)].concepts
    else:
        # Entity not found in any expected location
        return 0
    if "young" in agent_concept:
        return 1
    else:
        return 0
    
def _get_entity_position(world_matrix, agents, entity_name, agent_type, layer_id):
    """
    [MODIFIED] Helper function to get entity position in WORLD coordinates.
    
    Now ALWAYS returns world coordinates from the stored world_pos attribute,
    ensuring consistent coordinate system for ALL entities (local FOV and GNA global).
    
    The world_pos is computed when PesudoAgent is created:
    - Local entities: world_pos = local_pos + (x_start, y_start)  
    - Global entities: world_pos = original GNA position
    
    Args:
        world_matrix: Local FOV world matrix (NOT used for position anymore)
        agents: Dict of PesudoAgent objects
        entity_name: Full entity name (e.g., "Car_Agent_5")
        agent_type: Entity type ("Car" or "Pedestrian")
        layer_id: Layer index
    
    Returns:
        torch.Tensor: [x, y] position in WORLD coordinates, or None if not found.
    """
    import torch
    
    # Find the agent in the agents dict (could be under different key formats)
    agent_obj = agents.get(str(layer_id), 
                           agents.get(f"ego_{layer_id}", 
                           agents.get(f"global_{layer_id}")))
    if agent_obj is None:
        return None
    
    # [MODIFIED] Now ALWAYS use stored world_pos for consistent coordinates
    # This works for both local FOV entities and GNA global entities
    if hasattr(agent_obj, 'world_pos') and agent_obj.world_pos is not None:
        world_pos = agent_obj.world_pos
        if isinstance(world_pos, torch.Tensor):
            return world_pos
        else:
            return torch.tensor(world_pos)
    
    # Fallback for legacy code: check global_pos (deprecated, for backwards compatibility)
    if hasattr(agent_obj, 'global_pos') and agent_obj.global_pos is not None:
        global_pos = agent_obj.global_pos
        if isinstance(global_pos, torch.Tensor):
            return global_pos
        else:
            return torch.tensor(global_pos)
    
    # Last resort fallback: try matrix lookup (should not happen with new code)
    # This is only reached if PesudoAgent was created without world_pos
    logger.warning(f"Entity {entity_name} has no world_pos - falling back to matrix lookup (may cause coordinate issues)")
    if layer_id >= world_matrix.shape[0]:
        return None
    
    agent_layer = world_matrix[layer_id]
    agent_positions = (agent_layer == TYPE_MAP[agent_type]).nonzero()
    if len(agent_positions) == 0:
        return None
    
    return agent_positions[0]

# =============================================================================
# [MODIFIED] INTERSECTION PREDICATES - Handle both local and GNA global entities
# =============================================================================
def IsAtInter(world_matrix, intersect_matrix, agents, entity1):
    """
    [MODIFIED] Check if entity is at an intersection (about to enter).
    
    For GNA global entities: Uses pre-computed is_at_intersection from GNA.
    For local FOV entities: Uses intersection_matrix lookup (local coordinates).
    """
    if "PH" in entity1:
        return 0

    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    
    # [FIXED] Check for all possible key formats including global entities
    agent_obj = agents.get(str(layer_id), agents.get(f"ego_{layer_id}", agents.get(f"global_{layer_id}")))
    if agent_obj is None:
        return 0  # Entity not found
    
    # [ADDED] GNA global entity handling - use pre-computed intersection state
    if hasattr(agent_obj, 'in_fov_matrix') and not agent_obj.in_fov_matrix:
        # Use pre-computed intersection info from GNA
        return 1 if agent_obj.is_at_intersection else 0
    
    # Regular FOV entity - use world_matrix
    if layer_id >= world_matrix.shape[0]:
        return 0  # Layer doesn't exist in world matrix
    
    agent_layer = world_matrix[layer_id]
    agent_positions = (agent_layer == TYPE_MAP[agent_type]).nonzero()
    if len(agent_positions) == 0:
        return 0  # Entity not found in world matrix
    
    agent_position = agent_positions[0]
    # at intersection needs to care if the car is "entering" or "leaving", so use intersect_matrix[0]
    if agent_type == "Car":
        if intersect_matrix[0, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0
    else:
        if intersect_matrix[1, agent_position[0], agent_position[1]]:
            return 1
        else:
            return 0

def IsInInter(world_matrix, intersect_matrix, agents, entity1):
    """
    [MODIFIED] Check if entity is inside an intersection.
    
    For GNA global entities: Uses pre-computed is_in_intersection from GNA.
    For local FOV entities: Uses intersection_matrix lookup (local coordinates).
    """
    if "PH" in entity1:
        return 0
    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    
    # [FIXED] Check for all possible key formats including global entities
    agent_obj = agents.get(str(layer_id), agents.get(f"ego_{layer_id}", agents.get(f"global_{layer_id}")))
    if agent_obj is None:
        return 0  # Entity not found
    
    # [ADDED] GNA global entity handling - use pre-computed intersection state
    if hasattr(agent_obj, 'in_fov_matrix') and not agent_obj.in_fov_matrix:
        # Use pre-computed intersection info from GNA
        return 1 if agent_obj.is_in_intersection else 0
    
    # Regular FOV entity
    if layer_id >= world_matrix.shape[0]:
        return 0
    
    agent_layer = world_matrix[layer_id]
    agent_positions = (agent_layer == TYPE_MAP[agent_type]).nonzero()
    if len(agent_positions) == 0:
        return 0
    
    agent_position = agent_positions[0]
    if intersect_matrix[2, agent_position[0], agent_position[1]]:
        return 1
    else:
        return 0

# =============================================================================
# [MODIFIED] SPATIAL PREDICATES - Now use _get_entity_position for consistency
# =============================================================================
# All spatial predicates (IsClose, CollidingClose, LeftOf, RightOf, NextTo) now:
# 1. Use _get_entity_position() to get positions in WORLD coordinates
# 2. Support both local FOV entities and GNA global entities
# 3. Ensure consistent distance/direction calculations across all entity types
# =============================================================================

def IsClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    """[MODIFIED] Uses world coordinates via _get_entity_position()."""
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    
    # [MODIFIED] Use helper function for consistent world coordinates
    agent_position1 = _get_entity_position(world_matrix, agents, entity1, agent_type1, layer_id1)
    agent_position2 = _get_entity_position(world_matrix, agents, entity2, agent_type2, layer_id2)
    
    if agent_position1 is None or agent_position2 is None:
        return 0
    
    import torch
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis > CLOSE_RANGE_MIN and eudis <= CLOSE_RANGE_MAX:
        return 1
    else:
        return 0

def HigherPri(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    _, _, agent_layer1 = entity1.split("_")
    _, _, agent_layer2 = entity2.split("_")

    if agent_layer1 in agents.keys():
        agent_prio1 = agents[agent_layer1].priority
    elif "ego_{}".format(agent_layer1) in agents.keys():
        agent_prio1 = agents["ego_{}".format(agent_layer1)].priority
    elif "global_{}".format(agent_layer1) in agents.keys():
        agent_prio1 = agents["global_{}".format(agent_layer1)].priority
    else:
        # Entity not found in any expected location
        return 0

    if agent_layer2 in agents.keys():
        agent_prio2 = agents[agent_layer2].priority
    elif "ego_{}".format(agent_layer2) in agents.keys():
        agent_prio2 = agents["ego_{}".format(agent_layer2)].priority
    elif "global_{}".format(agent_layer2) in agents.keys():
        agent_prio2 = agents["global_{}".format(agent_layer2)].priority
    else:
        # Entity not found in any expected location
        return 0

    if agent_prio1 < agent_prio2:
        return 1
    else:
        return 0

def CollidingClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    import torch
    
    # Get positions using helper function
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    
    agent_position1 = _get_entity_position(world_matrix, agents, entity1, agent_type1, layer_id1)
    agent_position2 = _get_entity_position(world_matrix, agents, entity2, agent_type2, layer_id2)
    
    if agent_position1 is None or agent_position2 is None:
        return 0
    
    # Get the moving direction of the first agent
    agent_obj1 = agents.get(str(layer_id1), agents.get(f"ego_{layer_id1}", agents.get(f"global_{layer_id1}")))
    if agent_obj1 is None:
        agent1_dire = None
    else:
        # Handle both moving_direction (PseudoAgent) and last_move_dir (GlobalPseudoAgent)
        agent1_dire = getattr(agent_obj1, 'moving_direction', None) or getattr(agent_obj1, 'last_move_dir', None)
    
    if agent1_dire == None:
        return 0
    else:
        dist = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
        if dist > OCC_CHECK_RANGE[agent_type1]:
            return 0
        elif dist == 0:
            return np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            agent1_dire_vec = torch.tensor(DIRECTION_VECTOR[agent1_dire])
            angle = torch.acos(torch.dot(agent1_dire_vec, (agent_position2 - agent_position1)) / dist)
            if angle < OCC_CHECK_ANGEL:
                if agent_type1 == "Car":
                    # Cars will definitely stop to avoid collide
                    return 1
                else:
                    # Pedestrians will probably stop to avoid collide
                    sample = np.random.rand()
                    if sample < PED_AGGR:
                        return 1
                    else:
                        return 0
    return 0

def LeftOf(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    import torch
    
    # Get positions using helper function
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    
    agent_position1 = _get_entity_position(world_matrix, agents, entity1, agent_type1, layer_id1)
    agent_position2 = _get_entity_position(world_matrix, agents, entity2, agent_type2, layer_id2)
    
    if agent_position1 is None or agent_position2 is None:
        return 0
    
    # Get the direction of entity2
    # [FIXED] Added global_{layer_id} lookup for GNA global entities
    agent_obj2 = agents.get(str(layer_id2), agents.get(f"ego_{layer_id2}", agents.get(f"global_{layer_id2}")))
    if agent_obj2 is None:
        agent2_dire = None
    else:
        # Handle both moving_direction (PseudoAgent) and last_move_dir (GlobalPseudoAgent)
        agent2_dire = getattr(agent_obj2, 'moving_direction', None) or getattr(agent_obj2, 'last_move_dir', None)
    
    if agent2_dire == None:
        return 0
    else:
        agent2_dire_vec = torch.tensor(DIRECTION_VECTOR[agent2_dire])
        relative_pos = agent_position1 - agent_position2
        dx, dy = agent2_dire_vec # Direction Agent 2 is facing
        rx, ry = relative_pos # Vector from Agent 2 to Agent 1
        z_component = dx * ry - dy * rx
        if z_component > 0:
            return 1
        else:
            return 0

def RightOf(world_matrix, intersect_matrix, agents, entity1, entity2):
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    import torch
    
    # Get positions using helper function
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    
    agent_position1 = _get_entity_position(world_matrix, agents, entity1, agent_type1, layer_id1)
    agent_position2 = _get_entity_position(world_matrix, agents, entity2, agent_type2, layer_id2)
    
    if agent_position1 is None or agent_position2 is None:
        return 0
    
    # Get the direction of entity2
    # [FIXED] Added global_{layer_id} lookup for GNA global entities
    agent_obj2 = agents.get(str(layer_id2), agents.get(f"ego_{layer_id2}", agents.get(f"global_{layer_id2}")))
    if agent_obj2 is None:
        agent2_dire = None
    else:
        # Handle both moving_direction (PseudoAgent) and last_move_dir (GlobalPseudoAgent)
        agent2_dire = getattr(agent_obj2, 'moving_direction', None) or getattr(agent_obj2, 'last_move_dir', None)
    
    if agent2_dire == None:
        return 0
    else:
        agent2_dire_vec = torch.tensor(DIRECTION_VECTOR[agent2_dire])
        relative_pos = agent_position1 - agent_position2
        dx, dy = agent2_dire_vec # Direction Agent 2 is facing
        rx, ry = relative_pos # Vector from Agent 2 to Agent 1
        z_component = dx * ry - dy * rx
        if z_component < 0:
            return 1
        else:
            return 0

def NextTo(world_matrix, intersect_matrix, agents, entity1, entity2):
    # Next to checker, closer than Close checker
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    
    import torch
    
    # Get positions using helper function
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    
    agent_position1 = _get_entity_position(world_matrix, agents, entity1, agent_type1, layer_id1)
    agent_position2 = _get_entity_position(world_matrix, agents, entity2, agent_type2, layer_id2)
    
    if agent_position1 is None or agent_position2 is None:
        return 0
    
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis < CLOSE_RANGE_MIN:
        return 1
    else:
        return 0