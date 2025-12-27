import time
import logging
import numpy as np
import torch
from ..core.config import *


class GlobalNavigationAssistant:
    """
    Global Navigation Assistant (GNA) that collects pre-grounding data from all agents
    and broadcasts global context to enable enhanced local reasoning with global awareness.
    """

    def __init__(self, top_k=5, selection_mode="priority"):
        self.broadcast_history = []
        self.current_global_context = {}
        self.enabled = True
        self.broadcast_id_counter = 0
        self.top_k = top_k
        self.selection_mode = selection_mode  # "priority" or "random"

        logging.info(f"Global Navigation Assistant (GNA) initialized - Status: {'ENABLED' if self.enabled else 'DISABLED'}, Top-K: {self.top_k}, Selection Mode: {self.selection_mode}")

    def collect_pre_grounding_global(self, all_agents, city_grid, intersection_matrix):
        """
        Collect pre-grounding context from ALL agents in the scene.

        Args:
            all_agents: List of all agent objects in the environment
            city_grid: Current city grid state
            intersection_matrix: Current intersection state

        Returns:
            Dict containing global context from all agents
        """
        logging.info(f"GNA: Starting collection of pre-grounding data from {len(all_agents)} agents")
        global_context = {}

        for agent in all_agents:
            if agent is None:
                continue

            agent_id = f"{agent.type}_{agent.id}"

            # Extract agent's local world view (what they see in their FOV)
            agent_world_view = self.extract_agent_world_view(agent, city_grid, intersection_matrix)

            # Get FOV entities for this agent
            fov_entities = self.get_agent_fov_entities(agent, city_grid)

            # Collect agent properties
            agent_concepts = agent.concepts if hasattr(agent, 'concepts') else {}
            
            # Get absolute position in world
            agent_pos = agent.pos.tolist() if hasattr(agent.pos, 'tolist') else agent.pos
            
            # Get intersection state at agent's position
            is_at_intersection = False
            is_in_intersection = False
            if len(agent_pos) == 2 and intersection_matrix is not None:
                pos_x, pos_y = int(agent_pos[0]), int(agent_pos[1])
                if agent.type == "Car" or "car" in agent.type.lower():
                    # Cars use intersection[0] for "at" and intersection[2] for "in"
                    if intersection_matrix.shape[0] > 0 and pos_x < intersection_matrix.shape[1] and pos_y < intersection_matrix.shape[2]:
                        is_at_intersection = bool(intersection_matrix[0, pos_x, pos_y].item())
                    if intersection_matrix.shape[0] > 2 and pos_x < intersection_matrix.shape[1] and pos_y < intersection_matrix.shape[2]:
                        is_in_intersection = bool(intersection_matrix[2, pos_x, pos_y].item())
                else:
                    # Pedestrians use intersection[1] for "at" and intersection[2] for "in"
                    # Must match the logic in pred_converter/z3.py IsAtInter and IsInInter
                    if intersection_matrix.shape[0] > 1 and pos_x < intersection_matrix.shape[1] and pos_y < intersection_matrix.shape[2]:
                        is_at_intersection = bool(intersection_matrix[1, pos_x, pos_y].item())
                    if intersection_matrix.shape[0] > 2 and pos_x < intersection_matrix.shape[1] and pos_y < intersection_matrix.shape[2]:
                        is_in_intersection = bool(intersection_matrix[2, pos_x, pos_y].item())
            
            agent_properties = {
                'type': agent.type,
                'position': agent_pos,
                'goal': agent.goal.tolist() if hasattr(agent.goal, 'tolist') else agent.goal,
                'current_action': agent.last_move_dir,
                'priority': agent.priority,
                'layer_id': agent.layer_id,
                'concepts': agent_concepts,
                'is_at_intersection': is_at_intersection,
                'is_in_intersection': is_in_intersection
            }

            # Debug: Log concepts for first few agents
            if agent.id <= 3:
                logging.debug(f"GNA: Agent {agent_id} concepts: {agent_concepts}")

            # Environmental context
            environmental_context = {
                'nearby_intersections': self.get_nearby_intersections(agent.pos, intersection_matrix),
                'traffic_conditions': self.analyze_local_traffic(agent.pos, city_grid),
                'movable_region': agent.movable_region.tolist() if hasattr(agent, 'movable_region') else None
            }

            global_context[agent_id] = {
                'world_state': agent_world_view,
                'fov_entities': fov_entities,
                'agent_properties': agent_properties,
                'environmental_context': environmental_context,
                'collection_timestamp': time.time()
            }
            logging.debug(f"GNA: Collected data for agent {agent_id} (type: {agent.type}, position: {agent.pos})")

        logging.info(f"GNA: Successfully collected pre-grounding data from {len(global_context)} agents")
        return global_context

    def get_entity_type_from_concepts(self, concepts):
        """
        Determine the actual entity type based on concepts.
        Returns the most specific entity type.
        """
        if not isinstance(concepts, dict):
            return "Unknown"

        # Priority order: most specific to least specific
        # Based on user's hierarchy: Bus, Ambulance, Old, Tiro, Police, Pedestrian, Young, Reckless, Car
        # NOTE: Must match Z3 naming conventions for sub-rule matching
        # Ordered list ensures deterministic selection when multiple concepts are present
        concept_priority_order = [
            ('bus', 'Bus'),
            ('ambulance', 'Ambulance'),
            ('old', 'Old'),
            ('tiro', 'Tiro'),
            ('police', 'Police'),
            ('young', 'Young'),
            ('reckless', 'Reckless')
        ]

        # Check for specific concept types in priority order (deterministic)
        # Accept both int (1) and float (1.0) for robustness
        for concept_key, entity_type in concept_priority_order:
            if concept_key in concepts and concepts[concept_key] in [1, 1.0]:
                return entity_type

        # Fall back to base type
        base_type = concepts.get('type', 'Unknown')
        if base_type == 'Car':
            return 'Car'
        elif base_type == 'Pedestrian':
            return 'Pedestrian'

        return base_type

    def get_entity_priority(self, agent_data, ego_agent_data=None):
        """
        Get priority score for an entity based on its concepts and relevance to ego agent.
        Higher score = higher priority (based on occurrence count in sub-rules).
        """
        agent_properties = agent_data['agent_properties']
        concepts = agent_properties.get('concepts', {})

        # Determine actual entity type from concepts
        entity_type = self.get_entity_type_from_concepts(concepts)

        # Priority based on actual occurrence count in sub-rules
        # Higher number = higher priority (more critical)
        # NOTE: Must match Z3 naming conventions for sub-rule matching
        entity_priority_map = {
            "Ambulance": 6, #7,  # 7 / 0.41 occurrences - MOST critical
            "Old": 3, #6,        # 5 / 0.29 occurrences
            "Police": 8, #4,     # 4 / 0.53 occurrences
            "Bus": 4, #2,        # 2 / 0.31 occurrences (tied)
            "Pedestrian": 2, #2,    2 / 0.23 occurrences (tied)
            "Reckless": 7, #2,   # 2 / 0.50 occurrences (tied)
            "Tiro": 5, #2,       # 2 / 0.33 occurrences (tied)
            "Young": 5, #2,      # 2 / 0.33 occurrences (tied)
            "Car": 1, #1,         # 1 / 0.06 occurrences - LEAST critical
        }

        base_priority = entity_priority_map.get(entity_type, 0)

        # If ego agent data is provided, adjust priority based on relevance
        if ego_agent_data is not None:
            relevance_adjustment = self.calculate_relevance_to_ego(agent_data, ego_agent_data)
            base_priority += relevance_adjustment

        return base_priority

    def calculate_relevance_to_ego(self, agent_data, ego_agent_data):
        """
        Calculate how relevant this agent is to the ego agent's decision making.
        Returns a bonus (positive number for close agents) to add to priority score.
        Higher total priority = higher relevance (closer = more relevant).
        """
        agent_pos = agent_data['agent_properties']['position']
        ego_pos = ego_agent_data['agent_properties']['position']

        # Calculate Euclidean distance
        if isinstance(agent_pos, list) and isinstance(ego_pos, list):
            distance = ((agent_pos[0] - ego_pos[0]) ** 2 + (agent_pos[1] - ego_pos[1]) ** 2) ** 0.5
        else:
            # Handle tensor positions
            import torch
            agent_pos_tensor = torch.tensor(agent_pos) if not isinstance(agent_pos, torch.Tensor) else agent_pos
            ego_pos_tensor = torch.tensor(ego_pos) if not isinstance(ego_pos, torch.Tensor) else ego_pos
            distance = torch.dist(agent_pos_tensor.float(), ego_pos_tensor.float()).item()

        # Distance-based bonus: closer agents get priority boost
        # Agents within 10 units get bonus, beyond that no bonus
        distance_bonus = max(0, (10 - distance) / 5)  # Bonus decreases with distance

        # TODO: Could add more relevance factors:
        # - Movement direction toward ego
        # - Whether in ego's FOV
        # - Potential collision risk
        # - Shared goal/path

        return distance_bonus

    def rank_entities_by_priority(self, global_context):
        """
        Rank all entities by priority hierarchy and return top-k.

        Args:
            global_context: Dict of all agent data

        Returns:
            List of (agent_id, priority_score) tuples, sorted by priority (higher = better)
        """
        entity_priorities = []
        entity_type_counts = {}
        all_positions = []

        # Collect all agent positions for relevance calculation
        for agent_id, agent_data in global_context.items():
            pos = agent_data['agent_properties']['position']
            if isinstance(pos, list):
                all_positions.append(pos)
            else:
                # Handle tensor positions
                import torch
                pos_tensor = torch.tensor(pos) if not isinstance(pos, torch.Tensor) else pos
                all_positions.append(pos_tensor.tolist())

        for agent_id, agent_data in global_context.items():
            # Get base type priority
            base_priority = self.get_entity_priority(agent_data)

            # Add global relevance adjustment (how close this agent is to ANY other agent)
            min_distance_to_any_agent = float('inf')
            agent_pos = agent_data['agent_properties']['position']
            if isinstance(agent_pos, list):
                agent_pos_list = agent_pos
            else:
                import torch
                agent_pos_tensor = torch.tensor(agent_pos) if not isinstance(agent_pos, torch.Tensor) else agent_pos
                agent_pos_list = agent_pos_tensor.tolist()

            for other_pos in all_positions:
                if other_pos != agent_pos_list:  # Don't compare to self
                    distance = ((agent_pos_list[0] - other_pos[0]) ** 2 + (agent_pos_list[1] - other_pos[1]) ** 2) ** 0.5
                    min_distance_to_any_agent = min(min_distance_to_any_agent, distance)

            # Boost priority for agents that are close to other agents (more relevant globally)
            # Agents within 15 units of any other agent get a priority boost
            # Since higher = better now, we ADD the boost
            relevance_boost = max(0, (15 - min_distance_to_any_agent) / 3) if min_distance_to_any_agent < float('inf') else 0
            
            final_priority = base_priority  # Currently not using relevance_boost
            # To enable: final_priority = base_priority + relevance_boost

            
            
            concepts = agent_data['agent_properties'].get('concepts', {})
            entity_type = self.get_entity_type_from_concepts(concepts)

            entity_priorities.append((agent_id, final_priority))
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

            # Debug: Log agent details
            if isinstance(concepts, dict) and concepts:
                logging.debug(f"GNA: Agent {agent_id} - Entity Type: {entity_type}, Base Priority: {base_priority}, Relevance Boost: {relevance_boost:.1f}, Final Priority: {final_priority:.1f}, Concepts: {list(concepts.keys())}")
            else:
                logging.debug(f"GNA: Agent {agent_id} - Entity Type: {entity_type}, Base Priority: {base_priority}, Relevance Boost: {relevance_boost:.1f}, Final Priority: {final_priority:.1f}, No concepts")

        # Sort by priority (higher number = higher priority)
        entity_priorities.sort(key=lambda x: x[1], reverse=True)

        logging.info(f"GNA: Entity type distribution: {entity_type_counts}")

        # Create a more informative priority list showing entity types and final priorities
        top_priorities_info = []
        for agent_id, priority in entity_priorities[:10]:
            if agent_id in global_context:
                concepts = global_context[agent_id]['agent_properties'].get('concepts', {})
                entity_type = self.get_entity_type_from_concepts(concepts)
                top_priorities_info.append(f"{entity_type}({agent_id}:{priority:.1f})")
            else:
                top_priorities_info.append(f"{agent_id}:{priority:.1f}")

        logging.info(f"GNA: Top 10 priorities: {top_priorities_info}")

        return entity_priorities

    def select_random_entities(self, global_context):
        """
        Randomly select top-k entities from global context.
        Uses a separate random generator to avoid being affected by global seed.

        Args:
            global_context: Full global context dict

        Returns:
            List of randomly selected (agent_id, priority_score) tuples
        """
        import random
        import time

        # Use a separate random generator with current time as seed to ensure randomness
        # This avoids being affected by the global numpy/torch random seed
        rng = random.Random(time.time_ns())

        all_agent_ids = list(global_context.keys())
        if len(all_agent_ids) <= self.top_k:
            # If we have fewer or equal agents than top_k, return all with dummy priority
            return [(agent_id, 0) for agent_id in all_agent_ids]

        # Randomly select top_k agents using separate RNG
        selected_ids = rng.sample(all_agent_ids, self.top_k)
        return [(agent_id, 0) for agent_id in selected_ids]  # Dummy priority for random selection

    def filter_top_k_entities(self, global_context):
        """
        Filter global context to include only top-k entities using specified selection mode.

        Args:
            global_context: Full global context dict

        Returns:
            Filtered global context with only top-k entities
        """
        if self.top_k <= 0:
            return {}  # No global context if k=0

        if self.selection_mode == "priority":
            # Use priority-based selection (original behavior)
            ranked_entities = self.rank_entities_by_priority(global_context)
            selected_entities = ranked_entities[:self.top_k]
        elif self.selection_mode == "random":
            # Use random selection
            selected_entities = self.select_random_entities(global_context)
        else:
            raise ValueError(f"Unknown GNA selection mode: {self.selection_mode}. Must be 'priority' or 'random'")

        top_k_ids = {agent_id for agent_id, _ in selected_entities}

        # Filter global context
        filtered_context = {
            agent_id: global_context[agent_id]
            for agent_id in top_k_ids
            if agent_id in global_context
        }

        selection_method = "priority-based" if self.selection_mode == "priority" else "random"
        logging.debug(f"GNA: Filtered to top-{self.top_k} entities ({selection_method}) from {len(global_context)} total. Selected: {list(filtered_context.keys())}")
        return filtered_context

    def extract_agent_world_view(self, agent, city_grid, intersection_matrix):
        """
        Extract the world view that this agent would use for local reasoning.
        This includes their FOV area and relevant state information.
        """
        # Get the agent's FOV boundaries
        x_start, y_start, x_end, y_end = self.get_fov(agent.pos, agent.last_move_dir,
                                                      city_grid.shape[1], city_grid.shape[2])

        # Extract the relevant portion of the city grid
        agent_world_view = {
            'city_grid_fov': city_grid[:, x_start:x_end, y_start:y_end].clone(),
            'intersection_fov': intersection_matrix[:, x_start:x_end, y_start:y_end].clone(),
            'fov_boundaries': {
                'x_start': x_start, 'y_start': y_start,
                'x_end': x_end, 'y_end': y_end
            }
        }

        return agent_world_view

    def get_agent_fov_entities(self, agent, city_grid):
        """
        Get the entities that would be in this agent's FOV.
        This simulates what entities the agent would consider for local reasoning.
        """
        # This would normally be done by the local planner's break_world_matrix
        # For now, we'll create a simplified version
        fov_entities = []

        # Get FOV boundaries
        x_start, y_start, x_end, y_end = self.get_fov(agent.pos, agent.last_move_dir,
                                                      city_grid.shape[1], city_grid.shape[2])

        # Look for other agents in the FOV area
        for layer_idx in range(city_grid.shape[0]):
            layer = city_grid[layer_idx, x_start:x_end, y_start:y_end]

            # Find non-zero elements (agents)
            nonzero_pos = torch.nonzero(layer, as_tuple=False)
            for pos in nonzero_pos:
                actual_x, actual_y = pos[0] + x_start, pos[1] + y_start
                entity_value = layer[pos[0], pos[1]].item()

                if entity_value != 0:
                    fov_entities.append({
                        'layer_id': layer_idx,
                        'position': [actual_x, actual_y],
                        'entity_type': LABEL_MAP.get(entity_value, 'unknown'),
                        'entity_value': entity_value
                    })

        return fov_entities

    def get_nearby_intersections(self, agent_pos, intersection_matrix):
        """Get information about nearby intersections."""
        x, y = agent_pos
        nearby_intersections = []

        # Check a small area around the agent
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < intersection_matrix.shape[1] and
                    0 <= ny < intersection_matrix.shape[2]):
                    if intersection_matrix[0, nx, ny] or intersection_matrix[1, nx, ny]:
                        nearby_intersections.append({
                            'position': [nx, ny],
                            'car_intersection': bool(intersection_matrix[0, nx, ny]),
                            'pedestrian_intersection': bool(intersection_matrix[1, nx, ny])
                        })

        return nearby_intersections

    def analyze_local_traffic(self, agent_pos, city_grid):
        """Analyze local traffic conditions around the agent."""
        x, y = agent_pos
        traffic_info = {
            'nearby_agents': [],
            'density': 0,
            'potential_conflicts': []
        }

        # Count agents in nearby area
        agent_count = 0
        search_radius = 3

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < city_grid.shape[1] and 0 <= ny < city_grid.shape[2]):
                    # Check all layers for agents
                    for layer_idx in range(city_grid.shape[0]):
                        if city_grid[layer_idx, nx, ny] != 0:
                            agent_count += 1
                            entity_value = city_grid[layer_idx, nx, ny].item()
                            traffic_info['nearby_agents'].append({
                                'position': [nx, ny],
                                'layer': layer_idx,
                                'type': LABEL_MAP.get(entity_value, 'unknown')
                            })

        traffic_info['density'] = agent_count
        return traffic_info

    def broadcast_global_context(self, global_context):
        """
        Broadcast the filtered global context (top-k entities) to all agents.

        Args:
            global_context: Dict containing pre-grounding data from all agents

        Returns:
            Broadcast object ready for distribution
        """
        # Filter to top-k entities based on priority
        filtered_context = self.filter_top_k_entities(global_context)

        broadcast = {
            'broadcast_id': f"gna_broadcast_{self.broadcast_id_counter}",
            'timestamp': time.time(),
            'global_context': filtered_context,  # Use filtered context
            'metadata': {
                'total_agents': len(global_context),  # Original count
                'filtered_agents': len(filtered_context),  # Filtered count
                'top_k': self.top_k,
                'collection_time': time.time(),
                'broadcast_size': len(str(filtered_context))  # Rough size estimate
            }
        }

        self.current_global_context = global_context  # Keep full context for history
        self.broadcast_history.append(broadcast)
        self.broadcast_id_counter += 1

        selection_method = "priority-based" if self.selection_mode == "priority" else "random"
        logging.info(f"GNA: Broadcasting filtered global context - ID: {broadcast['broadcast_id']}, Total agents: {len(global_context)}, Filtered: {len(filtered_context)} (top-{self.top_k}, {selection_method})")

        if filtered_context:
            entity_types = {}
            for agent_id, data in filtered_context.items():
                concepts = data['agent_properties'].get('concepts', {})
                entity_type = self.get_entity_type_from_concepts(concepts)
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            logging.info(f"GNA: Filtered entity composition: {entity_types}")

            # Log broadcast content summary
            logging.info("GNA: Filtered broadcast content summary:")
            for agent_id, agent_data in filtered_context.items():
                pos = agent_data['agent_properties']['position']
                concepts = agent_data['agent_properties'].get('concepts', {})
                entity_type = self.get_entity_type_from_concepts(concepts)
                goal = agent_data['agent_properties']['goal']
                fov_entities = len(agent_data['fov_entities'])
                traffic_density = agent_data['environmental_context']['traffic_conditions']['density']
                logging.info(f"  {agent_id}: pos={pos}, goal={goal}, entity_type={entity_type}, fov_entities={fov_entities}, traffic_density={traffic_density}")
        else:
            logging.info("GNA: No entities selected for broadcast (k=0 or no agents)")

        return broadcast

    def orchestrate_global_reasoning(self, city_env):
        """
        Main orchestration method called each time step.

        Args:
            city_env: City environment instance

        Returns:
            Broadcast object with global context
        """
        if not self.enabled:
            logging.debug("GNA: Orchestration skipped - GNA is disabled")
            return None

        logging.info("GNA: Starting orchestration cycle")

        # Phase 1: Collect pre-grounding data from all agents
        global_context = self.collect_pre_grounding_global(
            city_env.agents, city_env.city_grid, city_env.intersection_matrix
        )

        # Phase 2: Broadcast global context
        broadcast = self.broadcast_global_context(global_context)

        # Phase 3: Make broadcast available to all agents
        city_env.set_global_context_for_agents(broadcast)

        logging.info("GNA: Orchestration cycle completed successfully")
        return broadcast

    def get_fov(self, position, direction, width, height):
        """
        Calculate FOV boundaries (copied from z3.py for consistency).
        """
        if direction == None:
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        elif direction == "Left":
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+2, height)
        elif direction == "Right":
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-2, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        elif direction == "Up":
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+2, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        elif direction == "Down":
            x_start = max(position[0]-2, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)
        else:  # Default case
            x_start = max(position[0]-AGENT_FOV, 0)
            y_start = max(position[1]-AGENT_FOV, 0)
            x_end = min(position[0]+AGENT_FOV+1, width)
            y_end = min(position[1]+AGENT_FOV+1, height)

        return x_start, y_start, x_end, y_end

    def enable(self):
        """Enable GNA broadcasting."""
        self.enabled = True
        logging.info("GNA: ENABLED - Global Navigation Assistant is now active")

    def disable(self):
        """Disable GNA broadcasting."""
        self.enabled = False
        logging.info("GNA: DISABLED - Global Navigation Assistant is now inactive")

    def clear_history(self):
        """Clear broadcast history."""
        self.broadcast_history = []
        self.broadcast_id_counter = 0

