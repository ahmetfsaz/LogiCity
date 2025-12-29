"""
Global Navigation Assistant (GNA) Module
=========================================

PURPOSE:
    The GNA acts as a "god's eye view" system that collects information from ALL agents
    in the simulation and broadcasts a filtered subset (top-k) to each agent. This allows
    agents to have awareness of important entities beyond their local Field of View (FOV).

KEY CONCEPTS:
    - Pre-grounding: Collecting agent state BEFORE Z3 logical reasoning happens
    - Global Context: Information about all agents in the environment
    - Top-K Selection: Only the K most important entities are broadcast (assuming communication bw limits)
    - Priority-based Selection: Entities are ranked by their importance in reasoning

WORKFLOW (called each timestep):
    1. COLLECT: Gather data from all agents (position, type, concepts, intersection state)
    2. RANK: Sort entities by priority (based on rule occurrence counts or goal oriented states)
    3. FILTER: Select top-k most important entities
    4. BROADCAST: Send filtered context to all agents for use in local reasoning

INTEGRATION POINTS:
    - Called by City.update() or CityEnv.update_rl() each timestep
    - Broadcasts are received by agents via receive_global_context() in basic.py
    - Global entities are integrated into Z3 reasoning in z3.py and z3_rl.py

CONFIGURATION (in YAML config files):
    - enable_gna: true/false - Whether GNA is active
    - gna_top_k: integer - How many entities to include in broadcast
    - gna_selection_mode: "priority" or "random" - How to select entities, using priority ranking or random selection
"""

import time
import logging
import numpy as np
import torch
from ..core.config import *


class GlobalNavigationAssistant:
    """
    Global Navigation Assistant (GNA) that collects pre-grounding data from all agents
    and broadcasts global context to enable enhanced local reasoning with global awareness.
    
    The GNA serves as a centralized information aggregator that:
    1. Sees ALL agents in the environment (omniscient view)
    2. Extracts relevant properties from each agent
    3. Ranks entities by their importance to traffic rules
    4. Broadcasts top-k most important entities to all agents
    
    This enables agents to reason about entities they can't directly see in their FOV.
    """

    # ==================================================================================
    # INITIALIZATION
    # ==================================================================================
    
    def __init__(self, top_k=5, selection_mode="priority"):
        """
        Initialize the Global Navigation Assistant.
        
        Args:
            top_k (int): Maximum number of entities to include in each broadcast.
                         - Higher k = more global awareness but more communication cost
                         - k=0 effectively disables GNA
                         - Default: 5
            
            selection_mode (str): How to select which entities to broadcast.
                         - "priority": Rank by entity importance in reasoning (ambulance > car, etc.)
                         - "random": Random selection (for baseline comparison)
                         - Default: "priority"
        """
        # History of all broadcasts (for debugging/analysis)
        self.broadcast_history = []
        
        # Current global context (full, unfiltered) - kept for reference
        self.current_global_context = {}
        
        # Whether GNA is active (can be toggled at runtime)
        self.enabled = True
        
        # Counter for unique broadcast IDs
        self.broadcast_id_counter = 0
        
        # Configuration parameters
        self.top_k = top_k  # How many entities to broadcast
        self.selection_mode = selection_mode  # "priority" or "random"

        logging.info(f"Global Navigation Assistant (GNA) initialized - "
                     f"Status: {'ENABLED' if self.enabled else 'DISABLED'}, "
                     f"Top-K: {self.top_k}, Selection Mode: {self.selection_mode}")

    # ==================================================================================
    # MAIN ORCHESTRATION - Called each timestep
    # ==================================================================================

    def orchestrate_global_reasoning(self, city_env):
        """
        MAIN ENTRY POINT - Called each timestep by City.update().
        
        Orchestrates the complete GNA cycle:
            Phase 1: Collect data from ALL agents
            Phase 2: Filter to top-k and create broadcast
            Phase 3: Distribute broadcast to all agents
        
        Args:
            city_env: The City or CityEnv instance containing:
                      - city_env.agents: List of all agent objects
                      - city_env.city_grid: Current world state tensor
                      - city_env.intersection_matrix: Intersection information
        
        Returns:
            dict: The broadcast object containing filtered global context,
                  or None if GNA is disabled.
        
        Side Effects:
            - Calls city_env.set_global_context_for_agents() to distribute broadcast
            - Updates self.broadcast_history
        """
        # Skip if GNA is disabled
        if not self.enabled:
            logging.debug("GNA: Orchestration skipped - GNA is disabled")
            return None

        logging.info("GNA: Starting orchestration cycle")

        # -------------------------------------------------------------------------
        # PHASE 1: COLLECT - Gather pre-grounding data from ALL agents
        # -------------------------------------------------------------------------
        # This creates a comprehensive snapshot of the entire simulation state
        global_context = self.collect_pre_grounding_global(
            city_env.agents,           # List of all agent objects
            city_env.city_grid,        # World state tensor [layers, width, height]
            city_env.intersection_matrix  # Intersection data [channels, width, height]
        )

        # -------------------------------------------------------------------------
        # PHASE 2: BROADCAST - Filter to top-k and create broadcast message
        # -------------------------------------------------------------------------
        # This ranks entities, selects top-k, and packages into broadcast format
        broadcast = self.broadcast_global_context(global_context)

        # -------------------------------------------------------------------------
        # PHASE 3: DISTRIBUTE - Send broadcast to all agents
        # -------------------------------------------------------------------------
        # Each agent will receive this via receive_global_context() in basic.py
        city_env.set_global_context_for_agents(broadcast)

        logging.info("GNA: Orchestration cycle completed successfully")
        return broadcast

    # ==================================================================================
    # PHASE 1: DATA COLLECTION - Gathering information from all agents
    # ==================================================================================

    def collect_pre_grounding_global(self, all_agents, city_grid, intersection_matrix):
        """
        Collect comprehensive pre-grounding context from ALL agents in the scene.
        
        This method creates a detailed snapshot of each agent's state including:
        - Agent properties (type, position, goal, priority, concepts)
        - Intersection state (is_at_intersection, is_in_intersection)
        - Environmental context (nearby intersections, traffic density)
        - FOV entities (what agents can see in their local view)
        
        The "pre-grounding" name indicates this data is collected BEFORE
        the Z3 logical reasoning step, allowing agents to incorporate
        global awareness into their local decision making.
        
        Args:
            all_agents: List of all agent objects in the environment.
                        Each agent has properties like: type, id, pos, goal, 
                        concepts, priority, layer_id, last_move_dir
            
            city_grid: Current city grid state tensor.
                       Shape: [num_layers, width, height]
                       Each layer contains one agent's position
            
            intersection_matrix: Intersection state tensor.
                                 Shape: [3, width, height]
                                 Channel 0: Car "at intersection" zones
                                 Channel 1: Pedestrian "at intersection" zones
                                 Channel 2: "In intersection" zones (both types)
        
        Returns:
            dict: Global context dictionary with structure:
                  {
                      "Car_1": {
                          "world_state": {...},      # Agent's FOV view
                          "fov_entities": [...],     # Entities in FOV
                          "agent_properties": {...}, # Type, pos, concepts, etc.
                          "environmental_context": {...},  # Traffic, intersections
                          "collection_timestamp": float
                      },
                      "Pedestrian_2": {...},
                      ...
                  }
        """
        logging.info(f"GNA: Starting collection of pre-grounding data from {len(all_agents)} agents")
        global_context = {}

        # Iterate through ALL agents in the simulation
        for agent in all_agents:
            # Skip None entries (can happen with sparse agent lists)
            if agent is None:
                continue

            # Create unique identifier for this agent (e.g., "Car_1", "Pedestrian_2")
            agent_id = f"{agent.type}_{agent.id}"

            # -----------------------------------------------------------------
            # Extract agent's local world view (their FOV region of the grid)
            # -----------------------------------------------------------------
            # This captures what the agent would "see" for local reasoning
            agent_world_view = self.extract_agent_world_view(
                agent, city_grid, intersection_matrix
            )

            # -----------------------------------------------------------------
            # Get list of entities within this agent's FOV
            # -----------------------------------------------------------------
            fov_entities = self.get_agent_fov_entities(agent, city_grid)

            # -----------------------------------------------------------------
            # Extract agent's concept attributes (ambulance, police, old, etc.)
            # -----------------------------------------------------------------
            # Concepts define special agent types that affect traffic rules
            agent_concepts = agent.concepts if hasattr(agent, 'concepts') else {}
            
            # -----------------------------------------------------------------
            # Get agent's absolute position in world coordinates
            # -----------------------------------------------------------------
            # Convert tensor to list if needed for JSON serialization
            agent_pos = agent.pos.tolist() if hasattr(agent.pos, 'tolist') else agent.pos
            
            # -----------------------------------------------------------------
            # Determine intersection state at agent's current position
            # -----------------------------------------------------------------
            # These are pre-computed here so Z3 can use them for global entities
            # without needing to access the world_matrix
            is_at_intersection = False
            is_in_intersection = False
            
            if len(agent_pos) == 2 and intersection_matrix is not None:
                pos_x, pos_y = int(agent_pos[0]), int(agent_pos[1])
                
                # IMPORTANT: Cars and Pedestrians use DIFFERENT intersection channels!
                # This must match the logic in pred_converter/z3.py IsAtInter and IsInInter
                if agent.type == "Car" or "car" in agent.type.lower():
                    # Cars use intersection_matrix[0] for "at intersection"
                    # Cars use intersection_matrix[2] for "in intersection"
                    if (intersection_matrix.shape[0] > 0 and 
                        pos_x < intersection_matrix.shape[1] and 
                        pos_y < intersection_matrix.shape[2]):
                        is_at_intersection = bool(intersection_matrix[0, pos_x, pos_y].item())
                    if (intersection_matrix.shape[0] > 2 and 
                        pos_x < intersection_matrix.shape[1] and 
                        pos_y < intersection_matrix.shape[2]):
                        is_in_intersection = bool(intersection_matrix[2, pos_x, pos_y].item())
                else:
                    # Pedestrians use intersection_matrix[1] for "at intersection"
                    # Pedestrians use intersection_matrix[2] for "in intersection"
                    if (intersection_matrix.shape[0] > 1 and 
                        pos_x < intersection_matrix.shape[1] and 
                        pos_y < intersection_matrix.shape[2]):
                        is_at_intersection = bool(intersection_matrix[1, pos_x, pos_y].item())
                    if (intersection_matrix.shape[0] > 2 and 
                        pos_x < intersection_matrix.shape[1] and 
                        pos_y < intersection_matrix.shape[2]):
                        is_in_intersection = bool(intersection_matrix[2, pos_x, pos_y].item())
            
            # -----------------------------------------------------------------
            # Package all agent properties into a single dictionary
            # -----------------------------------------------------------------
            agent_properties = {
                'type': agent.type,              # "Car" or "Pedestrian"
                'position': agent_pos,           # [x, y] world coordinates
                'goal': agent.goal.tolist() if hasattr(agent.goal, 'tolist') else agent.goal,
                'current_action': agent.last_move_dir,  # "Left", "Right", "Up", "Down", or None
                'priority': agent.priority,      # Numeric priority for right-of-way
                'layer_id': agent.layer_id,      # Layer index in city_grid
                'concepts': agent_concepts,      # Dict like {'ambulance': 1, 'type': 'Car'}
                'is_at_intersection': is_at_intersection,   # Pre-computed for Z3
                'is_in_intersection': is_in_intersection    # Pre-computed for Z3
            }

            # Debug logging for first few agents
            if agent.id <= 3:
                logging.debug(f"GNA: Agent {agent_id} concepts: {agent_concepts}")

            # -----------------------------------------------------------------
            # Gather environmental context around this agent
            # -----------------------------------------------------------------
            environmental_context = {
                'nearby_intersections': self.get_nearby_intersections(
                    agent.pos, intersection_matrix
                ),
                'traffic_conditions': self.analyze_local_traffic(
                    agent.pos, city_grid
                ),
                'movable_region': (agent.movable_region.tolist() 
                                   if hasattr(agent, 'movable_region') else None)
            }

            # -----------------------------------------------------------------
            # Store complete agent context in global dictionary
            # -----------------------------------------------------------------
            global_context[agent_id] = {
                'world_state': agent_world_view,
                'fov_entities': fov_entities,
                'agent_properties': agent_properties,
                'environmental_context': environmental_context,
                'collection_timestamp': time.time()
            }
            logging.debug(f"GNA: Collected data for agent {agent_id} "
                          f"(type: {agent.type}, position: {agent.pos})")

        logging.info(f"GNA: Successfully collected pre-grounding data from "
                     f"{len(global_context)} agents")
        return global_context

    # ==================================================================================
    # PHASE 2A: ENTITY TYPE IDENTIFICATION - Determining what kind of entity an agent is
    # ==================================================================================

    def get_entity_type_from_concepts(self, concepts):
        """
        Determine the most specific entity type from an agent's concepts dictionary.
        
        Agents can have multiple concepts (e.g., a Car that is also an Ambulance),
        but we need a single canonical type for priority ranking. This method
        returns the MOST SPECIFIC type based on a priority order.
        
        Priority Order (most specific first):
            Bus > Ambulance > Old > Tiro > Police > Young > Reckless > Car/Pedestrian
        
        IMPORTANT: The returned type names MUST match Z3 naming conventions
        exactly for sub-rule matching to work correctly.
        
        Args:
            concepts: Dictionary of agent concepts.
                      Example: {'type': 'Car', 'ambulance': 1, 'priority': 1}
                      Active concepts have value 1 or 1.0
        
        Returns:
            str: Canonical entity type name (e.g., "Ambulance", "Police", "Car")
        """
        if not isinstance(concepts, dict):
            return "Unknown"

        # Define the priority order for concept checking
        # CRITICAL: This list ensures DETERMINISTIC type selection when
        # an agent has multiple active concepts
        # Format: (concept_key_in_dict, canonical_type_name)
        concept_priority_order = [
            ('bus', 'Bus'),           # Highest priority - special vehicle
            ('ambulance', 'Ambulance'),
            ('old', 'Old'),           # Elderly pedestrian
            ('tiro', 'Tiro'),         # Inexperienced driver
            ('police', 'Police'),
            ('young', 'Young'),       # Young pedestrian
            ('reckless', 'Reckless')  # Reckless driver
        ]

        # Check concepts in priority order
        # We accept both int (1) and float (1.0) since YAML loading can vary
        for concept_key, entity_type in concept_priority_order:
            if concept_key in concepts and concepts[concept_key] in [1, 1.0]:
                return entity_type

        # No special concept found - fall back to base type
        base_type = concepts.get('type', 'Unknown')
        if base_type == 'Car':
            return 'Car'
        elif base_type == 'Pedestrian':
            return 'Pedestrian'

        return base_type

    # ==================================================================================
    # PHASE 2B: PRIORITY CALCULATION - Ranking entities by importance
    # ==================================================================================

    def get_entity_priority(self, agent_data, ego_agent_data=None):
        """
        Calculate the priority score for an entity.
        
        Higher score = MORE important entity = should be included in top-k broadcast.
        
        Priority is based on how often this entity type appears in traffic sub-rules.
        Entities that appear in more rules are more "critical" to correct reasoning.
        If all rules are treated as equally important, then the number of appearances are weighted by the number of rules in calculating the priority score.

        Args:
            agent_data: Dictionary containing agent information from global_context.
                        Must have 'agent_properties' with 'concepts' key.
    
        Returns:
            float: Priority score. Higher = more important.
        """
        agent_properties = agent_data['agent_properties']
        concepts = agent_properties.get('concepts', {})

        # Get the canonical entity type from concepts
        entity_type = self.get_entity_type_from_concepts(concepts)

        # =====================================================================
        # GNA PRIORITY SCORES - Goal-Oriented Weighted Ratios
        # =====================================================================
        # These values are used for GNA entity ranking/selection (top-k broadcast).
        # Higher number = higher priority = more likely to be included in broadcast.
        #
        # NOTE: This is DIFFERENT from ENTITY_OCCURRENCE_SCORES in config.py!
        #   - entity_priority_map (here): Goal-oriented OR entity count-based weighted ratios for GNA ranking
        #   - ENTITY_OCCURRENCE_SCORES (config.py): Pure occurrence counts for sub-rule analysis
        #
        # The goal-oriented weighting considers:
        #   - How often the entity appears in sub-rules (occurrence count)
        #   - How critical the entity is for goal-oriented decisions (weighted ratio for stop, fast, slow rules)
        #
        # Example: Police has only 4 occurrences but highest priority (8) because of weighting for stop, fast, slow rules.
        # =====================================================================
        entity_priority_map = {
            "Ambulance": 6,   # 7 occurrences / 0.41 goal-oriented weighted ratio
            "Old": 3,         # 5 occurrences / 0.29 goal-oriented weighted ratio
            "Police": 8,      # 4 occurrences / 0.53 goal-oriented weighted ratio
            "Bus": 4,         # 2 occurrences / 0.31 goal-oriented weighted ratio
            "Pedestrian": 2,  # 2 occurrences / 0.23 goal-oriented weighted ratio
            "Reckless": 7,    # 2 occurrences / 0.50 goal-oriented weighted ratio
            "Tiro": 5,        # 2 occurrences / 0.33 goal-oriented weighted ratio
            "Young": 5,       # 2 occurrences / 0.33 goal-oriented weighted ratio
            "Car": 1,         # 1 occurrence  / 0.06 goal-oriented weighted ratio
        }
        # =====================================================================

        return entity_priority_map.get(entity_type, 0)

    # It is possible to add a relevance adjustment based on distance to ego agent. Closer an entity is to the ego agent, the higher its priority.
    # Similarly, if same entity is seen in multiple places, the priority is higher the closer it is to the ego agent.
    # However, this is not implemented in this version of the GNA.

    # ==================================================================================
    # PHASE 2C: ENTITY RANKING AND SELECTION
    # ==================================================================================

    def rank_entities_by_priority(self, global_context):
        """
        Rank ALL entities by priority and prepare for top-k selection.
        
        This method:
        1. Calculates priority score for each entity
        2. Sorts entities by final priority (highest first)
        3. Logs distribution statistics
        
        Args:
            global_context: Complete global context dict from collect_pre_grounding_global()
        
        Returns:
            list: List of (agent_id, priority_score) tuples, sorted by priority descending.
                  Example: [("Ambulance_3", 8.5), ("Police_2", 7.2), ("Car_1", 1.0)]
        """
        entity_priorities = []  # Will store (agent_id, priority) tuples
        entity_type_counts = {}  # For logging type distribution
        all_positions = []  # For relevance calculation

        # -----------------------------------------------------------------
        # First pass: Collect all positions for cross-agent distance calculation
        # -----------------------------------------------------------------
        for agent_id, agent_data in global_context.items():
            pos = agent_data['agent_properties']['position']
            if isinstance(pos, list):
                all_positions.append(pos)
            else:
                pos_tensor = (torch.tensor(pos) 
                              if not isinstance(pos, torch.Tensor) 
                              else pos)
                all_positions.append(pos_tensor.tolist())

        # -----------------------------------------------------------------
        # Second pass: Calculate priority for each agent
        # -----------------------------------------------------------------
        for agent_id, agent_data in global_context.items():
            # Get base priority from entity type
            base_priority = self.get_entity_priority(agent_data)

            # Calculate minimum distance to any other agent
            # (for optional relevance boost)
            min_distance_to_any_agent = float('inf')
            agent_pos = agent_data['agent_properties']['position']
            if isinstance(agent_pos, list):
                agent_pos_list = agent_pos
            else:
                agent_pos_tensor = (torch.tensor(agent_pos) 
                                    if not isinstance(agent_pos, torch.Tensor) 
                                    else agent_pos)
                agent_pos_list = agent_pos_tensor.tolist()
            
            # Track entity type for statistics
            concepts = agent_data['agent_properties'].get('concepts', {})
            entity_type = self.get_entity_type_from_concepts(concepts)

            entity_priorities.append((agent_id, base_priority))
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

            # Debug logging
            if isinstance(concepts, dict) and concepts:
                logging.debug(f"GNA: Agent {agent_id} - Entity Type: {entity_type}, "
                              f"Base Priority: {base_priority}, "
                              f"Concepts: {list(concepts.keys())}")
            else:
                logging.debug(f"GNA: Agent {agent_id} - Entity Type: {entity_type}, "
                              f"Base Priority: {base_priority}, No concepts")

        # -----------------------------------------------------------------
        # Sort by priority (higher = better, so reverse=True)
        # -----------------------------------------------------------------
        entity_priorities.sort(key=lambda x: x[1], reverse=True)

        # Log statistics
        logging.info(f"GNA: Entity type distribution: {entity_type_counts}")

        # Create informative log of top 10 priorities
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
        Randomly select top-k entities (for baseline comparison experiments).
        
        Args:
            global_context: Full global context dictionary
        
        Returns:
            list: List of (agent_id, priority_score) tuples.
                  Priority is set to 0 for all (not meaningful in random mode).
        """
        import random
        import time

        # Create separate RNG to avoid being affected by global seed
        # time.time_ns() ensures different selection each call
        rng = random.Random(time.time_ns())

        all_agent_ids = list(global_context.keys())
        
        # If we have fewer agents than top_k, just return all
        if len(all_agent_ids) <= self.top_k:
            return [(agent_id, 0) for agent_id in all_agent_ids]

        # Randomly sample top_k agents
        selected_ids = rng.sample(all_agent_ids, self.top_k)
        return [(agent_id, 0) for agent_id in selected_ids]

    def filter_top_k_entities(self, global_context):
        """
        Filter global context to include only the top-k most important entities.
        
        Supports two selection modes:
            - "priority": Use get_entity_priority() ranking
            - "random": Random selection (for experiments)
        
        Args:
            global_context: Full global context dict with all agents
        
        Returns:
            dict: Filtered global context with only top-k entities
        """
        # k=0 means no global context (GNA effectively disabled)
        if self.top_k <= 0:
            return {}

        # Select entities based on mode
        if self.selection_mode == "priority":
            # Priority-based selection (default)
            ranked_entities = self.rank_entities_by_priority(global_context)
            selected_entities = ranked_entities[:self.top_k]
        elif self.selection_mode == "random":
            # Random selection (for baseline experiments)
            selected_entities = self.select_random_entities(global_context)
        else:
            raise ValueError(f"Unknown GNA selection mode: {self.selection_mode}. "
                             f"Must be 'priority' or 'random'")

        # Extract selected agent IDs
        top_k_ids = {agent_id for agent_id, _ in selected_entities}

        # Filter the global context to only include selected entities
        filtered_context = {
            agent_id: global_context[agent_id]
            for agent_id in top_k_ids
            if agent_id in global_context
        }

        selection_method = ("priority-based" if self.selection_mode == "priority" 
                            else "random")
        logging.debug(f"GNA: Filtered to top-{self.top_k} entities ({selection_method}) "
                      f"from {len(global_context)} total. "
                      f"Selected: {list(filtered_context.keys())}")
        
        return filtered_context

    # ==================================================================================
    # PHASE 2D: BROADCAST CREATION - Packaging filtered context for distribution
    # ==================================================================================

    def broadcast_global_context(self, global_context):
        """
        Create and package the broadcast message containing filtered global context.
        
        The broadcast object contains:
            - Unique broadcast ID for tracking
            - Timestamp for temporal ordering
            - Filtered global context (top-k entities only)
            - Metadata (counts, timing, etc.)
        
        Args:
            global_context: Complete global context from collect_pre_grounding_global()
        
        Returns:
            dict: Broadcast object with structure:
                  {
                      "broadcast_id": "gna_broadcast_42",
                      "timestamp": 1703712000.123,
                      "global_context": {filtered entities},
                      "metadata": {
                          "total_agents": 20,
                          "filtered_agents": 5,
                          "top_k": 5,
                          ...
                      }
                  }
        """
        # Filter to top-k entities based on selection mode
        filtered_context = self.filter_top_k_entities(global_context)

        # Create broadcast object
        broadcast = {
            'broadcast_id': f"gna_broadcast_{self.broadcast_id_counter}",
            'timestamp': time.time(),
            'global_context': filtered_context,  # Only filtered entities!
            'metadata': {
                'total_agents': len(global_context),    # Original count
                'filtered_agents': len(filtered_context),  # After filtering
                'top_k': self.top_k,
                'collection_time': time.time(),
                'broadcast_size': len(str(filtered_context))  # Rough byte estimate
            }
        }

        # Keep full context in memory for analysis/debugging
        self.current_global_context = global_context
        self.broadcast_history.append(broadcast)
        self.broadcast_id_counter += 1

        # Log broadcast summary
        selection_method = ("priority-based" if self.selection_mode == "priority" 
                            else "random")
        logging.info(f"GNA: Broadcasting filtered global context - "
                     f"ID: {broadcast['broadcast_id']}, "
                     f"Total agents: {len(global_context)}, "
                     f"Filtered: {len(filtered_context)} "
                     f"(top-{self.top_k}, {selection_method})")

        # Log detailed composition if we have entities
        if filtered_context:
            # Count entity types in filtered set
            entity_types = {}
            for agent_id, data in filtered_context.items():
                concepts = data['agent_properties'].get('concepts', {})
                entity_type = self.get_entity_type_from_concepts(concepts)
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            logging.info(f"GNA: Filtered entity composition: {entity_types}")

            # Log each selected entity
            logging.info("GNA: Filtered broadcast content summary:")
            for agent_id, agent_data in filtered_context.items():
                pos = agent_data['agent_properties']['position']
                concepts = agent_data['agent_properties'].get('concepts', {})
                entity_type = self.get_entity_type_from_concepts(concepts)
                goal = agent_data['agent_properties']['goal']
                fov_entities = len(agent_data['fov_entities'])
                traffic_density = agent_data['environmental_context']['traffic_conditions']['density']
                logging.info(f"  {agent_id}: pos={pos}, goal={goal}, "
                             f"entity_type={entity_type}, fov_entities={fov_entities}, "
                             f"traffic_density={traffic_density}")
        else:
            logging.info("GNA: No entities selected for broadcast (k=0 or no agents)")

        return broadcast

    # ==================================================================================
    # HELPER METHODS - FOV and World View Extraction
    # ==================================================================================

    def extract_agent_world_view(self, agent, city_grid, intersection_matrix):
        """
        Extract the world view (FOV region) for a specific agent.
        
        This creates a copy of the city_grid and intersection_matrix
        cropped to the agent's Field of View boundaries.
        
        Args:
            agent: Agent object with pos and last_move_dir attributes
            city_grid: Full city grid tensor [layers, width, height]
            intersection_matrix: Full intersection tensor [3, width, height]
        
        Returns:
            dict: Agent's world view containing:
                  - city_grid_fov: Cropped city grid tensor
                  - intersection_fov: Cropped intersection tensor
                  - fov_boundaries: {x_start, y_start, x_end, y_end}
        """
        # Calculate FOV boundaries based on agent position and direction
        x_start, y_start, x_end, y_end = self.get_fov(
            agent.pos, 
            agent.last_move_dir,
            city_grid.shape[1],  # width
            city_grid.shape[2]   # height
        )

        # Extract the FOV regions (clone to avoid modifying originals)
        agent_world_view = {
            'city_grid_fov': city_grid[:, x_start:x_end, y_start:y_end].clone(),
            'intersection_fov': intersection_matrix[:, x_start:x_end, y_start:y_end].clone(),
            'fov_boundaries': {
                'x_start': x_start, 
                'y_start': y_start,
                'x_end': x_end, 
                'y_end': y_end
            }
        }

        return agent_world_view

    def get_agent_fov_entities(self, agent, city_grid):
        """
        Get list of entities within an agent's Field of View.
        
        Scans all layers of the city_grid within the FOV boundaries
        to find non-zero cells (which indicate agent presence).
        
        Args:
            agent: Agent object with pos and last_move_dir
            city_grid: Full city grid tensor
        
        Returns:
            list: List of entity dictionaries, each containing:
                  - layer_id: Grid layer index
                  - position: [x, y] in world coordinates
                  - entity_type: "Car" or "Pedestrian" 
                  - entity_value: Numeric value from grid
        """
        fov_entities = []

        # Get FOV boundaries
        x_start, y_start, x_end, y_end = self.get_fov(
            agent.pos, 
            agent.last_move_dir,
            city_grid.shape[1], 
            city_grid.shape[2]
        )

        # Scan each layer in the FOV region
        for layer_idx in range(city_grid.shape[0]):
            layer = city_grid[layer_idx, x_start:x_end, y_start:y_end]

            # Find non-zero cells (agent positions)
            nonzero_pos = torch.nonzero(layer, as_tuple=False)
            for pos in nonzero_pos:
                # Convert local FOV coordinates to world coordinates
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

    # ==================================================================================
    # HELPER METHODS - Environmental Context Extraction
    # ==================================================================================

    def get_nearby_intersections(self, agent_pos, intersection_matrix):
        """
        Find intersections near the agent's current position.
        
        Checks a 3x3 grid around the agent for intersection cells.
        
        Args:
            agent_pos: [x, y] position of the agent
            intersection_matrix: Intersection tensor [3, width, height]
        
        Returns:
            list: List of nearby intersection dictionaries, each containing:
                  - position: [x, y]
                  - car_intersection: bool (channel 0)
                  - pedestrian_intersection: bool (channel 1)
        """
        x, y = agent_pos
        nearby_intersections = []

        # Check 3x3 grid around agent
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                # Bounds checking
                if (0 <= nx < intersection_matrix.shape[1] and
                    0 <= ny < intersection_matrix.shape[2]):
                    # Check if this cell is an intersection
                    if intersection_matrix[0, nx, ny] or intersection_matrix[1, nx, ny]:
                        nearby_intersections.append({
                            'position': [nx, ny],
                            'car_intersection': bool(intersection_matrix[0, nx, ny]),
                            'pedestrian_intersection': bool(intersection_matrix[1, nx, ny])
                        })

        return nearby_intersections

    def analyze_local_traffic(self, agent_pos, city_grid):
        """
        Analyze traffic conditions in the local area around an agent.
        
        Counts agents within a search radius and categorizes them by type.
        
        Args:
            agent_pos: [x, y] position of the agent
            city_grid: Full city grid tensor
        
        Returns:
            dict: Traffic information containing:
                  - nearby_agents: List of {position, layer, type}
                  - density: Total agent count in area
                  - potential_conflicts: (placeholder for future use)
        """
        x, y = agent_pos
        traffic_info = {
            'nearby_agents': [],
            'density': 0,
            'potential_conflicts': []  # TODO: Implement conflict detection
        }

        # Count agents within search radius
        agent_count = 0
        search_radius = 3  # Check 7x7 grid (3 cells in each direction)

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = x + dx, y + dy
                # Bounds checking
                if (0 <= nx < city_grid.shape[1] and 
                    0 <= ny < city_grid.shape[2]):
                    # Check all layers for agents at this position
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

    # ==================================================================================
    # HELPER METHODS - FOV Calculation
    # ==================================================================================

    def get_fov(self, position, direction, width, height):
        """
        Calculate Field of View boundaries based on position and movement direction.
        
        The FOV is directional - agents see more in the direction they're moving.
        AGENT_FOV constant (from config.py) determines the FOV size.
        
        FOV shapes by direction:
            - None (stationary): Square FOV centered on position
            - Left: FOV extends left, narrow on right
            - Right: FOV extends right, narrow on left
            - Up: FOV extends up, narrow below
            - Down: FOV extends down, narrow above
        
        NOTE: This is copied from z3.py for consistency - changes should
        be synchronized between the two files.
        
        Args:
            position: [x, y] agent position
            direction: "Left", "Right", "Up", "Down", or None
            width: Grid width for bounds clamping
            height: Grid height for bounds clamping
        
        Returns:
            tuple: (x_start, y_start, x_end, y_end) FOV boundaries
        """
        if direction == None:
            # Stationary: symmetric square FOV
            x_start = max(position[0] - AGENT_FOV, 0)
            y_start = max(position[1] - AGENT_FOV, 0)
            x_end = min(position[0] + AGENT_FOV + 1, width)
            y_end = min(position[1] + AGENT_FOV + 1, height)
        elif direction == "Left":
            # Moving left: full FOV left, narrow (2 cells) right
            x_start = max(position[0] - AGENT_FOV, 0)
            y_start = max(position[1] - AGENT_FOV, 0)
            x_end = min(position[0] + AGENT_FOV + 1, width)
            y_end = min(position[1] + 2, height)
        elif direction == "Right":
            # Moving right: narrow left, full FOV right
            x_start = max(position[0] - AGENT_FOV, 0)
            y_start = max(position[1] - 2, 0)
            x_end = min(position[0] + AGENT_FOV + 1, width)
            y_end = min(position[1] + AGENT_FOV + 1, height)
        elif direction == "Up":
            # Moving up: full FOV up, narrow below
            x_start = max(position[0] - AGENT_FOV, 0)
            y_start = max(position[1] - AGENT_FOV, 0)
            x_end = min(position[0] + 2, width)
            y_end = min(position[1] + AGENT_FOV + 1, height)
        elif direction == "Down":
            # Moving down: narrow above, full FOV below
            x_start = max(position[0] - 2, 0)
            y_start = max(position[1] - AGENT_FOV, 0)
            x_end = min(position[0] + AGENT_FOV + 1, width)
            y_end = min(position[1] + AGENT_FOV + 1, height)
        else:
            # Default case (unknown direction): symmetric square
            x_start = max(position[0] - AGENT_FOV, 0)
            y_start = max(position[1] - AGENT_FOV, 0)
            x_end = min(position[0] + AGENT_FOV + 1, width)
            y_end = min(position[1] + AGENT_FOV + 1, height)

        return x_start, y_start, x_end, y_end

    # ==================================================================================
    # CONTROL METHODS - Enable/Disable/Reset
    # ==================================================================================

    def enable(self):
        """
        Enable GNA broadcasting.
        
        When enabled, orchestrate_global_reasoning() will collect and broadcast
        global context. Call this to resume GNA after it was disabled.
        """
        self.enabled = True
        logging.info("GNA: ENABLED - Global Navigation Assistant is now active")

    def disable(self):
        """
        Disable GNA broadcasting.
        
        When disabled, orchestrate_global_reasoning() returns None without
        collecting any data. Useful for baseline experiments or performance testing.
        """
        self.enabled = False
        logging.info("GNA: DISABLED - Global Navigation Assistant is now inactive")

    def clear_history(self):
        """
        Clear broadcast history and reset counter.
        
        Useful between episodes or for memory management in long simulations.
        """
        self.broadcast_history = []
        self.broadcast_id_counter = 0
