import torch
import logging  # [ADDED] For GNA integration logging
from torch.distributions import Categorical
from ..core.config import *

# [ADDED] Logger for GNA-related debug messages
logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, size, id, world_state_matrix, concepts, init_info=None, debug=False, region=240):
        self.size = size
        self.concepts = concepts
        self.type = concepts["type"]
        self.priority = concepts["priority"]
        self.id = id
        # -1 is always the stop action
        # Actions: ["left", "right", "up", "down", "stop"]
        self.action_space = torch.tensor(range(5))
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "Left_1", 
            1: "Right_1", 
            2: "Up_1", 
            3: "Down_1", 
            4: "Stop"
            }
        # move direction will be used to determine FOV and Spatial Predicates
        self.last_move_dir = None
        self.start = None
        self.goal = None
        self.pos = None
        self.layer_id = 0
        self.reach_goal = False
        self.reach_goal_buffer = 0
        self.debug = debug
        self.region = region
        
        # =====================================================================
        # [ADDED] Global Navigation Assistant (GNA) Integration
        # =====================================================================
        # This attribute stores the GNA broadcast received each timestep.
        # It allows agents to have awareness of entities beyond their local FOV.
        # Set to None when GNA is disabled or no broadcast received yet.
        self.global_context = None
        
        self.init(world_state_matrix, init_info, debug)

    def init(self, world_state_matrix, init_info=None, debug=False):
        # init global planner, global traj, local planner, start and goal point
        pass

    def get_start(self, world_state_matrix):
        pass

    def get_goal(self, world_state_matrix):
        pass
        
    def move(self, action, ped_layer, curr_label):
        pass
        
    def get_next_action(self, world_state_matrix):
        pass

    def get_movable_area(self, world_state_matrix):
        pass

    # =========================================================================
    # [ADDED] GNA INTEGRATION METHODS
    # =========================================================================
    # These methods enable agents to receive global context from GNA about
    # entities beyond their local Field of View (FOV).
    #
    # ARCHITECTURE:
    # -------------
    # Two pseudo-agent classes in the system:
    #
    # 1. PesudoAgent (z3.py) - ORIGINAL class, we modified to add world_pos
    # 2. GlobalPseudoAgent (this file) - NEW, intermediate format for GNA → Z3
    #
    # FLOW:
    #   1. City.update() → GNA.orchestrate_global_reasoning()
    #   2. GNA broadcasts top-k entities
    #   3. City calls receive_global_context() on each agent
    #   4. Agent stores GlobalPseudoAgent objects in self.global_entities
    #   5. Z3's break_world_matrix() converts them to PesudoAgent with world_pos
    #
    # COORDINATES: All positions use WORLD coordinates for consistency.
    # =========================================================================

    def receive_global_context(self, broadcast):
        """
        [ADDED] Receive global context broadcast from GNA and prepare global entities for Z3.
        
        This method is called by City.set_global_context_for_agents() each timestep
        after GNA has collected and filtered the global context.
        
        The broadcast contains information about the top-k most important entities
        in the environment, allowing this agent to reason about entities it can't
        directly see in its FOV.
        
        Args:
            broadcast: Global context broadcast from GNA, with structure:
                       {
                           'broadcast_id': str,
                           'timestamp': float,
                           'global_context': {
                               'Car_1': {agent_data},
                               'Pedestrian_2': {agent_data},
                               ...
                           },
                           'metadata': {...}
                       }
                       Can be None if GNA is disabled.
        
        Side Effects:
            - Sets self.global_context to the received broadcast
            - Populates self.global_entities with GlobalPseudoAgent objects
              (later converted to PesudoAgent by Z3 planner)
        """
        # Store the raw broadcast for potential later use
        self.global_context = broadcast
        
        # Initialize container for global entities
        # Key: agent_id (e.g., "Car_3"), Value: GlobalPseudoAgent object
        # These will be read by Z3's break_world_matrix() and converted to PesudoAgent
        self.global_entities = {}

        if broadcast is not None:
            # Extract the filtered global context (top-k entities)
            global_context = broadcast.get('global_context', {})
            agent_count = len(global_context)
            broadcast_id = broadcast.get('broadcast_id', 'unknown')
            
            logger.debug(f"Agent {self.type}_{self.id}: Received GNA broadcast "
                         f"{broadcast_id} with {agent_count} global entities")

            # Convert each global entity to GlobalPseudoAgent format
            # Skip self - we don't need global info about ourselves
            for agent_id, agent_data in global_context.items():
                if agent_id != f"{self.type}_{self.id}":  # Don't include self
                    self.global_entities[agent_id] = self._prepare_global_entity_for_z3(agent_data)

            # Debug: Log this agent's own data from global context (if present)
            if f"{self.type}_{self.id}" in global_context:
                my_data = global_context[f"{self.type}_{self.id}"]
                my_pos = my_data['agent_properties']['position']
                my_goal = my_data['agent_properties']['goal']
                logger.debug(f"Agent {self.type}_{self.id}: My global data - "
                             f"pos={my_pos}, goal={my_goal}")

            logger.debug(f"Agent {self.type}_{self.id}: Prepared "
                         f"{len(self.global_entities)} global entities for Z3 reasoning")
        else:
            # GNA is disabled or no broadcast available
            logger.debug(f"Agent {self.type}_{self.id}: Received null GNA broadcast "
                         f"(GNA disabled)")
            self.global_entities = {}

    def _prepare_global_entity_for_z3(self, agent_data):
        """
        [ADDED] Convert GNA broadcast data into a GlobalPseudoAgent object.
        
        This creates an intermediate representation that Z3's break_world_matrix()
        can read and convert to the final PesudoAgent format.
        
        RELATIONSHIP TO PesudoAgent (z3.py):
        ------------------------------------
        - PesudoAgent is the ORIGINAL class from the codebase (we only modified it)
        - GlobalPseudoAgent is a NEW class we created for GNA integration
        - GlobalPseudoAgent is converted to PesudoAgent in break_world_matrix()
        - Both now use world_pos for consistent coordinate system
        
        KEY ATTRIBUTES:
        ---------------
        - pos: Position in WORLD coordinates (absolute grid position)
        - is_at_intersection / is_in_intersection: Pre-computed by GNA
          (needed because global entities can't use local intersection_matrix)
        
        Args:
            agent_data: Agent data dictionary from GNA broadcast.
                        Structure: {
                            'agent_properties': {
                                'type': str,
                                'layer_id': int,
                                'concepts': dict,
                                'position': [x, y],  # In WORLD coordinates
                                'current_action': str or None,
                                'goal': [x, y],
                                'is_at_intersection': bool,
                                'is_in_intersection': bool
                            },
                            ...
                        }
        
        Returns:
            GlobalPseudoAgent: Intermediate object to be converted to PesudoAgent by Z3.
        """
        agent_properties = agent_data['agent_properties']

        # Define GlobalPseudoAgent class
        # This is an INTERMEDIATE format between GNA broadcast and Z3's PesudoAgent
        class GlobalPseudoAgent:
            """
            [ADDED] Intermediate agent representation for GNA → Z3 conversion.
            
            This class bridges the gap between GNA's broadcast format and
            the PesudoAgent format that Z3 expects. When Z3's break_world_matrix()
            processes global entities, it reads GlobalPseudoAgent and creates
            the final PesudoAgent with proper world_pos coordinates.
            
            Attributes match what Z3's break_world_matrix() expects to read:
                - type, layer_id, concepts, pos, last_move_dir, goal, priority
                - is_at_intersection, is_in_intersection (pre-computed by GNA)
            """
            def __init__(self, agent_type, layer_id, concepts, position, direction, goal, 
                        is_at_intersection=False, is_in_intersection=False):
                self.type = agent_type              # "Car" or "Pedestrian"
                self.layer_id = layer_id            # Layer index in city_grid
                self.concepts = concepts if concepts else {}  # Dict of agent attributes
                self.pos = position                 # [x, y] in WORLD coordinates
                self.last_move_dir = direction      # "Left", "Right", "Up", "Down", or None
                self.goal = goal                    # [x, y] goal position
                self.priority = concepts.get('priority', 1) if concepts else 1
                
                # Pre-computed intersection states (from GNA)
                # These are needed because global entities can't look up the local
                # intersection_matrix (which is cropped to the ego's FOV)
                self.is_at_intersection = is_at_intersection
                self.is_in_intersection = is_in_intersection

        # Create and return the GlobalPseudoAgent with data from GNA broadcast
        # Position is already in WORLD coordinates from GNA
        return GlobalPseudoAgent(
            agent_type=agent_properties['type'],
            layer_id=int(agent_properties.get('layer_id', 0)),
            concepts=agent_properties.get('concepts', {}),
            position=agent_properties['position'],  # Already in WORLD coords
            direction=agent_properties.get('current_action'),
            goal=agent_properties['goal'],
            is_at_intersection=agent_properties.get('is_at_intersection', False),
            is_in_intersection=agent_properties.get('is_in_intersection', False)
        )

    # =========================================================================
    # [END OF ADDED GNA INTEGRATION METHODS]
    # =========================================================================

    def get_action(self, local_action_dist):
        """
        Combine local and global action distributions to select final action.
        
        [MODIFIED] Comments updated to reflect Z3/GNA integration.
        Note: "local_action_dist" comes from Z3 reasoning which now includes
        global entities (if GNA is enabled). The reasoning is already "globally aware".
        """
        if len(local_action_dist.nonzero()) == 1:
            # Z3 reasoning is very strict, only one action is possible
            final_action_dist = local_action_dist
        else:
            global_action = self.get_global_action()
            if len(global_action.nonzero()) == 1:
                # Z3 gives multiple actions, but global planner is very strict
                final_action_dist = global_action
            else:
                # Z3 gives multiple actions, use global planner to filter
                final_action_dist = torch.logical_and(local_action_dist, global_action).float()
                if len(final_action_dist.nonzero()) == 0:
                    # Z3 and global planner have conflict, take global
                    final_action_dist = global_action
        
        assert len(final_action_dist.nonzero()) >= 1
        normalized_action_dist = final_action_dist / final_action_dist.sum()
        dist = Categorical(normalized_action_dist)
        action_index = dist.sample()
        return self.action_space[action_index]

    def move(self, action, ped_layer):
        curr_pos = torch.nonzero((ped_layer==TYPE_MAP[self.type]).float())[0]
        assert torch.all(self.pos == curr_pos), (self.pos, curr_pos)
        next_pos = self.pos.clone()
        # becomes walked grid
        ped_layer[self.pos[0], self.pos[1]] += AGENT_WALKED_PATH_PLUS
        next_pos += self.action_to_move.get(action.item(), torch.tensor((0, 0)))
        self.pos = next_pos.clone()
        # Update Agent Map
        ped_layer[self.start[0], self.start[1]] = TYPE_MAP[self.type] + AGENT_START_PLUS
        ped_layer[self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
        ped_layer[self.pos[0], self.pos[1]] = TYPE_MAP[self.type]
        # Update last move direction
        move_dir = self.action_mapping[action.item()].split("_")[0]
        if move_dir != "Stop":
            self.last_move_dir = move_dir
        return ped_layer

    def get_global_action(self):
        global_action_dist = torch.zeros_like(self.action_space).float()
        current_pos = torch.all((self.global_traj == self.pos), dim=1).nonzero()[0]
        next_pos = current_pos + 1 if current_pos < len(self.global_traj) - 1 else 0
        del_pos = self.global_traj[next_pos] - self.pos
        for move in self.move_to_action.keys():
            if torch.dot(del_pos.squeeze(), move) > 0:
                next_point = self.pos + move
                step = torch.max(torch.abs(move)).item()
                # 1. if the next point is on the global traj
                if len(torch.all((self.global_traj[next_pos:next_pos+step] == next_point), dim=1).nonzero()) > 0:
                    global_action_dist[self.move_to_action[move]] = 1.0
        if torch.all(global_action_dist==0):
            global_action_dist[-1] = 1.0
            if torch.all(del_pos==0):
                self.global_traj.pop(next_pos)
        return global_action_dist
    
    def init_from_dict(self, init_info):
        self.start = torch.tensor(init_info["start"])
        self.pos = self.start.clone()
        self.goal = torch.tensor(init_info["goal"])
        self.type = init_info["type"]
        self.priority = init_info["priority"]
        self.concepts = init_info["concepts"]
        # for k, v in init_info["concepts"].items():
        #     assert k in self.concepts, "Concept {} not in the concepts of car!".format(k)
        #     assert self.concepts[k] == v, "Concept {} not match the concepts of car!".format(k)
