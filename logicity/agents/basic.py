import torch
import logging
from torch.distributions import Categorical
from ..core.config import *

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
        # Global Navigation Assistant context
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

    def receive_global_context(self, broadcast):
        """
        Receive global context broadcast from GNA and prepare global entities for Z3 integration.

        Args:
            broadcast: Global context broadcast from GNA
        """
        self.global_context = broadcast
        self.global_entities = {}  # Global entities prepared for Z3 integration

        if broadcast is not None:
            global_context = broadcast.get('global_context', {})
            agent_count = len(global_context)
            broadcast_id = broadcast.get('broadcast_id', 'unknown')
            logger.debug(f"Agent {self.type}_{self.id}: Received GNA broadcast {broadcast_id} with {agent_count} global entities")

            # Prepare global entities for Z3 integration
            for agent_id, agent_data in global_context.items():
                if agent_id != f"{self.type}_{self.id}":  # Don't include self
                    self.global_entities[agent_id] = self._prepare_global_entity_for_z3(agent_data)

            # Log what this agent sees in the global context
            if f"{self.type}_{self.id}" in global_context:
                my_data = global_context[f"{self.type}_{self.id}"]
                my_pos = my_data['agent_properties']['position']
                my_goal = my_data['agent_properties']['goal']
                logger.debug(f"Agent {self.type}_{self.id}: My global data - pos={my_pos}, goal={my_goal}")

            logger.debug(f"Agent {self.type}_{self.id}: Prepared {len(self.global_entities)} global entities for Z3 reasoning")
        else:
            logger.debug(f"Agent {self.type}_{self.id}: Received null GNA broadcast (GNA disabled)")
            self.global_entities = {}

    def _prepare_global_entity_for_z3(self, agent_data):
        """
        Prepare a global entity for integration into Z3 local reasoning.

        Args:
            agent_data: Agent data from GNA broadcast

        Returns:
            Dict with Z3-compatible entity information
        """
        agent_properties = agent_data['agent_properties']

        # Create a pseudo-agent object similar to what Z3 planner expects
        class GlobalPseudoAgent:
            def __init__(self, agent_type, layer_id, concepts, position, direction, goal, 
                        is_at_intersection=False, is_in_intersection=False):
                self.type = agent_type
                self.layer_id = layer_id
                self.concepts = concepts if concepts else {}
                self.pos = position
                self.last_move_dir = direction
                self.goal = goal
                self.priority = concepts.get('priority', 1) if concepts else 1
                self.global_pos = position  # Store absolute position
                self.in_fov_matrix = False  # Mark as not in FOV matrix
                self.is_at_intersection = is_at_intersection
                self.is_in_intersection = is_in_intersection

        return GlobalPseudoAgent(
            agent_type=agent_properties['type'],
            layer_id=int(agent_properties.get('layer_id', 0)),
            concepts=agent_properties.get('concepts', {}),
            position=agent_properties['position'],
            direction=agent_properties.get('current_action'),
            goal=agent_properties['goal'],
            is_at_intersection=agent_properties.get('is_at_intersection', False),
            is_in_intersection=agent_properties.get('is_in_intersection', False)
        )

    def enhance_reasoning_with_global_context(self, local_action_dist):
        """
        Enhance local reasoning with global context from GNA.

        Args:
            local_action_dist: Local action distribution from Z3 reasoning

        Returns:
            Enhanced action distribution considering global context
        """
        if self.global_context is None:
            # No global context available, use local reasoning only
            logger.debug(f"Agent {self.type}_{self.id}: Using local reasoning only (no GNA context)")
            return local_action_dist

        # Extract relevant global information for this agent
        global_info = self.analyze_global_context_for_agent()

        # Use global context to modify local action distribution
        enhanced_dist = self.apply_global_context_to_reasoning(local_action_dist, global_info)

        logger.debug(f"Agent {self.type}_{self.id}: Enhanced reasoning with {len(global_info)} global agents")
        if global_info:
            # Log some insights about global context usage
            close_agents = sum(1 for info in global_info.values() if 'close_proximity' in info.get('potential_conflicts', []))
            competing_goals = sum(1 for info in global_info.values() if 'competing_goals' in info.get('potential_conflicts', []))
            logger.debug(f"Agent {self.type}_{self.id}: Global insights - close_agents={close_agents}, competing_goals={competing_goals}")
        return enhanced_dist

    def analyze_global_context_for_agent(self):
        """
        Analyze the global context to extract relevant information for this agent.

        Returns:
            Dict with relevant global information
        """
        if self.global_context is None:
            return {}

        global_context = self.global_context.get('global_context', {})
        agent_info = {}

        # Analyze other agents' situations
        for agent_id, context in global_context.items():
            if agent_id == f"{self.type}_{self.id}":
                continue  # Skip self

            # Extract relevant information about other agents
            other_agent_info = {
                'position': context['agent_properties']['position'],
                'goal': context['agent_properties']['goal'],
                'type': context['agent_properties']['type'],
                'traffic_density': context['environmental_context']['traffic_conditions']['density'],
                'potential_conflicts': self.identify_potential_conflicts(context)
            }
            agent_info[agent_id] = other_agent_info

        return agent_info

    def identify_potential_conflicts(self, other_agent_context):
        """
        Identify potential conflicts with another agent based on global context.

        Args:
            other_agent_context: Context information about another agent

        Returns:
            List of potential conflicts
        """
        conflicts = []

        # Check if other agent is heading toward same goal
        other_goal = other_agent_context['agent_properties']['goal']
        if torch.allclose(torch.tensor(self.goal), torch.tensor(other_goal)):
            conflicts.append('competing_goals')

        # Check if paths might intersect
        # This is a simplified check - in practice would need more sophisticated path analysis
        other_pos = other_agent_context['agent_properties']['position']
        distance = torch.dist(torch.tensor(self.pos, dtype=torch.float),
                            torch.tensor(other_pos, dtype=torch.float))
        if distance < 10:  # Within 10 units
            conflicts.append('close_proximity')

        return conflicts

    def apply_global_context_to_reasoning(self, local_action_dist, global_info):
        """
        Apply global context insights to modify local action distribution.

        Args:
            local_action_dist: Original local action distribution
            global_info: Processed global context information

        Returns:
            Modified action distribution considering global context
        """
        enhanced_dist = local_action_dist.clone()

        # Example enhancement: If many agents are in close proximity,
        # increase preference for caution (stop/slow actions)
        close_agents = sum(1 for info in global_info.values()
                          if 'close_proximity' in info.get('potential_conflicts', []))

        if close_agents > 2:  # Many agents nearby
            # Boost cautious actions (assuming higher indices are more cautious)
            # This is a simplified example - actual logic would depend on action mapping
            pass

        # If competing for same goal, adjust priorities
        competing_agents = sum(1 for info in global_info.values()
                              if 'competing_goals' in info.get('potential_conflicts', []))

        if competing_agents > 0:
            # Adjust behavior based on priority system
            # Higher priority agents might be more assertive
            pass

        return enhanced_dist

    def get_action(self, local_action_dist):
        # Global context is now integrated directly into Z3 reasoning via break_world_matrix
        # No separate enhancement step needed - single integrated reasoning step

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
        # now only one action is possible
        assert len(final_action_dist.nonzero()) >= 1
        # sample from the enhanced planner
        normalized_action_dist = final_action_dist / final_action_dist.sum()
        dist = Categorical(normalized_action_dist)
        # Sample an action index from the distribution
        action_index = dist.sample()
        # Get the actual action from the action space using the sampled index
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