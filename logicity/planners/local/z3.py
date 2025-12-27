import re
import gc
import time
import copy
import torch
import logging
import importlib
import numpy as np
from z3 import *
from ...core.config import *
from multiprocessing import Pool
from ...utils.find import find_agent
from ...utils.sample import split_into_subsets
from .basic import LocalPlanner

logger = logging.getLogger(__name__)

# used for grounding
class PesudoAgent:
    def __init__(self, type, layer_id, concepts, moving_direction, global_pos=None, in_fov_matrix=True):
        self.type = type
        self.layer_id = layer_id
        self.type = concepts["type"]
        self.priority = concepts["priority"]
        self.concepts = concepts
        self.moving_direction = moving_direction
        self.global_pos = global_pos  # Absolute position in the world (for global entities)
        self.in_fov_matrix = in_fov_matrix  # Whether entity is in the local FOV world matrix

class Z3Planner(LocalPlanner):
    def __init__(self, logic_engine_file):
        # Store the rule file path for determining rule type
        self.rule_file_path = logic_engine_file['rule']

        super().__init__(logic_engine_file)

        # Hard-coded sub-rules for each rule type
        self.sub_rules = self._get_hardcoded_subrules()

        # Tally counters for simulation-wide statistics
        # NOTE: These accumulate once per agent per timestep.
        # With N agents, these grow by N * (number of sub-rules) per timestep.
        self.total_fully_observed = 0
        self.total_partially_observed = 0
        self.total_fully_unobserved = 0
        self.total_agent_analyses = 0  # Track number of agent analyses for normalization

        # Global oracle tally counters
        # NOTE: These accumulate once per timestep (single oracle analysis).
        # These grow by (number of sub-rules) per timestep, regardless of agent count.
        self.global_total_fully_observed = 0
        self.global_total_partially_observed = 0
        self.global_total_fully_unobserved = 0
        self.global_oracle_analyses = 0  # Track number of oracle analyses

    def _get_hardcoded_subrules(self):
        """Return hard-coded sub-rules dictionary based on rule file type."""

        # Check if this is expert or easy rules
        is_expert = "expert" in self.rule_file_path.lower()

        if is_expert:
            return self._get_expert_subrules()
        else:
            return self._get_easy_subrules()

    def _get_expert_subrules(self):
        """Hard-coded sub-rules for expert rules."""

        return {
            'Stop': [
                {
                    'id': 1,
                    'required_entity_types': ['Ambulance', 'Old'],  # From Not(IsAmbulance), Not(IsOld)
                    'spatial_predicates': ['IsAtInter', 'IsInInter']
                },
                {
                    'id': 2,
                    'required_entity_types': ['Ambulance', 'Old'],  # From Not(IsAmbulance), Not(IsOld)
                    'spatial_predicates': ['IsAtInter', 'IsAtInter', 'HigherPri']
                },
                {
                    'id': 3,
                    'required_entity_types': ['Ambulance', 'Old'],  # From Not(IsAmbulance), Not(IsOld), IsAmbulance
                    'spatial_predicates': ['IsInInter', 'IsInInter']
                },
                {
                    'id': 4,
                    'required_entity_types': ['Ambulance', 'Police', 'Car'],  # From Not(IsAmbulance), Not(IsPolice), IsCar, IsPolice
                    'spatial_predicates': ['LeftOf', 'IsClose']
                },
                {
                    'id': 5,
                    'required_entity_types': ['Bus', 'Pedestrian'],  # From IsBus, IsPedestrian
                    'spatial_predicates': ['RightOf', 'NextTo']
                },
                {
                    'id': 6,
                    'required_entity_types': ['Ambulance', 'Old'],  # From IsAmbulance, IsOld
                    'spatial_predicates': ['RightOf']
                },
                {
                    'id': 7,
                    'required_entity_types': ['Ambulance', 'Old'],  # From Not(IsAmbulance), Not(IsOld)
                    'spatial_predicates': ['CollidingClose']
                }
            ],
            'Slow': [
                {
                    'id': 1,
                    'required_entity_types': ['Tiro', 'Pedestrian'],  # From IsTiro, IsPedestrian
                    'spatial_predicates': ['IsClose']
                },
                {
                    'id': 2,
                    'required_entity_types': ['Tiro'],  # From IsTiro
                    'spatial_predicates': ['IsInInter', 'IsAtInter']
                },
                {
                    'id': 3,
                    'required_entity_types': ['Police', 'Young'],  # From IsPolice, IsYoung, IsYoung
                    'spatial_predicates': ['NextTo']
                }
            ],
            'Fast': [
                {
                    'id': 1,
                    'required_entity_types': ['Reckless'],  # From IsReckless
                    'spatial_predicates': ['IsAtInter']
                },
                {
                    'id': 2,
                    'required_entity_types': ['Bus'],  # From IsBus
                    'spatial_predicates': []
                },
                {
                    'id': 3,
                    'required_entity_types': ['Police', 'Reckless'],  # From IsPolice, IsReckless
                    'spatial_predicates': []
                }
            ]
        }

    def _get_easy_subrules(self):
        """Hard-coded sub-rules for easy rules (spatial-only)."""

        return {
            'Stop': [
                {
                    'id': 1,
                    'required_entity_types': [],  # No attribute predicates
                    'spatial_predicates': ['IsAtInter', 'IsInInter']
                },
                {
                    'id': 2,
                    'required_entity_types': [],  # No attribute predicates
                    'spatial_predicates': ['IsAtInter', 'IsAtInter', 'HigherPri']
                },
                {
                    'id': 3,
                    'required_entity_types': [],  # No attribute predicates
                    'spatial_predicates': ['CollidingClose']
                }
            ],
            'Slow': [
                {
                    'id': 1,
                    'required_entity_types': [],  # Not(Slow(entity)) - no attributes needed
                    'spatial_predicates': []
                }
            ],
            'Fast': [
                {
                    'id': 1,
                    'required_entity_types': [],  # Not(Fast(entity)) - no attributes needed
                    'spatial_predicates': []
                }
            ]
        }

    def get_agent_attribute_observations(self, ego_agent, partial_agents, gna_broadcast=None):
        """
        Get observed attribute predicates for an agent.

        Args:
            ego_agent: The ego agent
            partial_agents: Dictionary of agents in FOV
            gna_broadcast: GNA broadcast data (optional)

        Returns:
            Dictionary of observed attribute predicates
        """
        observed_attributes = {}

        # 1. Observe ego agent's own attributes
        if hasattr(ego_agent, 'concepts') and ego_agent.concepts:
            for concept_key, concept_value in ego_agent.concepts.items():
                if concept_value in [1, 1.0]:  # Active concept (accept both int and float)
                    attr_name = f"Is{concept_key.capitalize()}(entity)"
                    observed_attributes[attr_name] = True

        # 2. Observe attributes of agents in local FOV
        for agent_key, agent in partial_agents.items():
            if agent_key == 'ego':  # Skip ego agent (already handled)
                continue

            if hasattr(agent, 'concepts') and agent.concepts:
                # Map layer ID to dummy entity (dummyEntityA, dummyEntityB, etc.)
                dummy_entity = self._map_layer_to_dummy_entity(agent.layer_id, partial_agents)

                for concept_key, concept_value in agent.concepts.items():
                    if concept_value in [1, 1.0]:  # Active concept (accept both int and float)
                        attr_name = f"Is{concept_key.capitalize()}({dummy_entity})"
                        observed_attributes[attr_name] = True

        # 3. Observe attributes from GNA broadcast (global context)
        if gna_broadcast and 'global_context' in gna_broadcast:
            global_context = gna_broadcast['global_context']
            for global_agent_id, global_data in global_context.items():
                agent_props = global_data.get('agent_properties', {})
                concepts = agent_props.get('concepts', {})

                if concepts:
                    # Map global agent to appropriate dummy entity
                    dummy_entity = f"dummyEntityA"  # Default to A, could be extended
                    for concept_key, concept_value in concepts.items():
                        if concept_value in [1, 1.0]:  # Active concept (accept both int and float)
                            attr_name = f"Is{concept_key.capitalize()}({dummy_entity})"
                            observed_attributes[attr_name] = True

        return observed_attributes

    def _map_layer_to_dummy_entity(self, layer_id, partial_agents):
        """Map agent layer ID to dummy entity variable name."""
        # Simple mapping: first dummy is A, second is B, etc.
        agent_keys = [k for k in partial_agents.keys() if k != 'ego']
        try:
            index = agent_keys.index(str(layer_id))
            if index == 0:
                return 'dummyEntityA'
            elif index == 1:
                return 'dummyEntityB'
            else:
                return f"dummyEntity{chr(65 + index)}"  # A, B, C, ...
        except ValueError:
            return 'dummyEntityA'  # Default fallback

    def assess_subrule_grounding(self, sub_rule, ego_agent, partial_agents, gna_broadcast=None):
        """
        Assess the grounding status of a sub-rule based on entity type presence.

        Args:
            sub_rule: Sub-rule dictionary with required_entity_types
            ego_agent: Ego agent
            partial_agents: Partial agents in FOV
            gna_broadcast: GNA broadcast data

        Returns:
            Dictionary with grounding status and details
        """
        required_entity_types = sub_rule['required_entity_types']

        # If no entity types are required (spatial-only rules), always fully observed
        if not required_entity_types:
            return {
                'grounding_status': 'fully_observed',
                'satisfiability': 'sat',  # Spatial rules are always satisfiable
                'required_entity_types': required_entity_types,
                'observed_entity_types': [],
                'missing_entity_types': []
            }

        # Check which required entity types are present in FOV or GNA
        present_entity_types = self._get_present_entity_types(ego_agent, partial_agents, gna_broadcast)

        observed = [et for et in required_entity_types if et in present_entity_types]
        missing = [et for et in required_entity_types if et not in present_entity_types]

        if len(observed) == len(required_entity_types):
            grounding_status = 'fully_observed'
            satisfiability = 'sat'  # All required entity types present
        elif len(observed) > 0:
            grounding_status = 'partially_observed'
            satisfiability = 'partial'  # Some entity types present
        else:
            grounding_status = 'fully_unobserved'
            satisfiability = 'unknown'  # No entity types present

        return {
            'grounding_status': grounding_status,
            'satisfiability': satisfiability,
            'required_entity_types': required_entity_types,
            'observed_entity_types': observed,
            'missing_entity_types': missing
        }

    def _get_present_entity_types(self, ego_agent, partial_agents, gna_broadcast=None):
        """
        Get set of entity types present in FOV and GNA broadcast.

        Args:
            ego_agent: Ego agent
            partial_agents: Dictionary of agents in FOV
            gna_broadcast: GNA broadcast data

        Returns:
            Set of present entity type names
        """
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

    def calculate_local_gna_informativeness(self, ego_agent, partial_agents, gna_broadcast):
        """
        Calculate informativeness score for entities visible to an agent.
        Sum of occurrence scores for all individual entities in FOV + GNA.
        
        Args:
            ego_agent: The ego agent
            partial_agents: Dictionary of agents in FOV
            gna_broadcast: GNA broadcast data
        
        Returns:
            Tuple of (local_gna_score, entity_count)
        """
        local_gna_score = 0
        entities_counted = set()  # Track entity IDs to avoid double-counting
        
        # 1. Count ego agent
        if hasattr(ego_agent, 'concepts') and ego_agent.concepts:
            ego_type = self._get_entity_type_from_concepts(ego_agent.concepts)
            if ego_type:
                ego_id = f"ego_{ego_agent.id}"
                local_gna_score += ENTITY_OCCURRENCE_SCORES.get(ego_type, 0)
                entities_counted.add(ego_id)
        
        # 2. Count FOV entities (partial_agents)
        for agent_name, agent in partial_agents.items():
            if hasattr(agent, 'concepts') and agent.concepts:
                entity_type = self._get_entity_type_from_concepts(agent.concepts)
                if entity_type:
                    # Use layer_id to uniquely identify entity
                    entity_id = f"fov_{agent.layer_id}"
                    if entity_id not in entities_counted:
                        local_gna_score += ENTITY_OCCURRENCE_SCORES.get(entity_type, 0)
                        entities_counted.add(entity_id)
        
        # 3. Count GNA broadcast entities (avoid double-counting with FOV)
        if gna_broadcast and 'global_context' in gna_broadcast:
            for agent_id, agent_data in gna_broadcast['global_context'].items():
                # Check if this entity is already counted in FOV
                # Extract layer_id from agent_id (format: "Type_layerid")
                gna_entity_id = f"gna_{agent_id}"
                if gna_entity_id not in entities_counted:
                    concepts = agent_data['agent_properties'].get('concepts', {})
                    entity_type = self._get_entity_type_from_concepts(concepts)
                    if entity_type:
                        local_gna_score += ENTITY_OCCURRENCE_SCORES.get(entity_type, 0)
                        entities_counted.add(gna_entity_id)
        
        return local_gna_score, len(entities_counted)

    def _analyze_global_subrule_satisfiability(self, world_matrix, agents, intersect_matrix):
        """
        Analyze sub-rule satisfiability for the global (oracle) world state.

        This checks which sub-rules would be fully observable if an agent had
        perfect knowledge of all entity types present in the entire world.
        """
        logger.info("Performing global sub-rule satisfiability analysis...")

        # Collect all entity types actually present in the scene
        all_entity_types = set()
        entity_type_counts = {}
        
        # Calculate global oracle informativeness (sum of all entity scores)
        global_oracle_informativeness = 0
        
        for agent_obj in agents:
            if hasattr(agent_obj, 'concepts') and agent_obj.concepts:
                entity_type = self._get_entity_type_from_concepts(agent_obj.concepts)
                if entity_type:
                    all_entity_types.add(entity_type)
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                    # Sum occurrence score for each individual entity
                    score = ENTITY_OCCURRENCE_SCORES.get(entity_type, 0)
                    global_oracle_informativeness += score
        
        # Store for later comparison with local+GNA scores
        self.current_global_oracle_informativeness = global_oracle_informativeness

        logger.info(f"Z3: Global Oracle - Entity types in scene: {sorted(all_entity_types)}")
        logger.info(f"Z3: Global Oracle - Entity counts: {entity_type_counts}")
        logger.info(f"Z3: Global Oracle - Total Informativeness Score: {global_oracle_informativeness}")

        # Evaluate each sub-rule with perfect knowledge
        global_fully_observed = 0
        global_partially_observed = 0
        global_fully_unobserved = 0
        global_total = 0

        subrule_details = {}  # For detailed logging

        for rule_name, sub_rules in self.sub_rules.items():
            subrule_details[rule_name] = {}
            
            for sub_rule in sub_rules:
                global_total += 1
                sub_rule_id = f"subrule_{sub_rule['id']}"
                required_types = set(sub_rule['required_entity_types'])

                # Spatial-only rules (no entity type requirements)
                if not required_types:
                    global_fully_observed += 1
                    grounding_status = 'fully_observed'
                    observed = []
                    missing = []
                # Check if all required types are present in the scene
                elif required_types.issubset(all_entity_types):
                    global_fully_observed += 1
                    grounding_status = 'fully_observed'
                    observed = list(required_types)
                    missing = []
                # Check if some required types are present
                elif required_types.intersection(all_entity_types):
                    global_partially_observed += 1
                    grounding_status = 'partially_observed'
                    observed = list(required_types.intersection(all_entity_types))
                    missing = list(required_types - all_entity_types)
                # No required types are present
                else:
                    global_fully_unobserved += 1
                    grounding_status = 'fully_unobserved'
                    observed = []
                    missing = list(required_types)

                subrule_details[rule_name][sub_rule_id] = {
                    'grounding_status': grounding_status,
                    'required': list(required_types),
                    'observed': observed,
                    'missing': missing
                }

        # Log detailed results
        logger.info(f"Z3: Global Oracle - Sub-rule breakdown:")
        for rule_name, rule_details in subrule_details.items():
            fully = sum(1 for d in rule_details.values() if d['grounding_status'] == 'fully_observed')
            partially = sum(1 for d in rule_details.values() if d['grounding_status'] == 'partially_observed')
            unobserved = sum(1 for d in rule_details.values() if d['grounding_status'] == 'fully_unobserved')
            logger.info(f"Z3:   {rule_name}: Fully={fully}, Partially={partially}, Unobserved={unobserved}")
            
            # Log missing types for partially observed/unobserved rules
            for sub_rule_id, details in rule_details.items():
                if details['missing']:
                    logger.debug(f"Z3:     {sub_rule_id}: Missing types {details['missing']}")

        # Log global summary
        logger.info(f"Z3: Global Oracle Summary - Fully observed: {global_fully_observed}, Partially: {global_partially_observed}, Unobserved: {global_fully_unobserved} (Total: {global_total})")

        # Update global tally counters
        self.global_total_fully_observed += global_fully_observed
        self.global_total_partially_observed += global_partially_observed
        self.global_total_fully_unobserved += global_fully_unobserved
        self.global_oracle_analyses += 1

        # Log global running totals
        global_total_overall = self.global_total_fully_observed + self.global_total_partially_observed + self.global_total_fully_unobserved
        logger.info(f"Z3: Global Oracle Running Totals - Fully observed: {self.global_total_fully_observed}, Partially: {self.global_total_partially_observed}, Unobserved: {self.global_total_fully_unobserved} (Total: {global_total_overall})")
        
        # Log oracle averages
        if self.global_oracle_analyses > 0:
            oracle_avg_fully = self.global_total_fully_observed / self.global_oracle_analyses
            oracle_avg_partially = self.global_total_partially_observed / self.global_oracle_analyses
            oracle_avg_unobserved = self.global_total_fully_unobserved / self.global_oracle_analyses
            logger.info(f"Z3: Global Oracle Averages - Fully: {oracle_avg_fully:.2f}, Partially: {oracle_avg_partially:.2f}, Unobserved: {oracle_avg_unobserved:.2f} (Oracle analyses: {self.global_oracle_analyses})")

    def _get_entity_type_from_concepts(self, concepts):
        """
        Extract entity type name from agent concepts.

        Args:
            concepts: Agent concepts dictionary

        Returns:
            Entity type string or None
        """
        # Priority order: most specific to least specific
        # Must match GNA naming and priority hierarchy
        # Ordered list ensures deterministic selection when multiple concepts are present
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

        # Check for specific concept types in priority order (deterministic)
        # Accept both int (1) and float (1.0) for robustness
        for concept_key, entity_type in concept_priority_order:
            if concept_key in concepts and concepts[concept_key] in [1, 1.0]:
                return entity_type

        # Default to base type if no specific attributes
        base_type = concepts.get('type', '').title()
        if base_type in ['Car', 'Pedestrian']:
            return base_type

        return None


    def analyze_subrule_satisfiability(self, ego_agent, partial_agents, partial_world, partial_intersections, gna_broadcast=None, ego_name=None):
        """
        Analyze satisfiability of all sub-rules for an agent.

        Args:
            ego_agent: The ego agent
            partial_agents: Agents in FOV
            partial_world: World state matrices
            partial_intersections: Intersection matrices
            gna_broadcast: GNA broadcast data

        Returns:
            Dictionary with sub-rule analysis for each rule
        """
        # Get observed attributes for this agent
        observed_attributes = self.get_agent_attribute_observations(ego_agent, partial_agents, gna_broadcast)

        analysis = {}

        # Analyze each rule's sub-rules
        for rule_name, sub_rules in self.sub_rules.items():
            rule_analysis = {}

            for sub_rule in sub_rules:
                sub_rule_id = f"subrule_{sub_rule['id']}"
                grounding_analysis = self.assess_subrule_grounding(sub_rule, ego_agent, partial_agents, gna_broadcast)

                rule_analysis[sub_rule_id] = {
                    'required_entity_types': sub_rule['required_entity_types'],
                    'spatial_predicates': sub_rule['spatial_predicates'],
                    **grounding_analysis
                }

            analysis[rule_name] = rule_analysis

        # Log detailed sub-rule analysis results (but skip for global oracle to avoid double-counting)
        if ego_name is not None and ego_name != "global_oracle":
            self._log_subrule_analysis(analysis, ego_name)

        return analysis

    def _log_subrule_analysis(self, analysis, ego_name):
        """Log detailed sub-rule analysis results for debugging and monitoring."""
        logger.info(f"Z3: Sub-rule analysis for agent {ego_name}")

        # Count statistics across all rules
        total_subrules = 0
        fully_observed = 0
        partially_observed = 0
        fully_unobserved = 0

        for rule_name, rule_analysis in analysis.items():
            logger.info(f"Z3:   Rule {rule_name}:")

            for subrule_id, subrule_data in rule_analysis.items():
                total_subrules += 1
                grounding_status = subrule_data['grounding_status']

                if grounding_status == 'fully_observed':
                    fully_observed += 1
                    satisfiability = subrule_data['satisfiability']
                elif grounding_status == 'partially_observed':
                    partially_observed += 1
                    satisfiability = 'partial'
                else:  # fully_unobserved
                    fully_unobserved += 1
                    satisfiability = 'unknown'

                required_types = subrule_data['required_entity_types']
                observed_types = subrule_data['observed_entity_types']
                missing_types = subrule_data['missing_entity_types']

                logger.info(f"Z3:     {subrule_id}: {grounding_status} ({len(observed_types)}/{len(required_types)} types) - {satisfiability}")
                if missing_types:
                    logger.info(f"Z3:       Missing types: {missing_types}")

        # Log summary statistics
        logger.info(f"Z3:   Summary - Fully observed: {fully_observed}, Partially: {partially_observed}, Unobserved: {fully_unobserved} (Total: {total_subrules})")

        # Update running tally counters
        self.total_fully_observed += fully_observed
        self.total_partially_observed += partially_observed
        self.total_fully_unobserved += fully_unobserved
        self.total_agent_analyses += 1

        # Log running totals
        total_overall = self.total_fully_observed + self.total_partially_observed + self.total_fully_unobserved
        logger.info(f"Z3:   Running Totals - Fully observed: {self.total_fully_observed}, Partially: {self.total_partially_observed}, Unobserved: {self.total_fully_unobserved} (Total: {total_overall})")
        
        # Log normalized averages per agent
        if self.total_agent_analyses > 0:
            avg_fully = self.total_fully_observed / self.total_agent_analyses
            avg_partially = self.total_partially_observed / self.total_agent_analyses
            avg_unobserved = self.total_fully_unobserved / self.total_agent_analyses
            logger.info(f"Z3:   Per-Agent Averages - Fully: {avg_fully:.2f}, Partially: {avg_partially:.2f}, Unobserved: {avg_unobserved:.2f} (Agents analyzed: {self.total_agent_analyses})")

    def _create_entities(self):
        # Create Z3 sorts for each entity type
        self.entity_types = []
        for entity_type in self.data["EntityTypes"]:
            # Create a Z3 sort (type) for each entity
            self.entity_types.append(entity_type)
        # Print the entity types
        entity_types_info = "\n".join(["- {}".format(entity) for entity in self.entity_types])
        logger.info("Number of Entity Types: {}\nEntity Types:\n{}".format(len(self.entity_types), entity_types_info))

    def _create_predicates(self):
        self.predicates = {}
        for pred_dict in self.data["Predicates"]:
            (pred_name, info), = pred_dict.items()
            method_name = info["method"].split('(')[0]
            arity = info["arity"]
            z3_func = None
            if arity == 1:
                # Unary predicate
                entity_type = info["method"].split('(')[1].split(')')[0]
                z3_func = "Function('{}', entity_sorts['{}'], BoolSort())".format(method_name, entity_type)
            elif arity == 2:
                # Binary predicate
                types = info["method"].split('(')[1].split(')')[0].split(', ')
                z3_func = "Function('{}', entity_sorts['{}'], entity_sorts['{}'], BoolSort())".format(method_name, types[0], types[1])

            # Store complete predicate information
            self.predicates[pred_name] = {
                "instance": z3_func,
                "arity": arity,
                "method": info["method"],
                "function": info.get("function", None),  # Optional, may be used for dynamic grounding
            }
        # Print the predicates
        predicates_info = "\n".join(["- {}: {}".format(predicate, details) for predicate, details in self.predicates.items()])
        logger.info("Number of Predicates: {}\nPredicates:\n{}".format(len(self.predicates), predicates_info))

    def _create_rules(self):
        self.rules = {}
        self.rule_tem = {}
        self.z3_vars = []
        for rule_dict in self.data["Rules"]:
            (rule_name, rule_info), = rule_dict.items()
            # Check if the rule is valid
            formula = rule_info["formula"]
            logger.info("Rule: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')
                if var_name not in self.z3_vars:
                    self.z3_vars.append(var_name)

            # Evaluate the modified formula string to create the Z3 expression
            self.rule_tem[rule_name] = formula
        rule_info = "\n".join(["- {}: {}".format(rule, details) for rule, details in self.rule_tem.items()])
        logger.info("Number of Rules: {}\nRules:\n{}".format(len(self.rule_tem), rule_info))
        logger.info("Rules will be grounded later...")

    def _extract_variables(self, formula):
        # Regular expression to find words that start with 'dummy'
        pattern = re.compile(r'\bdummy\w*\b')

        # Find all matches in the formula
        matches = pattern.findall(formula)

        # Remove duplicates by converting the list to a set, then back to a list
        unique_variables = list(set(matches))

        return unique_variables

    def plan(self, world_matrix, intersect_matrix, agents, layerid2listid, use_multiprocessing=True, gna_broadcast=None, enable_subrule_analysis=False):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        # Note that the local ones will have different size and agent id
        e = time.time()
        local_world_matrix = world_matrix.clone()
        local_intersections = intersect_matrix.clone()
        ego_agent, partial_agents, partial_world, partial_intersections = \
            self.break_world_matrix(local_world_matrix, agents, local_intersections, layerid2listid)
        logger.info("Break world time: {}".format(time.time()-e))

        # 3. Perform sub-rule analysis if enabled
        subrule_analysis = {}
        if enable_subrule_analysis:
            logger.info("Performing sub-rule satisfiability analysis...")
            
            # Perform global world state analysis first to get oracle informativeness
            self._analyze_global_subrule_satisfiability(world_matrix, agents, intersect_matrix)
            
            # Analyze each agent
            informativeness_scores = []
            for ego_name in partial_agents.keys():
                ego_agent_obj = ego_agent[ego_name]
                analysis = self.analyze_subrule_satisfiability(
                    ego_agent_obj, partial_agents[ego_name],
                    partial_world[ego_name], partial_intersections[ego_name],
                    gna_broadcast, ego_name
                )
                subrule_analysis[ego_name] = analysis
                
                # Calculate informativeness for this agent
                local_gna_score, entity_count = self.calculate_local_gna_informativeness(
                    ego_agent_obj, partial_agents[ego_name], gna_broadcast
                )
                
                # Normalize by global oracle score
                if hasattr(self, 'current_global_oracle_informativeness') and \
                   self.current_global_oracle_informativeness > 0:
                    normalized_informativeness = local_gna_score / self.current_global_oracle_informativeness
                else:
                    normalized_informativeness = 0.0
                
                informativeness_scores.append({
                    'agent_name': ego_name,
                    'local_gna_score': local_gna_score,
                    'entity_count': entity_count,
                    'normalized': normalized_informativeness
                })
            
            logger.info(f"Completed sub-rule analysis for {len(subrule_analysis)} agents")
            
            # Log average informativeness across all agents
            if informativeness_scores:
                avg_local_gna = sum(s['local_gna_score'] for s in informativeness_scores) / len(informativeness_scores)
                avg_entity_count = sum(s['entity_count'] for s in informativeness_scores) / len(informativeness_scores)
                avg_normalized = sum(s['normalized'] for s in informativeness_scores) / len(informativeness_scores)
                
                logger.info(f"Z3: Average Local+GNA Informativeness: {avg_local_gna:.2f} (avg {avg_entity_count:.1f} entities)")
                logger.info(f"Z3: Global Oracle Informativeness: {self.current_global_oracle_informativeness}")
                logger.info(f"Z3: Average Normalized Informativeness: {avg_normalized:.3f} ({avg_normalized*100:.1f}%)")
                
                # Store in subrule_analysis
                subrule_analysis['informativeness_summary'] = {
                    'avg_local_gna_score': avg_local_gna,
                    'avg_entity_count': avg_entity_count,
                    'avg_normalized': avg_normalized,
                    'global_oracle_score': self.current_global_oracle_informativeness
                }

        # 2. Choose between multi-processing and looping
        combined_results = {}
        agent_keys = list(partial_agents.keys())
        
        if use_multiprocessing:
            # Multi-processing approach
            agent_batches = split_into_batches(agent_keys, NUM_PROCESS)
            with Pool(processes=NUM_PROCESS) as pool:
                for batch_keys in agent_batches:
                    batch_results = pool.starmap(solve_sub_problem, 
                                                [(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                                self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                                partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name])
                                                for ego_name in batch_keys])
                    
                    for result in batch_results:
                        combined_results.update(result)
                    gc.collect()
        else:
            # Looping approach
            for ego_name in agent_keys:
                result = solve_sub_problem(ego_name, ego_agent[ego_name].action_mapping, ego_agent[ego_name].action_dist,
                                        self.rule_tem, self.entity_types, self.predicates, self.z3_vars,
                                        partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name])
                combined_results.update(result)

        e2 = time.time()
        logger.info("Solve sub-problem time: {}".format(e2-e))
        return combined_results, subrule_analysis
    
    def get_fov(self, position, direction, width, height):
        # Calculate the region of the city image that falls within the ego agent's field of view
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
        return x_start, y_start, x_end, y_end

    def break_world_matrix(self, world_matrix, agents, intersect_matrix, layerid2listid):
        ego_agent = {}
        partial_agents = {}
        partial_world = {}
        partial_intersection = {}
        for agent in agents:
            ego_name = "{}_{}".format(agent.type, agent.layer_id)
            ego_agent[ego_name] = agent
            ego_layer = world_matrix[agent.layer_id]
            ego_position = (ego_layer == TYPE_MAP[agent.type]).nonzero()[0]
            ego_direction = agent.last_move_dir
            x_start, y_start, x_end, y_end = self.get_fov(ego_position, ego_direction, world_matrix.shape[1], world_matrix.shape[2])
            partial_world_all = world_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_intersections = intersect_matrix[:, x_start:x_end, y_start:y_end].clone()
            partial_world_nonzero_int = torch.logical_and(partial_world_all != 0, \
                                                          partial_world_all == partial_world_all.to(torch.int64))
            # Apply torch.any across dimensions 1 and 2 sequentially
            non_zero_layers = partial_world_nonzero_int.any(dim=1).any(dim=1)
            non_zero_layer_indices = torch.where(non_zero_layers)[0]
            partial_world_squeezed = partial_world_all[non_zero_layers]
            partial_world[ego_name] = partial_world_squeezed
            partial_intersection[ego_name] = partial_intersections
            partial_agent = {}

            # First, add local agents (existing logic)
            for layer_id in range(partial_world_squeezed.shape[0]):
                layer = partial_world_squeezed[layer_id]
                layer_nonzero_int = torch.logical_and(layer != 0, layer == layer.to(torch.int64))
                if layer_nonzero_int.nonzero().shape[0] > 1:
                    continue
                non_zero_values = int(layer[layer_nonzero_int.nonzero()[0][0], layer_nonzero_int.nonzero()[0][1]])
                agent_type = LABEL_MAP[non_zero_values]
                # find this agent
                other_agent_layer_id = int(non_zero_layer_indices[layer_id])
                other_agent = agents[layerid2listid[other_agent_layer_id]]
                assert other_agent.type == agent_type
                if other_agent_layer_id == agent.layer_id:
                    partial_agent["ego_{}".format(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir)
                else:
                    partial_agent[str(layer_id)] = PesudoAgent(agent_type, layer_id, other_agent.concepts, other_agent.last_move_dir)

            # Second, add global entities from GNA broadcast
            if hasattr(agent, 'global_entities') and agent.global_entities:
                global_layer_offset = len(non_zero_layer_indices)  # Start global entities after local ones
                for global_agent_id, global_entity in agent.global_entities.items():
                    # Convert global position to local FOV coordinates
                    global_pos = global_entity.pos
                    if isinstance(global_pos, list):
                        global_pos = torch.tensor(global_pos)

                    # Check if global entity is within FOV bounds
                    if (x_start <= global_pos[0] < x_end and
                        y_start <= global_pos[1] < y_end):
                        # Entity is within FOV - add it
                        local_x = global_pos[0] - x_start
                        local_y = global_pos[1] - y_start

                        # Add to partial_world_squeezed if not already there
                        # Find or create layer for this global entity
                        entity_layer_id = global_entity.layer_id
                        if entity_layer_id not in non_zero_layer_indices:
                            # Add new layer for global entity
                            new_layer = torch.zeros_like(partial_world_squeezed[0:1])
                            new_layer[0, local_x, local_y] = TYPE_MAP[global_entity.type]
                            partial_world_squeezed = torch.cat([partial_world_squeezed, new_layer], dim=0)
                            non_zero_layer_indices = torch.cat([non_zero_layer_indices, torch.tensor([entity_layer_id])])

                            # Update partial_world
                            partial_world[ego_name] = partial_world_squeezed

                        # Add global entity to partial_agent
                        global_pseudo_id = f"global_{global_agent_id.split('_')[1]}"  # Use layer ID from global entity
                        partial_agent[global_pseudo_id] = PesudoAgent(
                            global_entity.type,
                            global_entity.layer_id,
                            global_entity.concepts,
                            global_entity.last_move_dir
                        )

                        logger.debug(f"Z3: Added global entity {global_agent_id} to {ego_name}'s local reasoning at position ({local_x}, {local_y})")

            partial_agents[ego_name] = partial_agent
        return ego_agent, partial_agents, partial_world, partial_intersection

    def format_rule_string(self, rule_str):
        indent_level = 0
        formatted_str = ""
        bracket_stack = []  # Stack to keep track of brackets

        for char in rule_str:
            if char == ',':
                formatted_str += ',\n' + ' ' * 4 * indent_level
            elif char == '(':
                bracket_stack.append('(')
                formatted_str += '(\n' + ' ' * 4 * (indent_level + 1)
                indent_level += 1
            elif char == ')':
                if not bracket_stack or bracket_stack[-1] != '(':
                    raise ValueError("Unmatched closing bracket detected.")
                bracket_stack.pop()
                indent_level -= 1
                formatted_str += '\n' + ' ' * 4 * indent_level + ')'
            else:
                formatted_str += char

        if bracket_stack:
            raise ValueError("Unmatched opening bracket detected.")

        return formatted_str

def solve_sub_problem(ego_name, 
                      ego_action_mapping,
                      ego_action_dist,
                      rule_tem, 
                      entity_types, 
                      predicates, 
                      var_names,
                      partial_agents, 
                      partial_world, 
                      partial_intersections):
    # 1. create solver
    local_solver = Solver()
    # 2. create sorts and variables
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    z3_vars = {var_name: Const(var_name, entity_sorts['Entity']) \
                       for var_name in var_names}
    # 3. partial world to entities
    local_entities = world2entity(entity_sorts, partial_intersections, partial_agents)
    # 4. create, ground predicates and add to solver
    local_predicates = copy.deepcopy(predicates)
    for pred_name, pred_info in local_predicates.items():
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]
        if method_full_name == "None":
            continue
        module_name, method_name = method_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        method = getattr(module, method_name)

        if arity == 1:
            # Unary predicate grounding
            for entity in local_entities[eval_pred.domain(0).name()]:
                entity_name = entity.decl().name()
                value = method(partial_world, partial_intersections, partial_agents, entity_name)
                if value:
                    local_solver.add(eval_pred(entity))
                else:
                    local_solver.add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in local_entities[eval_pred.domain(0).name()]:
                entity1_name = entity1.decl().name()
                for entity2 in local_entities[eval_pred.domain(1).name()]:
                    entity2_name = entity2.decl().name()
                    value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                    if value:
                        local_solver.add(eval_pred(entity1, entity2))
                    else:
                        local_solver.add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    local_rule_tem = copy.deepcopy(rule_tem)
    for rule_name, rule_template in local_rule_tem.items():
        # the first entity is the ego agent
        entity = local_entities["Entity"][0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template)
        local_solver.add(instantiated_rule)

    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    for var_name, z3_var in z3_vars.items():
        entity_list = local_entities['Entity']
        constraint = Or([z3_var == entity for entity in entity_list])
        local_solver.add(ForAll([z3_var], constraint))
    
    # 6. solve
    if local_solver.check() == sat:
        model = local_solver.model()
        # Interpret the solution to the FOL problem
        action_mapping = ego_action_mapping
        action_dist = torch.zeros_like(ego_action_dist)

        for key in local_predicates.keys():
            action = []
            for action_id, action_name in action_mapping.items():
                if key in action_name:
                    action.append(action_id)
            if len(action)>0:
                for a in action:
                    if is_true(model.evaluate(local_predicates[key]["instance"](local_entities["Entity"][0]))):
                        action_dist[a] = 1.0
        # No action specified, use the default action, Normal
        if action_dist.sum() == 0:
            for action_id, action_name in action_mapping.items():
                if "Normal" in action_name:
                    action_dist[action_id] = 1.0

        agents_actions = {ego_name: action_dist}
        return agents_actions
    else:
        # No solution means do not exist intersection/agent in the field of view, Normal
        # Interpret the solution to the FOL problem
        action_mapping = ego_action_mapping
        action_dist = torch.zeros_like(ego_action_dist)

        for action_id, action_name in action_mapping.items():
            if "Normal" in action_name:
                action_dist[action_id] = 1.0

        agents_actions = {ego_name: action_dist}
        return agents_actions

def split_into_batches(keys, batch_size):
    """Split keys into batches of a given size."""
    for i in range(0, len(keys), batch_size):
        yield keys[i:i + batch_size]

def world2entity(entity_sorts, partial_intersect, partial_agents):
    # all the enitities are stored in self.entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entities[entity_type] = []
        # For Agents
        if entity_type == "Entity":
            # all entities are agents
            for key, agent in partial_agents.items():
                if "ego" in key:
                    ego_agent = agent
                    continue
                agent_id = agent.layer_id
                agent_type = agent.type
                agent_name = f"Entity_{agent_type}_{agent_id}"
                # Create a Z3 constant for the agent
                agent_entity = Const(agent_name, entity_sorts[entity_type])
                entities[entity_type].append(agent_entity)
            agent_id = ego_agent.layer_id
            agent_type = ego_agent.type
            agent_name = f"Entity_{agent_type}_{agent_id}"
            # Create a Z3 constant for the agent
            agent_entity = Const(agent_name, entity_sorts[entity_type])
            # ego agent is the first
            entities[entity_type] = [agent_entity] + entities[entity_type]
    return entities


