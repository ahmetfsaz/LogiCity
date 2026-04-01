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
    - Top-K Selection: Only the K most important entities are broadcast (communication bw limit)

SELECTION MODES (gna_selection_mode):
    Semantic pipeline (per-agent personalised broadcasts):
        - "semantic"            : Per-agent vicinity grounding + inductive-
                                  probability-based optimal subset selection.
        - "semantic_random"     : Per-agent vicinity grounding + random subset
                                  selection (baseline for semantic pipeline).

    LNA pipeline (two-way car ↔ LNA communication, multi-zone):
        - "semantic_lna"        : Cars uplink top-k1 FOV entities (semantic) to
                                  their LNA; LNA downlinks top-k2 (semantic).
        - "semantic_lna_random" : Same structure but random selection for both
                                  k1 and k2 (baseline for LNA pipeline).

    Single-LNA pipeline (same as LNA but one global zone):
        - "semantic_lna_single"        : Same pipeline as semantic_lna but all
                                          cars share a single zone (no partitioning).
        - "semantic_lna_random_single" : Random baseline for the single-zone pipeline.

    Global pipeline (same broadcast to every agent):
        - "ranking_priority"    : Global type-based priority ranking.
        - "ranking_random"      : Random selection (baseline for global pipeline).

INTEGRATION POINTS:
    - Called by City.update() or CityEnv.update_rl() each timestep
    - Broadcasts are received by agents via receive_global_context() in basic.py
    - Global entities are integrated into Z3 reasoning in z3.py and z3_rl.py

CONFIGURATION (in YAML config files):
    - enable_gna: true/false - Whether GNA is active
    - gna_top_k: integer - How many entities to include in broadcast (semantic/ranking modes)
    - gna_top_k1: integer - (UNUSED) Uplink transmits all FOV entities; kept for config compat
    - gna_top_k2: integer - Downlink budget LNA→car (LNA modes only)
    - gna_selection_mode: "semantic" | "semantic_random" | "semantic_lna" | "semantic_lna_random" | "semantic_lna_single" | "semantic_lna_random_single" | "ranking_priority" | "ranking_random"
"""

import time
import random
import logging
import numpy as np
import torch
from ..core.config import *
from ..planners.local.z3 import PesudoAgent
from ..utils.pred_converter.z3 import (
    IsCar, IsPed, IsAmb, IsBus, IsPolice, IsTiro, IsReckless, IsOld, IsYoung,
    IsAtInter, IsInInter, IsClose, CollidingClose, LeftOf, RightOf, NextTo, HigherPri
)
from ..utils.semantic_selection import (
    groundings_to_q_sentence, select_optimal_subset,
    ALL_HYPOTHESES, EXTENDED_ALL_HYPOTHESES, SPATIAL_ALL_HYPOTHESES,
    get_hypotheses_for_rule_file,
)


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
    
    VALID_SELECTION_MODES = (
        "semantic", "semantic_random",
        "semantic_lna", "semantic_lna_random",
        "semantic_lna_single", "semantic_lna_random_single",
        "ranking_priority", "ranking_random",
    )

    def __init__(self, top_k=5, selection_mode="semantic", rule_yaml_file="",
                 top_k1=3, top_k2=3, agent_region=None):
        """
        Initialize the Global Navigation Assistant.

        Args:
            top_k (int): Maximum number of entities to include in each broadcast
                         (used by semantic / semantic_random / ranking_* modes).

            selection_mode (str): How to select which entities to broadcast.
                Semantic pipeline (per-agent personalised broadcasts):
                    - "semantic"            : vicinity grounding + inductive-
                                              probability optimal subset.
                    - "semantic_random"     : vicinity grounding + random subset.
                LNA pipeline (two-way car ↔ LNA communication):
                    - "semantic_lna"        : car uplink (semantic k1) →
                                              LNA downlink (semantic k2).
                    - "semantic_lna_random" : car uplink (random k1) →
                                              LNA downlink (random k2).
                Single-LNA pipeline (one global zone):
                    - "semantic_lna_single"        : same as semantic_lna but
                                                      all cars in one zone.
                    - "semantic_lna_random_single" : random baseline for
                                                      single-zone pipeline.
                Global pipeline (same broadcast to every agent):
                    - "ranking_priority"    : global type-based priority ranking.
                    - "ranking_random"      : random selection.

            rule_yaml_file (str): Path to the rule YAML file.

            top_k1 (int): (UNUSED) Retained for config compatibility. Uplink
                          now transmits all FOV entities unconditionally.

            top_k2 (int): Downlink budget — entities each LNA transmits back
                          to an ego car.  Only used by semantic_lna /
                          semantic_lna_random.

            agent_region (int | None): Active agent region size, forwarded to
                          the LNA grid so only relevant LNAs are instantiated.
        """
        if selection_mode not in self.VALID_SELECTION_MODES:
            raise ValueError(
                f"Unknown GNA selection_mode: {selection_mode!r}. "
                f"Must be one of {self.VALID_SELECTION_MODES}."
            )

        self.broadcast_history = []
        self.current_global_context = {}
        self.enabled = True
        self.broadcast_id_counter = 0

        self.top_k = top_k
        self.top_k1 = top_k1
        self.top_k2 = top_k2
        self.selection_mode = selection_mode
        self.rule_yaml_file = rule_yaml_file
        self.use_extended_rules = 'extended' in rule_yaml_file
        self.use_spatial_rules = 'spatial' in rule_yaml_file
        self.use_discriminative_rules = 'discriminative' in rule_yaml_file
        self.hypotheses = get_hypotheses_for_rule_file(rule_yaml_file)
        self.metrics_tracker = None

        # Lazy-init LNA grid only when needed
        self.lna_grid = None
        if selection_mode in ("semantic_lna", "semantic_lna_random"):
            from .lna import LocalNavigationAssistant
            self.lna_grid = LocalNavigationAssistant(
                world_size=WORLD_SIZE,
                agent_region=agent_region,
            )

        if self.use_discriminative_rules:
            self.rule_mode = 'discriminative'
            rule_label = 'discriminative (30)'
        elif self.use_spatial_rules:
            self.rule_mode = 'spatial'
            rule_label = 'spatial (30)'
        elif self.use_extended_rules:
            self.rule_mode = 'extended'
            rule_label = 'extended (48)'
        else:
            self.rule_mode = 'original'
            rule_label = 'original (12)'

        extra = ''
        if selection_mode in ("semantic_lna", "semantic_lna_random",
                               "semantic_lna_single", "semantic_lna_random_single"):
            extra = f", k1=all (no uplink selection), k2={self.top_k2}"
        logging.info(f"Global Navigation Assistant (GNA) initialized - "
                     f"Status: {'ENABLED' if self.enabled else 'DISABLED'}, "
                     f"Top-K: {self.top_k}{extra}, "
                     f"Selection Mode: {self.selection_mode}, "
                     f"Rule set: {rule_label}")

    # ==================================================================================
    # MAIN ORCHESTRATION - Called each timestep
    # ==================================================================================

    def orchestrate_global_reasoning(self, city_env):
        """
        MAIN ENTRY POINT - Called each timestep by City.update().

        Dispatches to one of three pipelines based on ``self.selection_mode``:

        **Semantic pipeline** (``"semantic"`` / ``"semantic_random"``):
            1. Collect pre-grounding data from ALL agents.
            2. For each ego agent, ground predicates on its vicinity
               entities and select a subset (optimal or random).
            3. Distribute *personalised* per-agent broadcasts.

        **LNA pipeline** (``"semantic_lna*"`` modes):
            1. Collect pre-grounding data from ALL agents.
            2. Each car selects top-k1 FOV entities (uplink to its LNA).
            3. Each LNA aggregates, filters, re-grounds, and selects
               top-k2 entities per ego car (downlink).
            4. Distribute personalised per-agent broadcasts.
            Multi-zone (``semantic_lna`` / ``semantic_lna_random``) partitions
            cars across LNA zones.  Single-zone (``semantic_lna_single`` /
            ``semantic_lna_random_single``) puts all cars in one zone.

        **Global pipeline** (``"ranking_priority"`` / ``"ranking_random"``):
            1. Collect pre-grounding data from ALL agents.
            2. Select the global top-k entities by type-priority ranking
               or random sampling.
            3. Broadcast the *same* set of entities to every agent.

        Args:
            city_env: The City or CityEnv instance containing:
                      - city_env.agents: List of all agent objects
                      - city_env.city_grid: Current world state tensor
                      - city_env.intersection_matrix: Intersection information

        Returns:
            dict: Combined broadcast (union of all per-agent selections),
                  or None if GNA is disabled.
        """
        if not self.enabled:
            logging.debug("GNA: Orchestration skipped - GNA is disabled")
            return None

        logging.info(f"GNA: Starting orchestration cycle "
                     f"(mode={self.selection_mode})")

        # -----------------------------------------------------------------
        # PHASE 1: COLLECT - Gather pre-grounding data from ALL agents
        # (shared by both pipelines)
        # -----------------------------------------------------------------
        global_context = self.collect_pre_grounding_global(
            city_env.agents,
            city_env.city_grid,
            city_env.intersection_matrix,
        )

        if self.selection_mode in ("semantic", "semantic_random"):
            return self._run_semantic_pipeline(city_env, global_context)
        elif self.selection_mode in ("semantic_lna", "semantic_lna_random",
                                      "semantic_lna_single", "semantic_lna_random_single"):
            return self._run_lna_pipeline(city_env, global_context)
        else:
            return self._run_global_pipeline(city_env, global_context)

    # ------------------------------------------------------------------
    # Semantic pipeline  ("semantic" / "semantic_random")
    # ------------------------------------------------------------------

    def _run_semantic_pipeline(self, city_env, global_context):
        """Per-agent vicinity grounding + optimal or random subset selection."""
        from ..utils.sim_metrics import ego_groundings_from_properties

        layer_to_agent = self._build_layer_to_agent_map(city_env.agents)

        per_agent_selections = {}
        agent_layer_ids = set(layer_to_agent.keys())

        for ego_id, ego_data in global_context.items():
            selected, all_grounded = self.select_vicinity_entities_for_agent(
                ego_id, ego_data, layer_to_agent,
                city_env.intersection_matrix,
            )
            per_agent_selections[ego_id] = selected

            # ----- Metrics evaluation -----
            if self.metrics_tracker is not None:
                ego_layer_id = ego_data['agent_properties']['layer_id']
                ego_agent = layer_to_agent.get(ego_layer_id)
                if ego_agent is None:
                    continue

                # Ground FOV entities (deduplicated by agent)
                fov_groundings = []
                fov_seen = set()
                for fe in ego_data.get('fov_entities', []):
                    fe_layer = fe['layer_id']
                    fe_agent = layer_to_agent.get(fe_layer)
                    if fe_agent is None or fe_agent.layer_id == ego_layer_id:
                        continue
                    fe_id = f"{fe_agent.type}_{fe_agent.id}"
                    if fe_id in fov_seen:
                        continue
                    fov_seen.add(fe_id)
                    fg = self.ground_predicates_for_vicinity_entity(
                        ego_agent, fe_agent, city_env.intersection_matrix,
                    )
                    fov_groundings.append({'groundings': fg})

                # Build entity groundings lists for evaluator
                baseline_entities = fov_groundings + [
                    {'groundings': e['groundings']} for e in all_grounded
                ]
                test_entities = fov_groundings + [
                    {'groundings': e['groundings']} for e in selected
                ]

                # Resolve ego groundings
                all_entities = baseline_entities or test_entities
                if all_entities:
                    ego_gr = all_entities[0]['groundings']['unary_ego']
                else:
                    ego_gr = ego_groundings_from_properties(
                        ego_data['agent_properties']
                    )

                # Flatten to evaluator format
                bl_flat = [e['groundings'] for e in baseline_entities]
                te_flat = [e['groundings'] for e in test_entities]

                evaluator = self.metrics_tracker.evaluator
                baseline_eval = evaluator.evaluate(ego_gr, bl_flat)
                test_eval = evaluator.evaluate(ego_gr, te_flat)

                self.metrics_tracker.record_step(
                    ego_id, baseline_eval, test_eval,
                )


        per_agent_broadcasts, combined_broadcast = \
            self.broadcast_per_agent_context(global_context, per_agent_selections)

        city_env.set_global_context_for_agents(per_agent_broadcasts)

        n_agents = len(global_context)
        if n_agents > 0:
            fov_counts = [
                len({e['layer_id'] for e in d.get('fov_entities', [])
                     if e['layer_id'] in agent_layer_ids})
                for d in global_context.values()
            ]
            vic_counts = [
                len({e['layer_id'] for e in d.get('vicinity_entities', [])
                     if e['layer_id'] in agent_layer_ids})
                for d in global_context.values()
            ]
            logging.info(
                f"GNA: Agents per ego — "
                f"FOV: avg={sum(fov_counts)/n_agents:.1f}, "
                f"min={min(fov_counts)}, max={max(fov_counts)} | "
                f"Vicinity: avg={sum(vic_counts)/n_agents:.1f}, "
                f"min={min(vic_counts)}, max={max(vic_counts)}"
            )

        logging.info("GNA: Semantic pipeline completed successfully")
        return combined_broadcast

    # ------------------------------------------------------------------
    # LNA pipeline  (all "semantic_lna*" modes)
    # ------------------------------------------------------------------

    def _run_lna_pipeline(self, city_env, global_context):
        """Two-way car ↔ LNA communication pipeline.

        Phase 1 (Uplink):   Each car transmits ALL FOV entities + itself to
                             the LNA.  No predicate grounding or selection is
                             performed (Phase 2 re-grounds from the downlink
                             ego's perspective).
        Phase 2 (LNA):      Each LNA aggregates other cars' transmissions,
                             filters, deduplicates, re-grounds from ego
                             perspective, and selects k2 for each ego car.
        Phase 3 (Downlink): Ego car receives FOV + k2 entities from LNA.

        Multi-zone modes (semantic_lna / semantic_lna_random) partition cars
        across an LNA grid.  Single-zone modes (semantic_lna_single /
        semantic_lna_random_single) place all cars in one global zone.
        """
        from ..utils.sim_metrics import ego_groundings_from_properties

        use_semantic = self.selection_mode in ("semantic_lna", "semantic_lna_single")
        layer_to_agent = self._build_layer_to_agent_map(city_env.agents)
        agent_layer_ids = set(layer_to_agent.keys())

        # Identify car agents only (cars transmit and receive)
        car_ids = []
        car_positions = {}
        for agent_id, agent_data in global_context.items():
            if agent_data['agent_properties']['type'] == 'Car':
                car_ids.append(agent_id)
                pos = agent_data['agent_properties']['position']
                car_positions[agent_id] = (int(pos[0]), int(pos[1]))

        # =============================================================
        # PHASE 1 — Uplink: each car transmits ALL FOV entities + itself
        #   No predicate grounding or k1 selection — Phase 2 re-grounds
        #   everything from the downlink ego's perspective.
        # =============================================================
        uplink = {}  # car_id → list of entity dicts (identity only)

        for car_id in car_ids:
            car_data = global_context[car_id]
            car_layer_id = car_data['agent_properties']['layer_id']
            car_agent = layer_to_agent.get(car_layer_id)
            if car_agent is None:
                continue

            fov_entities_raw = car_data.get('fov_entities', [])
            uplink_entries = []
            seen_fov = set()
            for fe in fov_entities_raw:
                fe_layer = fe['layer_id']
                fe_agent = layer_to_agent.get(fe_layer)
                if fe_agent is None:
                    continue
                if fe_agent.layer_id == car_layer_id:
                    continue
                fe_id = f"{fe_agent.type}_{fe_agent.id}"
                if fe_id in seen_fov:
                    continue
                seen_fov.add(fe_id)
                uplink_entries.append({
                    'entity_id': fe_id,
                    'layer_id': fe_layer,
                    'agent_ref': fe_agent,
                    'position': fe['position'],
                })

            # Include the car itself in the uplink
            car_pos = car_data['agent_properties']['position']
            uplink_entries.append({
                'entity_id': car_id,
                'layer_id': car_layer_id,
                'agent_ref': car_agent,
                'position': car_pos,
            })

            uplink[car_id] = uplink_entries

        # =============================================================
        # PHASE 2 — LNA Processing
        # =============================================================
        if self.selection_mode in ("semantic_lna_single", "semantic_lna_random_single"):
            lna_to_cars = {0: list(car_ids)}
        else:
            lna_to_cars = self.lna_grid.assign_cars_to_lnas(car_positions)

        downlink = {}  # ego_car_id → list of re-grounded entity dicts

        for lna_id, zone_car_ids in lna_to_cars.items():
            if len(zone_car_ids) < 2:
                for cid in zone_car_ids:
                    downlink[cid] = []
                continue

            for ego_car_id in zone_car_ids:
                ego_data = global_context[ego_car_id]
                ego_layer_id = ego_data['agent_properties']['layer_id']
                ego_agent = layer_to_agent.get(ego_layer_id)
                if ego_agent is None:
                    downlink[ego_car_id] = []
                    continue

                ego_pos = ego_data['agent_properties']['position']
                ego_fov = ego_data['world_state']['fov_boundaries']

                # Pool entities from OTHER cars in same LNA zone
                pool = {}  # entity_id → entity dict (dedup)
                for other_car_id in zone_car_ids:
                    if other_car_id == ego_car_id:
                        continue
                    for ent in uplink.get(other_car_id, []):
                        eid = ent['entity_id']
                        if eid == ego_car_id:
                            continue
                        if eid not in pool:
                            pool[eid] = ent

                # Filter: not in ego's FOV
                filtered = []
                for eid, ent in pool.items():
                    ex, ey = int(ent['position'][0]), int(ent['position'][1])
                    if (ego_fov['x_start'] <= ex < ego_fov['x_end'] and
                            ego_fov['y_start'] <= ey < ego_fov['y_end']):
                        continue
                    filtered.append(ent)

                # Filter: in ego's vicinity
                vic_x_start, vic_y_start, vic_x_end, vic_y_end = \
                    self.get_vicinity_fov(
                        ego_pos, ego_agent.last_move_dir if hasattr(ego_agent, 'last_move_dir') else None,
                        city_env.city_grid.shape[1],
                        city_env.city_grid.shape[2],
                    )
                vicinity_filtered = []
                for ent in filtered:
                    ex, ey = int(ent['position'][0]), int(ent['position'][1])
                    if (vic_x_start <= ex < vic_x_end and
                            vic_y_start <= ey < vic_y_end):
                        vicinity_filtered.append(ent)

                # Re-ground from ego car's perspective
                regrounded = []
                for ent in vicinity_filtered:
                    ent_agent = ent.get('agent_ref')
                    if ent_agent is None:
                        ent_agent = layer_to_agent.get(ent.get('layer_id'))
                    if ent_agent is None:
                        continue
                    new_gr = self.ground_predicates_for_vicinity_entity(
                        ego_agent, ent_agent, city_env.intersection_matrix,
                    )
                    regrounded.append({
                        'entity_id': ent['entity_id'],
                        'layer_id': ent.get('layer_id'),
                        'agent_ref': ent_agent,
                        'position': ent['position'],
                        'groundings': new_gr,
                        'agent_properties': {
                            'type': ent_agent.type,
                            'position': ent['position'],
                            'goal': (ent_agent.goal.tolist()
                                     if hasattr(ent_agent.goal, 'tolist')
                                     else ent_agent.goal),
                            'current_action': ent_agent.last_move_dir,
                            'priority': ent_agent.priority,
                            'layer_id': ent.get('layer_id'),
                            'concepts': (ent_agent.concepts
                                         if hasattr(ent_agent, 'concepts') else {}),
                            'is_at_intersection': new_gr['unary_entity']['IsAtInter'],
                            'is_in_intersection': new_gr['unary_entity']['IsInInter'],
                        },
                    })

                # Select k2 from re-grounded pool
                if len(regrounded) <= self.top_k2:
                    selected_down = list(regrounded)
                elif use_semantic:
                    for e in regrounded:
                        e['q_sentence'] = groundings_to_q_sentence(e['groundings'])
                    selected_down, _ = select_optimal_subset(
                        regrounded, self.hypotheses, self.top_k2,
                    )
                else:
                    rng = random.Random(time.time_ns())
                    selected_down = rng.sample(regrounded, self.top_k2)

                downlink[ego_car_id] = selected_down

        # =============================================================
        # PHASE 3 — Build per-agent broadcasts + metrics
        # =============================================================
        per_agent_selections = {}

        for ego_id, ego_data in global_context.items():
            ego_layer_id = ego_data['agent_properties']['layer_id']
            ego_agent = layer_to_agent.get(ego_layer_id)

            if ego_id in downlink:
                per_agent_selections[ego_id] = downlink[ego_id]
            else:
                per_agent_selections[ego_id] = []

            # ----- Metrics evaluation -----
            # In LNA modes only cars participate; skip non-car agents so they
            # don't drag down DSR/TSR with an unfair baseline-vs-empty comparison.
            if self.metrics_tracker is not None and ego_agent is not None \
                    and ego_data['agent_properties']['type'] == 'Car':
                # FOV groundings (same as semantic pipeline)
                fov_groundings = []
                fov_seen = set()
                for fe in ego_data.get('fov_entities', []):
                    fe_layer = fe['layer_id']
                    fe_agent = layer_to_agent.get(fe_layer)
                    if fe_agent is None or fe_agent.layer_id == ego_layer_id:
                        continue
                    fe_id = f"{fe_agent.type}_{fe_agent.id}"
                    if fe_id in fov_seen:
                        continue
                    fov_seen.add(fe_id)
                    fg = self.ground_predicates_for_vicinity_entity(
                        ego_agent, fe_agent, city_env.intersection_matrix,
                    )
                    fov_groundings.append({'groundings': fg})

                # ALL vicinity entities for baseline
                all_vicinity_grounded = []
                vic_entities = ego_data.get('vicinity_entities', [])
                vic_seen = set()
                for ve in vic_entities:
                    ve_layer = ve['layer_id']
                    ve_agent = layer_to_agent.get(ve_layer)
                    if ve_agent is None or ve_agent.layer_id == ego_layer_id:
                        continue
                    ve_id = f"{ve_agent.type}_{ve_agent.id}"
                    if ve_id in vic_seen:
                        continue
                    vic_seen.add(ve_id)
                    vg = self.ground_predicates_for_vicinity_entity(
                        ego_agent, ve_agent, city_env.intersection_matrix,
                    )
                    all_vicinity_grounded.append({'groundings': vg})

                baseline_entities = fov_groundings + all_vicinity_grounded
                test_entities = fov_groundings + [
                    {'groundings': e['groundings']}
                    for e in per_agent_selections.get(ego_id, [])
                ]

                all_entities = baseline_entities or test_entities
                if all_entities:
                    ego_gr = all_entities[0]['groundings']['unary_ego']
                else:
                    ego_gr = ego_groundings_from_properties(
                        ego_data['agent_properties']
                    )

                bl_flat = [e['groundings'] for e in baseline_entities]
                te_flat = [e['groundings'] for e in test_entities]

                evaluator = self.metrics_tracker.evaluator
                baseline_eval = evaluator.evaluate(ego_gr, bl_flat)
                test_eval = evaluator.evaluate(ego_gr, te_flat)

                self.metrics_tracker.record_step(
                    ego_id, baseline_eval, test_eval,
                )

        per_agent_broadcasts, combined_broadcast = \
            self.broadcast_per_agent_context(global_context, per_agent_selections)

        city_env.set_global_context_for_agents(per_agent_broadcasts)

        n_agents = len(global_context)
        n_cars = len(car_ids)
        n_with_downlink = sum(1 for d in downlink.values() if d)
        logging.info(
            f"GNA: LNA pipeline completed — "
            f"{n_cars} cars, {n_with_downlink} received downlink, "
            f"mode={self.selection_mode}, k1=all, k2={self.top_k2}"
        )

        return combined_broadcast

    # ------------------------------------------------------------------
    # Global pipeline  ("ranking_priority" / "ranking_random")
    # ------------------------------------------------------------------

    def _run_global_pipeline(self, city_env, global_context):
        """Global type-priority / random selection — same broadcast to all."""
        broadcast = self.broadcast_global_context(global_context)

        per_agent_broadcasts = {
            f"{agent.type}_{agent.id}": broadcast
            for agent in city_env.agents
            if agent is not None
        }

        city_env.set_global_context_for_agents(per_agent_broadcasts)

        logging.info("GNA: Global pipeline completed successfully "
                     f"(mode={self.selection_mode})")
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
            # Get list of entities in near-vicinity zone (beyond FOV, within AGENT_VICINITY)
            # -----------------------------------------------------------------
            vicinity_entities = self.get_agent_vicinity_entities(agent, city_grid)

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
                'vicinity_entities': vicinity_entities,
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

        Supports two global-pipeline modes:
            - "ranking_priority": Use get_entity_priority() ranking
            - "ranking_random":   Random selection (for experiments)
        
        Args:
            global_context: Full global context dict with all agents
        
        Returns:
            dict: Filtered global context with only top-k entities
        """
        # k=0 means no global context (GNA effectively disabled)
        if self.top_k <= 0:
            return {}

        if self.selection_mode == "ranking_priority":
            ranked_entities = self.rank_entities_by_priority(global_context)
            selected_entities = ranked_entities[:self.top_k]
        elif self.selection_mode == "ranking_random":
            selected_entities = self.select_random_entities(global_context)
        else:
            raise ValueError(f"Unknown GNA global-pipeline mode: {self.selection_mode}. "
                             f"Must be 'ranking_priority' or 'ranking_random'")

        top_k_ids = {agent_id for agent_id, _ in selected_entities}

        filtered_context = {
            agent_id: global_context[agent_id]
            for agent_id in top_k_ids
            if agent_id in global_context
        }

        logging.debug(f"GNA: Filtered to top-{self.top_k} entities "
                      f"({self.selection_mode}) from {len(global_context)} "
                      f"total. Selected: {list(filtered_context.keys())}")
        
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

        logging.info(f"GNA: Broadcasting filtered global context - "
                     f"ID: {broadcast['broadcast_id']}, "
                     f"Total agents: {len(global_context)}, "
                     f"Filtered: {len(filtered_context)} "
                     f"(top-{self.top_k}, {self.selection_mode})")

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
    # PHASE 3: PREDICATE-BASED PER-AGENT SELECTION
    # ==================================================================================
    # New pipeline: ground predicates for each ego agent's vicinity entities,
    # select top-k using those groundings, and create per-agent broadcasts.
    # ==================================================================================

    def _build_layer_to_agent_map(self, all_agents):
        """Build a lookup dict from grid layer_id to agent object."""
        mapping = {}
        for agent in all_agents:
            if agent is not None:
                mapping[agent.layer_id] = agent
        return mapping

    def _compute_intersection_state(self, agent_type, position, intersection_matrix):
        """
        Compute is_at_intersection and is_in_intersection for an entity
        at a given position, using the intersection_matrix directly.
        """
        pos_x, pos_y = int(position[0]), int(position[1])
        is_at = False
        is_in = False

        if intersection_matrix is None:
            return is_at, is_in

        if agent_type == "Car" or "car" in agent_type.lower():
            if (intersection_matrix.shape[0] > 0 and
                    pos_x < intersection_matrix.shape[1] and
                    pos_y < intersection_matrix.shape[2]):
                is_at = bool(intersection_matrix[0, pos_x, pos_y].item())
            if (intersection_matrix.shape[0] > 2 and
                    pos_x < intersection_matrix.shape[1] and
                    pos_y < intersection_matrix.shape[2]):
                is_in = bool(intersection_matrix[2, pos_x, pos_y].item())
        else:
            if (intersection_matrix.shape[0] > 1 and
                    pos_x < intersection_matrix.shape[1] and
                    pos_y < intersection_matrix.shape[2]):
                is_at = bool(intersection_matrix[1, pos_x, pos_y].item())
            if (intersection_matrix.shape[0] > 2 and
                    pos_x < intersection_matrix.shape[1] and
                    pos_y < intersection_matrix.shape[2]):
                is_in = bool(intersection_matrix[2, pos_x, pos_y].item())

        return is_at, is_in

    def ground_predicates_for_vicinity_entity(self, ego_agent, entity_agent, intersection_matrix):
        """
        Ground all predicates for a vicinity entity relative to the ego agent.

        Creates PesudoAgent objects for both ego and entity, then evaluates
        every predicate using the same grounding functions Z3 uses (from
        pred_converter/z3.py).  The world_matrix argument is None because all
        spatial lookups go through PesudoAgent.world_pos when in_fov_matrix=False.

        Returns:
            dict with keys 'unary_entity', 'unary_ego', 'binary_ego_entity',
            'binary_entity_ego' -- each mapping predicate names to bool.
        """
        ego_pos = ego_agent.pos.tolist() if hasattr(ego_agent.pos, 'tolist') else list(ego_agent.pos)
        entity_pos = entity_agent.pos.tolist() if hasattr(entity_agent.pos, 'tolist') else list(entity_agent.pos)

        ego_is_at, ego_is_in = self._compute_intersection_state(
            ego_agent.type, ego_pos, intersection_matrix)
        entity_is_at, entity_is_in = self._compute_intersection_state(
            entity_agent.type, entity_pos, intersection_matrix)

        ego_pseudo = PesudoAgent(
            type=ego_agent.type,
            layer_id=ego_agent.layer_id,
            concepts=ego_agent.concepts if hasattr(ego_agent, 'concepts') else {},
            moving_direction=ego_agent.last_move_dir if hasattr(ego_agent, 'last_move_dir') else None,
            world_pos=ego_pos,
            in_fov_matrix=False,
            is_at_intersection=ego_is_at,
            is_in_intersection=ego_is_in,
        )

        entity_pseudo = PesudoAgent(
            type=entity_agent.type,
            layer_id=entity_agent.layer_id,
            concepts=entity_agent.concepts if hasattr(entity_agent, 'concepts') else {},
            moving_direction=entity_agent.last_move_dir if hasattr(entity_agent, 'last_move_dir') else None,
            world_pos=entity_pos,
            in_fov_matrix=False,
            is_at_intersection=entity_is_at,
            is_in_intersection=entity_is_in,
        )

        agents_dict = {
            str(ego_agent.layer_id): ego_pseudo,
            str(entity_agent.layer_id): entity_pseudo,
        }

        ego_name = f"Entity_{ego_agent.type}_{ego_agent.layer_id}"
        entity_name = f"Entity_{entity_agent.type}_{entity_agent.layer_id}"

        wm = None  # world_matrix unused when in_fov_matrix=False
        im = intersection_matrix

        unary_entity = {
            'IsCar':        bool(IsCar(wm, im, agents_dict, entity_name)),
            'IsPedestrian': bool(IsPed(wm, im, agents_dict, entity_name)),
            'IsAmbulance':  bool(IsAmb(wm, im, agents_dict, entity_name)),
            'IsBus':        bool(IsBus(wm, im, agents_dict, entity_name)),
            'IsPolice':     bool(IsPolice(wm, im, agents_dict, entity_name)),
            'IsTiro':       bool(IsTiro(wm, im, agents_dict, entity_name)),
            'IsReckless':   bool(IsReckless(wm, im, agents_dict, entity_name)),
            'IsOld':        bool(IsOld(wm, im, agents_dict, entity_name)),
            'IsYoung':      bool(IsYoung(wm, im, agents_dict, entity_name)),
            'IsAtInter':    bool(IsAtInter(wm, im, agents_dict, entity_name)),
            'IsInInter':    bool(IsInInter(wm, im, agents_dict, entity_name)),
        }

        unary_ego = {
            'IsCar':        bool(IsCar(wm, im, agents_dict, ego_name)),
            'IsPedestrian': bool(IsPed(wm, im, agents_dict, ego_name)),
            'IsAmbulance':  bool(IsAmb(wm, im, agents_dict, ego_name)),
            'IsBus':        bool(IsBus(wm, im, agents_dict, ego_name)),
            'IsPolice':     bool(IsPolice(wm, im, agents_dict, ego_name)),
            'IsTiro':       bool(IsTiro(wm, im, agents_dict, ego_name)),
            'IsReckless':   bool(IsReckless(wm, im, agents_dict, ego_name)),
            'IsOld':        bool(IsOld(wm, im, agents_dict, ego_name)),
            'IsYoung':      bool(IsYoung(wm, im, agents_dict, ego_name)),
            'IsAtInter':    bool(IsAtInter(wm, im, agents_dict, ego_name)),
            'IsInInter':    bool(IsInInter(wm, im, agents_dict, ego_name)),
        }

        binary_ego_entity = {
            'IsClose':        bool(IsClose(wm, im, agents_dict, ego_name, entity_name)),
            'CollidingClose': bool(CollidingClose(wm, im, agents_dict, ego_name, entity_name)),
            'LeftOf':         bool(LeftOf(wm, im, agents_dict, ego_name, entity_name)),
            'RightOf':        bool(RightOf(wm, im, agents_dict, ego_name, entity_name)),
            'NextTo':         bool(NextTo(wm, im, agents_dict, ego_name, entity_name)),
            'HigherPri':      bool(HigherPri(wm, im, agents_dict, ego_name, entity_name)),
        }

        binary_entity_ego = {
            'IsClose':        bool(IsClose(wm, im, agents_dict, entity_name, ego_name)),
            'CollidingClose': bool(CollidingClose(wm, im, agents_dict, entity_name, ego_name)),
            'LeftOf':         bool(LeftOf(wm, im, agents_dict, entity_name, ego_name)),
            'RightOf':        bool(RightOf(wm, im, agents_dict, entity_name, ego_name)),
            'NextTo':         bool(NextTo(wm, im, agents_dict, entity_name, ego_name)),
            'HigherPri':      bool(HigherPri(wm, im, agents_dict, entity_name, ego_name)),
        }

        return {
            'unary_entity': unary_entity,
            'unary_ego': unary_ego,
            'binary_ego_entity': binary_ego_entity,
            'binary_entity_ego': binary_entity_ego,
        }

    def select_vicinity_entities_for_agent(self, ego_id, ego_data, layer_to_agent,
                                           intersection_matrix):
        """
        Ground predicates for the ego agent's vicinity entities and select top-k.

        For each vicinity entity, grounds all predicates relative to the ego
        agent, converts groundings to a Q-sentence, and uses
        ``select_optimal_subset`` (inductive-probability-based semantic
        selection) to choose the k entities that minimize the communication
        objective.

        Args:
            ego_id: Agent identifier string, e.g. "Car_1".
            ego_data: This agent's entry from global_context.
            layer_to_agent: Dict mapping layer_id -> agent object.
            intersection_matrix: Intersection tensor [3, W, H].

        Returns:
            tuple of (selected, grounded_entities):
            - selected: list of dicts chosen for broadcast
            - grounded_entities: full list of all grounded vicinity entities
              (before selection), for metrics baseline evaluation.
        """
        vicinity_entities = ego_data.get('vicinity_entities', [])
        if not vicinity_entities:
            return [], []

        ego_layer_id = ego_data['agent_properties']['layer_id']
        ego_agent = layer_to_agent.get(ego_layer_id)
        if ego_agent is None:
            logging.warning(f"GNA: Could not find ego agent for {ego_id} "
                            f"(layer_id={ego_layer_id})")
            return [], []

        grounded_entities = []
        for ve in vicinity_entities:
            entity_layer_id = ve['layer_id']
            entity_agent = layer_to_agent.get(entity_layer_id)
            if entity_agent is None:
                continue
            if entity_agent.layer_id == ego_agent.layer_id:
                continue

            groundings = self.ground_predicates_for_vicinity_entity(
                ego_agent, entity_agent, intersection_matrix
            )

            entity_id = f"{entity_agent.type}_{entity_agent.id}"
            grounded_entities.append({
                'entity_id': entity_id,
                'agent_properties': {
                    'type': entity_agent.type,
                    'position': ve['position'],
                    'goal': (entity_agent.goal.tolist()
                             if hasattr(entity_agent.goal, 'tolist')
                             else entity_agent.goal),
                    'current_action': entity_agent.last_move_dir,
                    'priority': entity_agent.priority,
                    'layer_id': entity_layer_id,
                    'concepts': (entity_agent.concepts
                                 if hasattr(entity_agent, 'concepts') else {}),
                    'is_at_intersection': groundings['unary_entity']['IsAtInter'],
                    'is_in_intersection': groundings['unary_entity']['IsInInter'],
                },
                'groundings': groundings,
            })

        seen_ids = set()
        unique_grounded = []
        for ge in grounded_entities:
            eid = ge['entity_id']
            if eid not in seen_ids:
                seen_ids.add(eid)
                unique_grounded.append(ge)
        grounded_entities = unique_grounded

        if self.selection_mode == "semantic":
            for entity in grounded_entities:
                entity['q_sentence'] = groundings_to_q_sentence(
                    entity['groundings']
                )

            selected, score = select_optimal_subset(
                grounded_entities, self.hypotheses, self.top_k,
            )

            logging.debug(f"GNA: Agent {ego_id} - "
                          f"{len(vicinity_entities)} vicinity entities, "
                          f"{len(grounded_entities)} grounded, "
                          f"{len(selected)} selected (obj_score={score:.6e})")
        else:
            # semantic_random: random subset from grounded vicinity entities
            if len(grounded_entities) <= self.top_k:
                selected = list(grounded_entities)
            else:
                rng = random.Random(time.time_ns())
                selected = rng.sample(grounded_entities, self.top_k)

            logging.debug(f"GNA: Agent {ego_id} - "
                          f"{len(vicinity_entities)} vicinity entities, "
                          f"{len(grounded_entities)} grounded, "
                          f"{len(selected)} selected (semantic_random)")

        return selected, grounded_entities

    def broadcast_per_agent_context(self, global_context, per_agent_selections):
        """
        Create per-agent broadcast messages and a combined broadcast for Z3.

        Args:
            global_context: Full global context from collect_pre_grounding_global().
            per_agent_selections: Dict mapping ego_id -> list of selected entity dicts.

        Returns:
            (per_agent_broadcasts, combined_broadcast)
            per_agent_broadcasts: {ego_id: broadcast_dict} for distribution.
            combined_broadcast: Single broadcast (union of all selections) for Z3
                                backward-compatibility.
        """
        broadcast_id = f"gna_broadcast_{self.broadcast_id_counter}"
        timestamp = time.time()

        per_agent_broadcasts = {}
        all_selected = {}

        for ego_id, selected_entities in per_agent_selections.items():
            selected_context = {}
            for entity in selected_entities:
                eid = entity['entity_id']
                entry = {
                    'agent_properties': entity['agent_properties'],
                    'groundings': entity['groundings'],
                    'fov_entities': [],
                    'vicinity_entities': [],
                    'world_state': {},
                    'environmental_context': {
                        'nearby_intersections': [],
                        'traffic_conditions': {
                            'density': 0,
                            'nearby_agents': [],
                            'potential_conflicts': [],
                        },
                        'movable_region': None,
                    },
                    'collection_timestamp': timestamp,
                }
                selected_context[eid] = entry
                if eid not in all_selected:
                    all_selected[eid] = entry

            per_agent_broadcasts[ego_id] = {
                'broadcast_id': broadcast_id,
                'timestamp': timestamp,
                'global_context': selected_context,
                'metadata': {
                    'total_agents': len(global_context),
                    'filtered_agents': len(selected_context),
                    'top_k': self.top_k,
                    'collection_time': timestamp,
                    'broadcast_size': len(selected_context),
                },
            }

        combined_broadcast = {
            'broadcast_id': broadcast_id,
            'timestamp': timestamp,
            'global_context': all_selected,
            'metadata': {
                'total_agents': len(global_context),
                'filtered_agents': len(all_selected),
                'top_k': self.top_k,
                'collection_time': timestamp,
                'broadcast_size': len(all_selected),
            },
        }

        self.current_global_context = global_context
        self.broadcast_history.append(combined_broadcast)
        self.broadcast_id_counter += 1

        agents_with_data = sum(1 for s in per_agent_selections.values() if s)
        logging.info(f"GNA: Per-agent broadcasts created - ID: {broadcast_id}, "
                     f"Total agents: {len(global_context)}, "
                     f"Agents with selections: {agents_with_data}, "
                     f"Unique entities selected: {len(all_selected)}")

        return per_agent_broadcasts, combined_broadcast

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

    def get_agent_vicinity_entities(self, agent, city_grid):
        """
        Get entities in the near-vicinity zone: beyond AGENT_FOV but within AGENT_VICINITY.
        Returns entities in the ring between the two radii, excluding those already in the FOV.
        """
        vicinity_entities = []

        fov_x_start, fov_y_start, fov_x_end, fov_y_end = self.get_fov(
            agent.pos, agent.last_move_dir,
            city_grid.shape[1], city_grid.shape[2]
        )

        vic_x_start, vic_y_start, vic_x_end, vic_y_end = self.get_vicinity_fov(
            agent.pos, agent.last_move_dir,
            city_grid.shape[1], city_grid.shape[2]
        )

        for layer_idx in range(city_grid.shape[0]):
            layer = city_grid[layer_idx, vic_x_start:vic_x_end, vic_y_start:vic_y_end]

            nonzero_pos = torch.nonzero(layer, as_tuple=False)
            for pos in nonzero_pos:
                actual_x = pos[0] + vic_x_start
                actual_y = pos[1] + vic_y_start

                if (fov_x_start <= actual_x < fov_x_end and
                        fov_y_start <= actual_y < fov_y_end):
                    continue

                entity_value = layer[pos[0], pos[1]].item()
                if entity_value != 0:
                    vicinity_entities.append({
                        'layer_id': layer_idx,
                        'position': [actual_x, actual_y],
                        'entity_type': LABEL_MAP.get(entity_value, 'unknown'),
                        'entity_value': entity_value
                    })

        return vicinity_entities

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

    def get_vicinity_fov(self, position, direction, width, height):
        """
        Calculate near-vicinity boundaries using AGENT_VICINITY radius.
        Same directional logic as get_fov() but with the larger radius.
        """
        if direction == None:
            x_start = max(position[0] - AGENT_VICINITY, 0)
            y_start = max(position[1] - AGENT_VICINITY, 0)
            x_end = min(position[0] + AGENT_VICINITY + 1, width)
            y_end = min(position[1] + AGENT_VICINITY + 1, height)
        elif direction == "Left":
            x_start = max(position[0] - AGENT_VICINITY, 0)
            y_start = max(position[1] - AGENT_VICINITY, 0)
            x_end = min(position[0] + AGENT_VICINITY + 1, width)
            y_end = min(position[1] + 2, height)
        elif direction == "Right":
            x_start = max(position[0] - AGENT_VICINITY, 0)
            y_start = max(position[1] - 2, 0)
            x_end = min(position[0] + AGENT_VICINITY + 1, width)
            y_end = min(position[1] + AGENT_VICINITY + 1, height)
        elif direction == "Up":
            x_start = max(position[0] - AGENT_VICINITY, 0)
            y_start = max(position[1] - AGENT_VICINITY, 0)
            x_end = min(position[0] + 2, width)
            y_end = min(position[1] + AGENT_VICINITY + 1, height)
        elif direction == "Down":
            x_start = max(position[0] - 2, 0)
            y_start = max(position[1] - AGENT_VICINITY, 0)
            x_end = min(position[0] + AGENT_VICINITY + 1, width)
            y_end = min(position[1] + AGENT_VICINITY + 1, height)
        else:
            x_start = max(position[0] - AGENT_VICINITY, 0)
            y_start = max(position[1] - AGENT_VICINITY, 0)
            x_end = min(position[0] + AGENT_VICINITY + 1, width)
            y_end = min(position[1] + AGENT_VICINITY + 1, height)

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
