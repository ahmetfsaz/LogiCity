"""
Simulation Metrics System
=========================
Evaluates the impact of GNA entity selection on agent decision-making by
comparing subrule evaluations under two information settings:

  Baseline : FOV + ALL vicinity entities  (full observable information)
  Test     : FOV + GNA-selected entities  (limited information)

Three metrics are produced:

  1. Decision Success Rate (Subrule-level)  —  **ego-gate-filtered**
     Each subrule has the form  EgoGate(ego) AND EntityCondition(entities).
     The ego gate depends only on the ego agent's own predicates (type,
     location) and is identical under baseline and test — it carries no
     information about GNA quality.  To avoid diluting the metric with
     trivially-correct evaluations, we **only count subrules whose ego
     gate is True** for the current agent at the current step.  Among those
     active subrules, we measure the fraction whose evaluation matches
     between baseline and test.  Rates are computed per agent, then averaged.

  2. Decision Success Rate (Action-level)
     Fraction of steps where the derived action (Stop / Slow / Fast / Normal)
     matches between baseline and test.

  3. Trajectory Success Rate
     Per-trajectory binary metric: the agent reached its goal without any
     action-level rule violation (mismatch between baseline and test
     actions).  No time/horizon limit is imposed.  Incomplete trajectories
     (simulation ended while in-flight) are excluded from the rate but
     reported separately.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# Subrule IDs — original 12, extended 48, spatial 30, discriminative 30
# ============================================================================
SUBRULE_IDS_ORIGINAL = [
    'stop_1', 'stop_2', 'stop_3', 'stop_4', 'stop_5', 'stop_6', 'stop_7',
    'slow_1', 'slow_2',
    'fast_1', 'fast_2', 'fast_3',
]

SUBRULE_IDS_EXTENDED = (
    [f'stop_{i}' for i in range(1, 21)]
    + [f'slow_{i}' for i in range(1, 16)]
    + [f'fast_{i}' for i in range(1, 14)]
)

SUBRULE_IDS_SPATIAL = (
    [f'stop_{i}' for i in range(1, 11)]
    + [f'slow_{i}' for i in range(1, 11)]
    + [f'fast_{i}' for i in range(1, 11)]
)

SUBRULE_IDS_DISCRIMINATIVE = (
    [f'stop_{i}' for i in range(1, 11)]
    + [f'slow_{i}' for i in range(1, 11)]
    + [f'fast_{i}' for i in range(1, 11)]
)

SUBRULE_IDS = SUBRULE_IDS_ORIGINAL  # default, overridden at tracker init


def ego_groundings_from_properties(agent_properties):
    """
    Derive an ego-grounding dict (same schema as ``unary_ego``) from the
    ``agent_properties`` dictionary that GNA stores per agent.

    Used as a fallback when no entity groundings are available (empty FOV
    and empty vicinity), so the ego predicates can still be evaluated.
    """
    concepts = agent_properties.get('concepts', {})
    agent_type = agent_properties.get('type', '')
    return {
        'IsCar':        agent_type == 'Car',
        'IsPedestrian': agent_type == 'Pedestrian',
        'IsAmbulance':  bool(concepts.get('ambulance', 0)),
        'IsBus':        bool(concepts.get('bus', 0)),
        'IsPolice':     bool(concepts.get('police', 0)),
        'IsTiro':       bool(concepts.get('tiro', 0)),
        'IsReckless':   bool(concepts.get('reckless', 0)),
        'IsOld':        bool(concepts.get('old', 0)),
        'IsYoung':      bool(concepts.get('young', 0)),
        'IsAtInter':    agent_properties.get('is_at_intersection', False),
        'IsInInter':    agent_properties.get('is_in_intersection', False),
    }


# ============================================================================
# SubruleEvaluator
# ============================================================================

class SubruleEvaluator:
    """
    Evaluates subrule conditions for simulation metrics.

    Supports four rule-set modes selectable via ``mode`` parameter:

    * ``'original'`` (default): 12 subrules from ``expert_rule.yaml``.
    * ``'extended'``:  48 subrules from ``expert_rule_extended.yaml``.
    * ``'spatial'``:   30 subrules from ``expert_rule_spatial.yaml``.
    * ``'discriminative'``: 30 subrules from ``expert_rule_discriminative.yaml``.

    Slow subrule 3 from the original rule file (involving two non-ego
    entities) is excluded in both original and extended modes.

    Returns three items per evaluation
    ----------------------------------
    **subrule_results** : dict[str, bool]
        Independent evaluation of each subrule (``Not(Stop)`` stripped
        from Slow / Fast conditions).

    **ego_gates** : dict[str, bool]
        The ego-only portion of each subrule.  When False, the subrule
        is trivially False regardless of entity information.  Used by
        ``SimMetricsTracker.record_step`` to exclude trivially-correct
        evaluations from the subrule-level DSR.

    **action** : str
        Cascading action derivation (Stop > Slow > Fast > Normal) with
        full ``stop_fired`` gating for Slow / Fast.
    """

    def __init__(self, extended=False, mode=None):
        if mode is not None:
            self.mode = mode
        elif extended:
            self.mode = 'extended'
        else:
            self.mode = 'original'
        self.extended = self.mode == 'extended'
        self.spatial = self.mode == 'spatial'
        self.discriminative = self.mode == 'discriminative'

    def evaluate(self, ego_groundings, entity_groundings_list):
        """
        Parameters
        ----------
        ego_groundings : dict
            Keys match the ``unary_ego`` schema:
            ``{'IsCar': bool, …, 'IsAtInter': bool, 'IsInInter': bool}``
        entity_groundings_list : list[dict]
            Each dict has keys ``'unary_entity'``, ``'binary_ego_entity'``,
            ``'binary_entity_ego'`` — each mapping predicate names to bool.

        Returns
        -------
        dict  with keys:
            ``'subrule_results'`` : dict[str, bool]
            ``'ego_gates'``       : dict[str, bool]
            ``'action'``          : str
        """
        sr = {}
        eg = {}  # ego gates: True when ego-only predicates pass
        el = entity_groundings_list  # shorthand

        # ==== STOP SUBRULES ==================================================

        eg['stop_1'] = (
            not ego_groundings.get('IsAmbulance', False)
            and not ego_groundings.get('IsOld', False)
            and ego_groundings.get('IsAtInter', False)
        )
        sr['stop_1'] = (
            eg['stop_1']
            and any(e['unary_entity'].get('IsInInter', False) for e in el)
        )

        eg['stop_2'] = (
            not ego_groundings.get('IsAmbulance', False)
            and not ego_groundings.get('IsOld', False)
            and ego_groundings.get('IsAtInter', False)
        )
        sr['stop_2'] = (
            eg['stop_2']
            and any(
                e['unary_entity'].get('IsAtInter', False)
                and e['binary_entity_ego'].get('HigherPri', False)
                for e in el)
        )

        eg['stop_3'] = (
            not ego_groundings.get('IsAmbulance', False)
            and not ego_groundings.get('IsOld', False)
            and ego_groundings.get('IsInInter', False)
        )
        sr['stop_3'] = (
            eg['stop_3']
            and any(
                e['unary_entity'].get('IsInInter', False)
                and e['unary_entity'].get('IsAmbulance', False)
                for e in el)
        )

        eg['stop_4'] = (
            not ego_groundings.get('IsAmbulance', False)
            and not ego_groundings.get('IsPolice', False)
            and ego_groundings.get('IsCar', False)
            and not ego_groundings.get('IsInInter', False)
            and not ego_groundings.get('IsAtInter', False)
        )
        sr['stop_4'] = (
            eg['stop_4']
            and any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsPolice', False)
                for e in el)
        )

        eg['stop_5'] = (
            ego_groundings.get('IsBus', False)
            and not ego_groundings.get('IsInInter', False)
            and not ego_groundings.get('IsAtInter', False)
        )
        sr['stop_5'] = (
            eg['stop_5']
            and any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsPedestrian', False)
                for e in el)
        )

        eg['stop_6'] = ego_groundings.get('IsAmbulance', False)
        sr['stop_6'] = (
            eg['stop_6']
            and any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['unary_entity'].get('IsOld', False)
                for e in el)
        )

        eg['stop_7'] = (
            not ego_groundings.get('IsAmbulance', False)
            and not ego_groundings.get('IsOld', False)
        )
        sr['stop_7'] = (
            eg['stop_7']
            and any(
                e['binary_ego_entity'].get('CollidingClose', False)
                for e in el)
        )

        if self.extended:
            eg['stop_8'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsInInter', False)
            )
            sr['stop_8'] = (
                eg['stop_8']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsPedestrian', False)
                    for e in el)
            )

            eg['stop_9'] = ego_groundings.get('IsPedestrian', False)
            sr['stop_9'] = (
                eg['stop_9']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['stop_10'] = ego_groundings.get('IsPedestrian', False)
            sr['stop_10'] = (
                eg['stop_10']
                and any(
                    e['binary_entity_ego'].get('LeftOf', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['stop_11'] = (
                ego_groundings.get('IsCar', False)
                and ego_groundings.get('IsAtInter', False)
            )
            sr['stop_11'] = (
                eg['stop_11']
                and any(
                    e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsAmbulance', False)
                    for e in el)
            )

            eg['stop_12'] = (
                not ego_groundings.get('IsBus', False)
                and ego_groundings.get('IsAtInter', False)
            )
            sr['stop_12'] = (
                eg['stop_12']
                and any(
                    e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsBus', False)
                    for e in el)
            )

            eg['stop_13'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsInInter', False)
            )
            sr['stop_13'] = (
                eg['stop_13']
                and any(
                    e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsOld', False)
                    for e in el)
            )

            eg['stop_14'] = ego_groundings.get('IsPedestrian', False)
            sr['stop_14'] = (
                eg['stop_14']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsAmbulance', False)
                    for e in el)
            )

            eg['stop_15'] = ego_groundings.get('IsPedestrian', False)
            sr['stop_15'] = (
                eg['stop_15']
                and any(
                    e['binary_entity_ego'].get('CollidingClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['stop_16'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsPolice', False)
            )
            sr['stop_16'] = (
                eg['stop_16']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsPolice', False)
                    for e in el)
            )

            eg['stop_17'] = (
                ego_groundings.get('IsCar', False)
                and ego_groundings.get('IsInInter', False)
            )
            sr['stop_17'] = (
                eg['stop_17']
                and any(
                    e['binary_entity_ego'].get('HigherPri', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    for e in el)
            )

            eg['stop_18'] = (
                not ego_groundings.get('IsOld', False)
                and not ego_groundings.get('IsAmbulance', False)
            )
            sr['stop_18'] = (
                eg['stop_18']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsAmbulance', False)
                    and e['unary_entity'].get('IsAtInter', False)
                    for e in el)
            )

            eg['stop_19'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsReckless', False)
            )
            sr['stop_19'] = (
                eg['stop_19']
                and any(
                    e['binary_ego_entity'].get('CollidingClose', False)
                    and e['unary_entity'].get('IsPedestrian', False)
                    for e in el)
            )

            eg['stop_20'] = (
                ego_groundings.get('IsPedestrian', False)
                and ego_groundings.get('IsAtInter', False)
            )
            sr['stop_20'] = (
                eg['stop_20']
                and any(
                    e['unary_entity'].get('IsInInter', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

        # ==== SLOW SUBRULES (Not(Stop) stripped for Mode A) ===================

        eg['slow_1'] = ego_groundings.get('IsTiro', False)
        sr['slow_1'] = (
            eg['slow_1']
            and any(
                e['unary_entity'].get('IsPedestrian', False)
                and e['binary_ego_entity'].get('IsClose', False)
                for e in el)
        )

        eg['slow_2'] = (
            ego_groundings.get('IsTiro', False)
            and ego_groundings.get('IsInInter', False)
        )
        sr['slow_2'] = (
            eg['slow_2']
            and any(
                e['unary_entity'].get('IsAtInter', False)
                for e in el)
        )

        if self.extended:
            eg['slow_3'] = ego_groundings.get('IsCar', False)
            sr['slow_3'] = (
                eg['slow_3']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsPedestrian', False)
                    for e in el)
            )

            eg['slow_4'] = ego_groundings.get('IsPedestrian', False)
            sr['slow_4'] = (
                eg['slow_4']
                and any(
                    e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['slow_5'] = (
                ego_groundings.get('IsCar', False)
                and ego_groundings.get('IsAtInter', False)
            )
            sr['slow_5'] = (
                eg['slow_5']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['slow_6'] = ego_groundings.get('IsCar', False)
            sr['slow_6'] = (
                eg['slow_6']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsOld', False)
                    for e in el)
            )

            eg['slow_7'] = (
                ego_groundings.get('IsPedestrian', False)
                and ego_groundings.get('IsAtInter', False)
            )
            sr['slow_7'] = (
                eg['slow_7']
                and any(
                    e['unary_entity'].get('IsAtInter', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['slow_8'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsAmbulance', False)
            )
            sr['slow_8'] = (
                eg['slow_8']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsAmbulance', False)
                    for e in el)
            )

            eg['slow_9'] = ego_groundings.get('IsPedestrian', False)
            sr['slow_9'] = (
                eg['slow_9']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsBus', False)
                    for e in el)
            )

            eg['slow_10'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsAtInter', False)
            )
            sr['slow_10'] = (
                eg['slow_10']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsAtInter', False)
                    for e in el)
            )

            eg['slow_11'] = ego_groundings.get('IsPedestrian', False)
            sr['slow_11'] = (
                eg['slow_11']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsReckless', False)
                    for e in el)
            )

            eg['slow_12'] = ego_groundings.get('IsCar', False)
            sr['slow_12'] = (
                eg['slow_12']
                and any(
                    e['binary_entity_ego'].get('LeftOf', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['slow_13'] = (
                not ego_groundings.get('IsAmbulance', False)
                and ego_groundings.get('IsInInter', False)
            )
            sr['slow_13'] = (
                eg['slow_13']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['slow_14'] = (
                ego_groundings.get('IsPedestrian', False)
                and ego_groundings.get('IsInInter', False)
            )
            sr['slow_14'] = (
                eg['slow_14']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['slow_15'] = ego_groundings.get('IsCar', False)
            sr['slow_15'] = (
                eg['slow_15']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsPedestrian', False)
                    for e in el)
            )

        # ==== FAST SUBRULES (Not(Stop) stripped for Mode A) ===================

        eg['fast_1'] = ego_groundings.get('IsReckless', False)
        sr['fast_1'] = (
            eg['fast_1']
            and any(
                e['unary_entity'].get('IsAtInter', False)
                for e in el)
        )

        eg['fast_2'] = ego_groundings.get('IsBus', False)
        sr['fast_2'] = eg['fast_2']

        eg['fast_3'] = ego_groundings.get('IsPolice', False)
        sr['fast_3'] = (
            eg['fast_3']
            and any(
                e['unary_entity'].get('IsReckless', False)
                for e in el)
        )

        if self.extended:
            eg['fast_4'] = ego_groundings.get('IsPolice', False)
            sr['fast_4'] = (
                eg['fast_4']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsAmbulance', False)
                    for e in el)
            )

            eg['fast_5'] = ego_groundings.get('IsAmbulance', False)
            sr['fast_5'] = (
                eg['fast_5']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['fast_6'] = ego_groundings.get('IsReckless', False)
            sr['fast_6'] = (
                eg['fast_6']
                and any(
                    e['binary_entity_ego'].get('LeftOf', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['fast_7'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsAtInter', False)
                and not ego_groundings.get('IsInInter', False)
            )
            sr['fast_7'] = (
                eg['fast_7']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['fast_8'] = (
                ego_groundings.get('IsBus', False)
                and not ego_groundings.get('IsAtInter', False)
                and not ego_groundings.get('IsInInter', False)
            )
            sr['fast_8'] = (
                eg['fast_8']
                and any(
                    e['binary_entity_ego'].get('LeftOf', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['fast_9'] = (
                ego_groundings.get('IsAmbulance', False)
                and ego_groundings.get('IsAtInter', False)
            )
            sr['fast_9'] = (
                eg['fast_9']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False)
                    for e in el)
            )

            eg['fast_10'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsBus', False)
                and not ego_groundings.get('IsAtInter', False)
            )
            sr['fast_10'] = (
                eg['fast_10']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsBus', False)
                    for e in el)
            )

            eg['fast_11'] = (
                ego_groundings.get('IsPedestrian', False)
                and not ego_groundings.get('IsOld', False)
                and not ego_groundings.get('IsAtInter', False)
            )
            sr['fast_11'] = (
                eg['fast_11']
                and any(
                    e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsPedestrian', False)
                    for e in el)
            )

            eg['fast_12'] = (
                ego_groundings.get('IsCar', False)
                and not ego_groundings.get('IsReckless', False)
                and not ego_groundings.get('IsAtInter', False)
            )
            sr['fast_12'] = (
                eg['fast_12']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['binary_ego_entity'].get('IsClose', False)
                    for e in el)
            )

            eg['fast_13'] = (
                ego_groundings.get('IsPedestrian', False)
                and not ego_groundings.get('IsOld', False)
                and not ego_groundings.get('IsInInter', False)
            )
            sr['fast_13'] = (
                eg['fast_13']
                and any(
                    e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsPedestrian', False)
                    for e in el)
            )

        # ==== SPATIAL SUBRULES (entirely separate rule set) ===================
        if self.spatial:
            sr.clear()
            eg.clear()

            # -- STOP (10) --
            # S1: CollidingClose(ego, x) AND IsCar(x)
            eg['stop_1'] = True
            sr['stop_1'] = any(
                e['binary_ego_entity'].get('CollidingClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # S2: CollidingClose(ego, x) AND IsPedestrian(x)
            eg['stop_2'] = True
            sr['stop_2'] = any(
                e['binary_ego_entity'].get('CollidingClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # S3: IsClose(x, ego) AND IsPedestrian(x)
            eg['stop_3'] = True
            sr['stop_3'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # S4: IsClose(x, ego) AND IsAmbulance(x)
            eg['stop_4'] = True
            sr['stop_4'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsAmbulance', False) for e in el)

            # S5: RightOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
            eg['stop_5'] = True
            sr['stop_5'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # S6: NextTo(x, ego) AND IsOld(x)
            eg['stop_6'] = True
            sr['stop_6'] = any(
                e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsOld', False) for e in el)

            # S7: LeftOf(x, ego) AND IsClose(x, ego) AND IsAmbulance(x)
            eg['stop_7'] = True
            sr['stop_7'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsAmbulance', False) for e in el)

            # S8: IsAtInter(ego) AND IsInInter(x)  [positional]
            eg['stop_8'] = ego_groundings.get('IsAtInter', False)
            sr['stop_8'] = (
                eg['stop_8']
                and any(e['unary_entity'].get('IsInInter', False) for e in el))

            # S9: IsInInter(ego) AND HigherPri(x, ego) AND IsClose(x, ego)  [positional]
            eg['stop_9'] = ego_groundings.get('IsInInter', False)
            sr['stop_9'] = (
                eg['stop_9']
                and any(
                    e['binary_entity_ego'].get('HigherPri', False)
                    and e['binary_entity_ego'].get('IsClose', False) for e in el))

            # S10: Not(IsAmbulance(ego)) AND IsClose(x, ego) AND IsAmbulance(x) AND IsAtInter(x)  [negated]
            eg['stop_10'] = not ego_groundings.get('IsAmbulance', False)
            sr['stop_10'] = (
                eg['stop_10']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsAmbulance', False)
                    and e['unary_entity'].get('IsAtInter', False) for e in el))

            # -- SLOW (10) --
            # SL1: IsClose(x, ego) AND IsCar(x)
            eg['slow_1'] = True
            sr['slow_1'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # SL2: IsClose(ego, x) AND IsPedestrian(x)
            eg['slow_2'] = True
            sr['slow_2'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # SL3: IsClose(x, ego) AND IsOld(x)
            eg['slow_3'] = True
            sr['slow_3'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsOld', False) for e in el)

            # SL4: IsClose(x, ego) AND IsReckless(x)
            eg['slow_4'] = True
            sr['slow_4'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsReckless', False) for e in el)

            # SL5: IsClose(ego, x) AND IsYoung(x)
            eg['slow_5'] = True
            sr['slow_5'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsYoung', False) for e in el)

            # SL6: LeftOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
            eg['slow_6'] = True
            sr['slow_6'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # SL7: RightOf(x, ego) AND IsClose(x, ego) AND IsPedestrian(x)
            eg['slow_7'] = True
            sr['slow_7'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # SL8: IsClose(ego, x) AND IsInInter(x)
            eg['slow_8'] = True
            sr['slow_8'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsInInter', False) for e in el)

            # SL9: IsAtInter(ego) AND IsClose(x, ego) AND IsCar(x)  [positional]
            eg['slow_9'] = ego_groundings.get('IsAtInter', False)
            sr['slow_9'] = (
                eg['slow_9']
                and any(
                    e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False) for e in el))

            # SL10: Not(IsAmbulance(ego)) AND NextTo(x, ego) AND IsAmbulance(x)  [negated]
            eg['slow_10'] = not ego_groundings.get('IsAmbulance', False)
            sr['slow_10'] = (
                eg['slow_10']
                and any(
                    e['binary_entity_ego'].get('NextTo', False)
                    and e['unary_entity'].get('IsAmbulance', False) for e in el))

            # -- FAST (10) --
            # F1: HigherPri(ego, x) AND IsClose(ego, x)
            eg['fast_1'] = True
            sr['fast_1'] = any(
                e['binary_ego_entity'].get('HigherPri', False)
                and e['binary_ego_entity'].get('IsClose', False) for e in el)

            # F2: HigherPri(ego, x) AND IsClose(ego, x) AND IsCar(x)
            eg['fast_2'] = True
            sr['fast_2'] = any(
                e['binary_ego_entity'].get('HigherPri', False)
                and e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # F3: Not(IsAtInter) AND Not(IsInInter) AND HigherPri(ego,x) AND IsCar(x)  [positional]
            eg['fast_3'] = (
                not ego_groundings.get('IsAtInter', False)
                and not ego_groundings.get('IsInInter', False))
            sr['fast_3'] = (
                eg['fast_3']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['unary_entity'].get('IsCar', False) for e in el))

            # F4: IsAtInter(ego) AND HigherPri(ego, x) AND IsAtInter(x)  [positional]
            eg['fast_4'] = ego_groundings.get('IsAtInter', False)
            sr['fast_4'] = (
                eg['fast_4']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['unary_entity'].get('IsAtInter', False) for e in el))

            # F5: IsInInter(ego) AND HigherPri(ego, x) AND IsClose(ego, x)  [positional]
            eg['fast_5'] = ego_groundings.get('IsInInter', False)
            sr['fast_5'] = (
                eg['fast_5']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['binary_ego_entity'].get('IsClose', False) for e in el))

            # F6: Not(IsAtInter) AND RightOf(x,ego) AND IsClose(ego,x) AND IsCar(x)  [positional]
            eg['fast_6'] = not ego_groundings.get('IsAtInter', False)
            sr['fast_6'] = (
                eg['fast_6']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False) for e in el))

            # F7: Not(IsInInter) AND Not(IsAtInter) AND RightOf(x,ego) AND IsClose(x,ego) AND IsCar(x)  [positional]
            eg['fast_7'] = (
                not ego_groundings.get('IsInInter', False)
                and not ego_groundings.get('IsAtInter', False))
            sr['fast_7'] = (
                eg['fast_7']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_entity_ego'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False) for e in el))

            # F8: Not(IsAtInter) AND LeftOf(x,ego) AND IsClose(ego,x) AND IsBus(x)  [positional]
            eg['fast_8'] = not ego_groundings.get('IsAtInter', False)
            sr['fast_8'] = (
                eg['fast_8']
                and any(
                    e['binary_entity_ego'].get('LeftOf', False)
                    and e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsBus', False) for e in el))

            # F9: Not(IsOld(ego)) AND HigherPri(ego,x) AND IsClose(ego,x) AND IsCar(x)  [negated]
            eg['fast_9'] = not ego_groundings.get('IsOld', False)
            sr['fast_9'] = (
                eg['fast_9']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsCar', False) for e in el))

            # F10: Not(IsOld(ego)) AND RightOf(x,ego) AND IsClose(ego,x)  [negated]
            eg['fast_10'] = not ego_groundings.get('IsOld', False)
            sr['fast_10'] = (
                eg['fast_10']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_ego_entity'].get('IsClose', False) for e in el))

        # ==== DISCRIMINATIVE SUBRULES (entirely separate rule set) ==============
        if self.discriminative:
            sr.clear()
            eg.clear()

            # -- STOP (10) --
            # S1: CollidingClose(ego, x) AND IsCar(x)
            eg['stop_1'] = True
            sr['stop_1'] = any(
                e['binary_ego_entity'].get('CollidingClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # S2: CollidingClose(ego, x) AND IsPedestrian(x)
            eg['stop_2'] = True
            sr['stop_2'] = any(
                e['binary_ego_entity'].get('CollidingClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # S3: IsClose(x, ego) AND IsAmbulance(x)
            eg['stop_3'] = True
            sr['stop_3'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsAmbulance', False) for e in el)

            # S4: NextTo(x, ego) AND IsOld(x)
            eg['stop_4'] = True
            sr['stop_4'] = any(
                e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsOld', False) for e in el)

            # S5: RightOf(x, ego) AND IsClose(x, ego) AND IsPolice(x)
            eg['stop_5'] = True
            sr['stop_5'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsPolice', False) for e in el)

            # S6: LeftOf(x, ego) AND NextTo(x, ego) AND IsPedestrian(x)
            eg['stop_6'] = True
            sr['stop_6'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # S7: RightOf(x, ego) AND IsClose(x, ego) AND IsReckless(x)
            eg['stop_7'] = True
            sr['stop_7'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsReckless', False) for e in el)

            # S8: NextTo(x, ego) AND IsBus(x)
            eg['stop_8'] = True
            sr['stop_8'] = any(
                e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsBus', False) for e in el)

            # S9: LeftOf(x, ego) AND IsClose(x, ego) AND IsTiro(x)
            eg['stop_9'] = True
            sr['stop_9'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsTiro', False) for e in el)

            # S10: IsClose(x, ego) AND IsYoung(x) AND RightOf(x, ego)
            eg['stop_10'] = True
            sr['stop_10'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsYoung', False)
                and e['binary_entity_ego'].get('RightOf', False) for e in el)

            # -- SLOW (10) --
            # SL1: IsClose(x, ego) AND IsCar(x)
            eg['slow_1'] = True
            sr['slow_1'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # SL2: IsClose(ego, x) AND IsPedestrian(x)
            eg['slow_2'] = True
            sr['slow_2'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # SL3: RightOf(x, ego) AND IsClose(x, ego) AND IsAmbulance(x)
            eg['slow_3'] = True
            sr['slow_3'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsAmbulance', False) for e in el)

            # SL4: LeftOf(x, ego) AND IsClose(x, ego) AND IsBus(x)
            eg['slow_4'] = True
            sr['slow_4'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsBus', False) for e in el)

            # SL5: NextTo(x, ego) AND IsPolice(x)
            eg['slow_5'] = True
            sr['slow_5'] = any(
                e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsPolice', False) for e in el)

            # SL6: IsClose(x, ego) AND IsOld(x)
            eg['slow_6'] = True
            sr['slow_6'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsOld', False) for e in el)

            # SL7: IsClose(x, ego) AND IsReckless(x)
            eg['slow_7'] = True
            sr['slow_7'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsReckless', False) for e in el)

            # SL8: IsClose(ego, x) AND IsTiro(x)
            eg['slow_8'] = True
            sr['slow_8'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsTiro', False) for e in el)

            # SL9: LeftOf(x, ego) AND NextTo(x, ego) AND IsYoung(x)
            eg['slow_9'] = True
            sr['slow_9'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_entity_ego'].get('NextTo', False)
                and e['unary_entity'].get('IsYoung', False) for e in el)

            # SL10: IsClose(x, ego) AND IsCar(x) AND HigherPri(x, ego)
            eg['slow_10'] = True
            sr['slow_10'] = any(
                e['binary_entity_ego'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False)
                and e['binary_entity_ego'].get('HigherPri', False) for e in el)

            # -- FAST (10) --
            # F1: HigherPri(ego, x) AND IsClose(ego, x) AND IsCar(x)
            eg['fast_1'] = True
            sr['fast_1'] = any(
                e['binary_ego_entity'].get('HigherPri', False)
                and e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsCar', False) for e in el)

            # F2: HigherPri(ego, x) AND NextTo(x, ego)
            eg['fast_2'] = True
            sr['fast_2'] = any(
                e['binary_ego_entity'].get('HigherPri', False)
                and e['binary_entity_ego'].get('NextTo', False) for e in el)

            # F3: RightOf(x, ego) AND IsClose(ego, x) AND IsTiro(x)
            eg['fast_3'] = True
            sr['fast_3'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsTiro', False) for e in el)

            # F4: LeftOf(x, ego) AND IsClose(ego, x) AND IsReckless(x)
            eg['fast_4'] = True
            sr['fast_4'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsReckless', False) for e in el)

            # F5: IsClose(ego, x) AND IsBus(x) AND HigherPri(ego, x)
            eg['fast_5'] = True
            sr['fast_5'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsBus', False)
                and e['binary_ego_entity'].get('HigherPri', False) for e in el)

            # F6: RightOf(x, ego) AND IsClose(ego, x) AND IsPedestrian(x)
            eg['fast_6'] = True
            sr['fast_6'] = any(
                e['binary_entity_ego'].get('RightOf', False)
                and e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsPedestrian', False) for e in el)

            # F7: Not(IsAtInter(ego)) AND Not(IsInInter(ego)) AND HigherPri(ego, x) AND IsCar(x)
            eg['fast_7'] = (
                not ego_groundings.get('IsAtInter', False)
                and not ego_groundings.get('IsInInter', False))
            sr['fast_7'] = (
                eg['fast_7']
                and any(
                    e['binary_ego_entity'].get('HigherPri', False)
                    and e['unary_entity'].get('IsCar', False) for e in el))

            # F8: IsClose(ego, x) AND IsPolice(x) AND HigherPri(ego, x)
            eg['fast_8'] = True
            sr['fast_8'] = any(
                e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsPolice', False)
                and e['binary_ego_entity'].get('HigherPri', False) for e in el)

            # F9: LeftOf(x, ego) AND IsClose(ego, x) AND IsOld(x)
            eg['fast_9'] = True
            sr['fast_9'] = any(
                e['binary_entity_ego'].get('LeftOf', False)
                and e['binary_ego_entity'].get('IsClose', False)
                and e['unary_entity'].get('IsOld', False) for e in el)

            # F10: Not(IsAtInter(ego)) AND RightOf(x, ego) AND IsClose(ego, x) AND IsAmbulance(x)
            eg['fast_10'] = not ego_groundings.get('IsAtInter', False)
            sr['fast_10'] = (
                eg['fast_10']
                and any(
                    e['binary_entity_ego'].get('RightOf', False)
                    and e['binary_ego_entity'].get('IsClose', False)
                    and e['unary_entity'].get('IsAmbulance', False) for e in el))

        # ==== MODE B: cascading action derivation =============================
        if self.spatial or self.discriminative:
            n_stop, n_slow, n_fast = 10, 10, 10
        elif self.extended:
            n_stop, n_slow, n_fast = 20, 15, 13
        else:
            n_stop, n_slow, n_fast = 7, 2, 3

        stop_fired = any(sr.get(f'stop_{i}', False) for i in range(1, n_stop + 1))

        if stop_fired:
            action = 'Stop'
        else:
            slow_fired = any(sr.get(f'slow_{i}', False) for i in range(1, n_slow + 1))
            fast_fired = any(sr.get(f'fast_{i}', False) for i in range(1, n_fast + 1))

            if slow_fired:
                action = 'Slow'
            elif fast_fired:
                action = 'Fast'
            else:
                action = 'Normal'

        return {
            'subrule_results': sr,
            'ego_gates': eg,
            'action': action,
        }


# ============================================================================
# SimMetricsTracker
# ============================================================================

class SimMetricsTracker:
    """
    Central tracker for simulation-mode metrics.

    Tracks per-agent, per-step subrule / action comparisons and per-agent,
    per-trajectory success.

    Usage
    -----
    1. ``start_trajectory(agent_id)``
       Called when an agent begins navigating toward a new goal.
    2. ``record_step(agent_id, baseline_eval, test_eval)``
       Called every simulation step (from GNA) with the evaluator output for
       both the baseline and test entity sets.
    3. ``end_trajectory(agent_id, reached_goal)``
       Called when the agent reaches its goal or when a new goal is assigned.
    4. ``get_summary()``
       Returns the full structured metrics report at end of simulation.
    """

    def __init__(self, extended=False, mode=None):
        if mode is not None:
            self.mode = mode
        elif extended:
            self.mode = 'extended'
        else:
            self.mode = 'original'
        self._evaluator = SubruleEvaluator(mode=self.mode)
        if self.mode == 'discriminative':
            self._subrule_ids = SUBRULE_IDS_DISCRIMINATIVE
        elif self.mode == 'spatial':
            self._subrule_ids = SUBRULE_IDS_SPATIAL
        elif self.mode == 'extended':
            self._subrule_ids = SUBRULE_IDS_EXTENDED
        else:
            self._subrule_ids = SUBRULE_IDS_ORIGINAL

        # Per-agent current trajectory state
        self._traj_state = {}  # agent_id -> dict

        # Counters for subrule-level decision success (per agent)
        self._subrule_correct = defaultdict(int)
        self._subrule_total = defaultdict(int)

        # Counters for action-level decision success (per agent)
        self._action_correct = defaultdict(int)
        self._action_total = defaultdict(int)

        # Per-subrule mismatch counts (global)
        self._subrule_mismatches = defaultdict(int)

        # Action confusion: (baseline_action, test_action) -> count
        self._action_confusion = defaultdict(int)

        # Completed trajectory records
        self._trajectory_records = []

        # Running trajectory ID counter per agent
        self._traj_counter = defaultdict(int)

        # Global step counter (set externally or incremented)
        self._current_step = 0

    @property
    def evaluator(self):
        return self._evaluator

    # -----------------------------------------------------------------
    # Trajectory lifecycle
    # -----------------------------------------------------------------

    def start_trajectory(self, agent_id):
        """Begin tracking a new trajectory for *agent_id*."""
        traj_id = self._traj_counter[agent_id]
        self._traj_counter[agent_id] += 1
        self._traj_state[agent_id] = {
            'traj_id': traj_id,
            'steps': 0,
            'violated': False,
        }
        logger.debug(f"Metrics: trajectory {traj_id} started for {agent_id}")

    def end_trajectory(self, agent_id, reached_goal):
        """Finalize and record the current trajectory for *agent_id*."""
        state = self._traj_state.get(agent_id)
        if state is None:
            logger.warning(f"Metrics: end_trajectory called for {agent_id} "
                           "but no trajectory is active")
            return

        success = reached_goal and not state['violated']

        self._trajectory_records.append({
            'agent_id': agent_id,
            'traj_id': state['traj_id'],
            'success': success,
            'reached_goal': reached_goal,
            'violated': state['violated'],
            'steps': state['steps'],
        })

        logger.debug(
            f"Metrics: trajectory {state['traj_id']} ended for {agent_id} — "
            f"success={success}, steps={state['steps']}, "
            f"violated={state['violated']}, reached_goal={reached_goal}"
        )

        del self._traj_state[agent_id]

    # -----------------------------------------------------------------
    # Per-step recording
    # -----------------------------------------------------------------

    def set_step(self, step):
        self._current_step = step

    def record_step(self, agent_id, baseline_eval, test_eval):
        """
        Compare baseline and test evaluator outputs and accumulate metrics.

        Subrule-level DSR only counts subrules whose **ego gate** is active
        (True) for the current agent at the current step.  Ego gates depend
        solely on the ego agent's own predicates (type, location) and are
        identical in baseline and test evaluations.  When the ego gate is
        False the subrule is trivially False in both settings, which would
        inflate the DSR without reflecting GNA's actual impact on entity-
        dependent decision-making.

        Parameters
        ----------
        agent_id : str
        baseline_eval : dict   output of ``SubruleEvaluator.evaluate()``
        test_eval     : dict   output of ``SubruleEvaluator.evaluate()``
        """
        bl_sr = baseline_eval['subrule_results']
        te_sr = test_eval['subrule_results']
        ego_gates = baseline_eval['ego_gates']

        n_correct = 0
        n_total = 0
        for sid in self._subrule_ids:
            if not ego_gates.get(sid, False):
                continue
            n_total += 1
            if bl_sr[sid] == te_sr[sid]:
                n_correct += 1
            else:
                self._subrule_mismatches[sid] += 1

        self._subrule_correct[agent_id] += n_correct
        self._subrule_total[agent_id] += n_total

        # Action-level
        action_match = baseline_eval['action'] == test_eval['action']
        self._action_correct[agent_id] += int(action_match)
        self._action_total[agent_id] += 1

        if not action_match:
            key = (baseline_eval['action'], test_eval['action'])
            self._action_confusion[key] += 1

        # Update trajectory state
        state = self._traj_state.get(agent_id)
        if state is not None:
            state['steps'] += 1
            if not action_match:
                state['violated'] = True

        logger.debug(
            f"Metrics step {self._current_step} {agent_id}: "
            f"subrule {n_correct}/{n_total}, "
            f"action {'MATCH' if action_match else 'MISMATCH'} "
            f"(baseline={baseline_eval['action']}, test={test_eval['action']})"
        )

    # -----------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------

    def get_decision_success_rate_subrule(self):
        """Per-agent ego-gate-filtered subrule accuracy, averaged across agents.

        Only subrules whose ego gate was True (i.e., the ego-only
        predicates were satisfied) at a given step contribute to the
        count.  This prevents agent-type mismatches (e.g., IsBus for a
        Car agent) from inflating the rate with trivially-correct False==False
        comparisons.
        """
        per_agent = {}
        for aid in self._subrule_total:
            total = self._subrule_total[aid]
            correct = self._subrule_correct[aid]
            per_agent[aid] = correct / max(total, 1)
        avg = sum(per_agent.values()) / max(len(per_agent), 1)
        return avg, per_agent

    def get_decision_success_rate_action(self):
        """Per-agent action accuracy, then average across agents."""
        per_agent = {}
        for aid in self._action_total:
            total = self._action_total[aid]
            correct = self._action_correct[aid]
            per_agent[aid] = correct / max(total, 1)
        avg = sum(per_agent.values()) / max(len(per_agent), 1)
        return avg, per_agent

    def get_trajectory_success_rate(self):
        """Successful / total across completed trajectories only.

        A trajectory is successful if the agent reached its goal without
        any action-level rule violation.  No time/horizon limit is imposed.
        Incomplete trajectories (simulation ended while in-flight) are
        excluded from the rate calculation but still reported separately.
        """
        n_incomplete = sum(1 for r in self._trajectory_records
                           if not r['success'] and not r['violated'])
        finished = [r for r in self._trajectory_records
                    if r['success'] or r['violated']]
        total_finished = len(finished)
        successful = sum(1 for r in finished if r['success'])
        n_violated = sum(1 for r in finished if r['violated'])
        rate = successful / max(total_finished, 1)
        return {
            'rate': rate,
            'total': total_finished,
            'successful': successful,
            'violated': n_violated,
            'incomplete': n_incomplete,
        }

    def finalize(self):
        """End all in-flight trajectories as incomplete (simulation ended)."""
        for agent_id in list(self._traj_state.keys()):
            self.end_trajectory(agent_id, reached_goal=False)

    def get_summary(self):
        """
        Return a JSON-serialisable summary dict containing all metrics.
        """
        self.finalize()
        sr_avg, sr_per_agent = self.get_decision_success_rate_subrule()
        act_avg, act_per_agent = self.get_decision_success_rate_action()
        traj = self.get_trajectory_success_rate()

        confusion_list = [
            {'baseline': k[0], 'test': k[1], 'count': v}
            for k, v in sorted(self._action_confusion.items(),
                                key=lambda x: -x[1])
        ]

        mismatch_list = [
            {'subrule': k, 'count': v}
            for k, v in sorted(self._subrule_mismatches.items(),
                                key=lambda x: -x[1])
        ]

        sr_total_correct = sum(self._subrule_correct.values())
        sr_total_evaluated = sum(self._subrule_total.values())
        sr_total_incorrect = sr_total_evaluated - sr_total_correct

        act_total_correct = sum(self._action_correct.values())
        act_total_evaluated = sum(self._action_total.values())
        act_total_incorrect = act_total_evaluated - act_total_correct

        return {
            'decision_success_rate_subrule': {
                'average': sr_avg,
                'correct': sr_total_correct,
                'incorrect': sr_total_incorrect,
                'total_evaluated': sr_total_evaluated,
                'per_agent': sr_per_agent,
                'subrule_mismatches': mismatch_list,
            },
            'decision_success_rate_action': {
                'average': act_avg,
                'correct': act_total_correct,
                'incorrect': act_total_incorrect,
                'total_evaluated': act_total_evaluated,
                'per_agent': act_per_agent,
                'action_confusion': confusion_list,
            },
            'trajectory_success_rate': traj,
            'trajectory_records': self._trajectory_records,
        }

    def log_summary(self):
        """Log the end-of-simulation summary at INFO level."""
        summary = self.get_summary()

        sr = summary['decision_success_rate_subrule']
        act = summary['decision_success_rate_action']
        traj = summary['trajectory_success_rate']

        logger.info("=" * 60)
        logger.info("=== Simulation Metrics Summary ===")
        logger.info("=" * 60)

        logger.info(f"Decision Success Rate (Subrule-level, ego-gate-filtered): "
                     f"{sr['average']:.4f}")
        logger.info(f"  Correct: {sr['correct']}  |  "
                     f"Incorrect: {sr['incorrect']}  |  "
                     f"Total evaluated: {sr['total_evaluated']}")
        top_agents = sorted(sr['per_agent'].items(), key=lambda x: x[0])
        for aid, rate in top_agents:
            logger.info(f"  {aid}: {rate:.4f}")
        if sr['subrule_mismatches']:
            logger.info("  Subrule mismatches:")
            for entry in sr['subrule_mismatches'][:5]:
                logger.info(f"    {entry['subrule']}: {entry['count']}")

        logger.info(f"Decision Success Rate (Action-level): "
                     f"{act['average']:.4f}")
        logger.info(f"  Correct: {act['correct']}  |  "
                     f"Incorrect: {act['incorrect']}  |  "
                     f"Total evaluated: {act['total_evaluated']}")
        top_agents = sorted(act['per_agent'].items(), key=lambda x: x[0])
        for aid, rate in top_agents:
            logger.info(f"  {aid}: {rate:.4f}")
        if act['action_confusion']:
            logger.info("  Action confusion:")
            for entry in act['action_confusion'][:5]:
                logger.info(f"    baseline={entry['baseline']} / "
                             f"test={entry['test']}: {entry['count']}")

        logger.info(f"Trajectory Success Rate: "
                     f"{traj['rate']:.4f} "
                     f"({traj['successful']}/{traj['total']})")
        logger.info(f"  Failures: {traj['violated']} violated")
        logger.info(f"  Incomplete (sim ended): {traj['incomplete']}")
        logger.info("=" * 60)

        return summary
