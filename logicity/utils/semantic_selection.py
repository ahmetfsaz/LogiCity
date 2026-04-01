"""
Goal-Oriented Semantic Selection via Inductive Probability
in a Dyadic First-Order Language
==========================================================

This module implements the optimization objective for selecting which
subset of grounded observations (evidence) to transmit via the Global
Navigation Assistant (GNA):

    min_{Ê ⊆ E}  Σ_i  p(Ê) · p(Γ_i | Ê) · (1 − p(Γ_i | Ê))

where Γ_i are traffic-rule hypotheses (Stop, Slow, Fast subrules).

Probability Model — Constituent-Based Inductive Logic
-----------------------------------------------------
Probabilities are assigned via uniform weight over *constituents*
(maximally specific world descriptions):

    p(e)   = |{C^w : C^w ⊨ e}| / |C|
    p(Γ|e) = |{C^w : C^w ⊨ e ∧ Γ}| / |{C^w : C^w ⊨ e}|

Hierarchy (bottom-up):

    1. Predicates: 11 monadic + 6 dyadic = 17 predicates.

    2. Q-sentences: Complete predicate-truth-value assignments for a
       pair (x1, x2).  Each monadic predicate appears for BOTH x1 and
       x2; each dyadic predicate appears in both directions P(x1,x2)
       and P(x2,x1).
           Total binary slots per Q-sentence:  T = 2·11 + 2·6 = 34
           Total Q-sentences:                  Q = 2^T = 2^34

    3. Attributive constituents (ACs): Subsets of Q-sentences that are
       asserted to "exist" (i.e., some pair in the world exhibits that
       relational pattern).
           Total ACs: 2^Q = 2^(2^34)

    4. Constituents: Subsets of ACs that are asserted to exist (i.e.,
       some entity in the world has that relational profile).
           Total constituents: 2^(2^Q) = 2^(2^(2^34))

LogiCity Special Case
---------------------
    - x1 is always the ego agent; x2 is a vicinity entity.
    - ALL 34 predicate slots are observed for every (ego, entity) pair.
    - Each pair therefore uniquely determines one Q-sentence.
    - Evidence = K distinct Q-sentences (fixed as existing).

Counting uses exclusion (complementary counting):

    Incompatible ACs  = those that do NOT include all K evidence
                        Q-sentences = 2^Q − 2^(Q−K)
    Excluded constituents = subsets of only incompatible ACs
                          = 2^(2^Q − 2^(Q−K))
    Compatible constituents = 2^(2^Q) − 2^(2^Q − 2^(Q−K))

    ⇒  p(e) = 1 − 2^(−2^(Q−K))
"""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple


# ====================================================================
# Type Aliases
# ====================================================================

QSentence = Tuple[bool, ...]
"""A complete Q-sentence: a length-34 tuple of bools, one per slot."""

Hypothesis = Dict[int, bool]
"""A partial slot assignment representing a traffic-rule hypothesis.
Maps slot indices (0..33) to required truth values."""


# ====================================================================
# Expert Subrules (Hypotheses)
# ====================================================================
#
# Extracted from config/rules/sim/expert/expert_rule.yaml.
# Each subrule is one And-clause in the traffic rules, serving as a
# hypothesis Γ_i in the optimization objective.
#
# Format: each subrule is a list of (slot_category, predicate, polarity)
# tuples, where:
#   slot_category : 'monadic_x1' | 'monadic_x2' | 'dyadic_x1_x2' | 'dyadic_x2_x1'
#   predicate     : predicate name string (e.g. 'IsAmbulance')
#   polarity      : True  = predicate is asserted
#                   False = predicate is negated (Not(...))
#
# YAML convention: entity = ego (x1), dummyEntityA = other (x2).
# Meta-predicates (Stop) are excluded — they are rule outcomes, not
# observable predicate slots.
#
# Slow subrule 3 (IsPolice(ego), IsYoung(otherA), IsYoung(otherB),
# NextTo(otherA, otherB)) is excluded entirely because it involves
# dummyEntityB, which cannot be represented in the (x1, x2) dyadic
# framework.
#

SubruleSpec = List[Tuple[str, str, bool]]

# ---- Original 12 subrules (from expert_rule.yaml) --------------------------

STOP_SUBRULES: List[SubruleSpec] = [
    # 1. Not(IsAmb(ego)), Not(IsOld(ego)), IsAtInter(ego), IsInInter(other)
    [('monadic_x1', 'IsAmbulance', False), ('monadic_x1', 'IsOld', False),
     ('monadic_x1', 'IsAtInter', True), ('monadic_x2', 'IsInInter', True)],
    # 2. Not(IsAmb(ego)), Not(IsOld(ego)), IsAtInter(ego), IsAtInter(other), HigherPri(other, ego)
    [('monadic_x1', 'IsAmbulance', False), ('monadic_x1', 'IsOld', False),
     ('monadic_x1', 'IsAtInter', True), ('monadic_x2', 'IsAtInter', True),
     ('dyadic_x2_x1', 'HigherPri', True)],
    # 3. Not(IsAmb(ego)), Not(IsOld(ego)), IsInInter(ego), IsInInter(other), IsAmb(other)
    [('monadic_x1', 'IsAmbulance', False), ('monadic_x1', 'IsOld', False),
     ('monadic_x1', 'IsInInter', True), ('monadic_x2', 'IsInInter', True),
     ('monadic_x2', 'IsAmbulance', True)],
    # 4. Not(IsAmb(ego)), Not(IsPolice(ego)), IsCar(ego), Not(IsInInter(ego)),
    #    Not(IsAtInter(ego)), LeftOf(other, ego), IsClose(other, ego), IsPolice(other)
    [('monadic_x1', 'IsAmbulance', False), ('monadic_x1', 'IsPolice', False),
     ('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsInInter', False),
     ('monadic_x1', 'IsAtInter', False), ('dyadic_x2_x1', 'LeftOf', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsPolice', True)],
    # 5. IsBus(ego), Not(IsInInter(ego)), Not(IsAtInter(ego)),
    #    RightOf(other, ego), NextTo(other, ego), IsPedestrian(other)
    [('monadic_x1', 'IsBus', True), ('monadic_x1', 'IsInInter', False),
     ('monadic_x1', 'IsAtInter', False), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsPedestrian', True)],
    # 6. IsAmb(ego), RightOf(other, ego), IsOld(other)
    [('monadic_x1', 'IsAmbulance', True), ('dyadic_x2_x1', 'RightOf', True),
     ('monadic_x2', 'IsOld', True)],
    # 7. Not(IsAmb(ego)), Not(IsOld(ego)), CollidingClose(ego, other)
    [('monadic_x1', 'IsAmbulance', False), ('monadic_x1', 'IsOld', False),
     ('dyadic_x1_x2', 'CollidingClose', True)],
]

SLOW_SUBRULES: List[SubruleSpec] = [
    # 1. IsTiro(ego), IsPedestrian(other), IsClose(ego, other)
    [('monadic_x1', 'IsTiro', True), ('monadic_x2', 'IsPedestrian', True),
     ('dyadic_x1_x2', 'IsClose', True)],
    # 2. IsTiro(ego), IsInInter(ego), IsAtInter(other)
    [('monadic_x1', 'IsTiro', True), ('monadic_x1', 'IsInInter', True),
     ('monadic_x2', 'IsAtInter', True)],
    # Subrule 3 excluded: involves dummyEntityB (see module header comment).
]

FAST_SUBRULES: List[SubruleSpec] = [
    # 1. IsReckless(ego), IsAtInter(other)
    [('monadic_x1', 'IsReckless', True), ('monadic_x2', 'IsAtInter', True)],
    # 2. IsBus(ego)
    [('monadic_x1', 'IsBus', True)],
    # 3. IsPolice(ego), IsReckless(other)
    [('monadic_x1', 'IsPolice', True), ('monadic_x2', 'IsReckless', True)],
]

EXPERT_SUBRULES: Dict[str, List[SubruleSpec]] = {
    "Stop": STOP_SUBRULES,
    "Slow": SLOW_SUBRULES,
    "Fast": FAST_SUBRULES,
}


# ---- Extended 48 subrules (from expert_rule_extended.yaml) ------------------
#
# 20 Stop + 15 Slow + 13 Fast.  The first 7/2/3 are identical to the
# originals above; the rest broaden ego gates and diversify entity-type
# / spatial-predicate requirements so that semantic selection cannot
# achieve DSR = 1.0 with just 1-2 selected entities.

EXTENDED_STOP_SUBRULES: List[SubruleSpec] = STOP_SUBRULES + [
    # S8: IsCar(ego) AND Not(IsInInter(ego)) AND IsClose(ego, x) AND IsPedestrian(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsInInter', False),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # S9: IsPedestrian(ego) AND RightOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # S10: IsPedestrian(ego) AND LeftOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'LeftOf', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # S11: IsCar(ego) AND IsAtInter(ego) AND NextTo(x, ego) AND IsAmbulance(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsAtInter', True),
     ('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsAmbulance', True)],
    # S12: Not(IsBus(ego)) AND IsAtInter(ego) AND NextTo(x, ego) AND IsBus(x)
    [('monadic_x1', 'IsBus', False), ('monadic_x1', 'IsAtInter', True),
     ('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsBus', True)],
    # S13: IsCar(ego) AND Not(IsInInter(ego)) AND NextTo(x, ego) AND IsOld(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsInInter', False),
     ('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsOld', True)],
    # S14: IsPedestrian(ego) AND IsClose(x, ego) AND IsAmbulance(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsAmbulance', True)],
    # S15: IsPedestrian(ego) AND CollidingClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'CollidingClose', True),
     ('monadic_x2', 'IsCar', True)],
    # S16: IsCar(ego) AND Not(IsPolice(ego)) AND RightOf(x, ego) AND NextTo(x, ego) AND IsPolice(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsPolice', False),
     ('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'NextTo', True),
     ('monadic_x2', 'IsPolice', True)],
    # S17: IsCar(ego) AND IsInInter(ego) AND HigherPri(x, ego) AND IsClose(x, ego)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsInInter', True),
     ('dyadic_x2_x1', 'HigherPri', True), ('dyadic_x2_x1', 'IsClose', True)],
    # S18: Not(IsOld(ego)) AND Not(IsAmbulance(ego)) AND IsClose(ego, x) AND IsAmbulance(x) AND IsAtInter(x)
    [('monadic_x1', 'IsOld', False), ('monadic_x1', 'IsAmbulance', False),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsAmbulance', True),
     ('monadic_x2', 'IsAtInter', True)],
    # S19: IsCar(ego) AND Not(IsReckless(ego)) AND CollidingClose(ego, x) AND IsPedestrian(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsReckless', False),
     ('dyadic_x1_x2', 'CollidingClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # S20: IsPedestrian(ego) AND IsAtInter(ego) AND IsInInter(x) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('monadic_x1', 'IsAtInter', True),
     ('monadic_x2', 'IsInInter', True), ('monadic_x2', 'IsCar', True)],
]

EXTENDED_SLOW_SUBRULES: List[SubruleSpec] = SLOW_SUBRULES + [
    # SL3: IsCar(ego) AND IsClose(ego, x) AND IsPedestrian(x)
    [('monadic_x1', 'IsCar', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsPedestrian', True)],
    # SL4: IsPedestrian(ego) AND NextTo(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'NextTo', True),
     ('monadic_x2', 'IsCar', True)],
    # SL5: IsCar(ego) AND IsAtInter(ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsAtInter', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # SL6: IsCar(ego) AND IsClose(ego, x) AND IsOld(x)
    [('monadic_x1', 'IsCar', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsOld', True)],
    # SL7: IsPedestrian(ego) AND IsAtInter(ego) AND IsAtInter(x) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('monadic_x1', 'IsAtInter', True),
     ('monadic_x2', 'IsAtInter', True), ('monadic_x2', 'IsCar', True)],
    # SL8: IsCar(ego) AND Not(IsAmbulance(ego)) AND IsClose(x, ego) AND IsAmbulance(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsAmbulance', False),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsAmbulance', True)],
    # SL9: IsPedestrian(ego) AND RightOf(x, ego) AND IsClose(x, ego) AND IsBus(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsBus', True)],
    # SL10: IsCar(ego) AND Not(IsAtInter(ego)) AND IsClose(ego, x) AND IsAtInter(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsAtInter', False),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsAtInter', True)],
    # SL11: IsPedestrian(ego) AND IsClose(x, ego) AND IsReckless(x)
    [('monadic_x1', 'IsPedestrian', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsReckless', True)],
    # SL12: IsCar(ego) AND LeftOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsCar', True), ('dyadic_x2_x1', 'LeftOf', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # SL13: Not(IsAmbulance(ego)) AND IsInInter(ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsAmbulance', False), ('monadic_x1', 'IsInInter', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # SL14: IsPedestrian(ego) AND IsInInter(ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsPedestrian', True), ('monadic_x1', 'IsInInter', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # SL15: IsCar(ego) AND RightOf(x, ego) AND NextTo(x, ego) AND IsPedestrian(x)
    [('monadic_x1', 'IsCar', True), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsPedestrian', True)],
]

EXTENDED_FAST_SUBRULES: List[SubruleSpec] = FAST_SUBRULES + [
    # F4: IsPolice(ego) AND IsClose(ego, x) AND IsAmbulance(x)
    [('monadic_x1', 'IsPolice', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsAmbulance', True)],
    # F5: IsAmbulance(ego) AND RightOf(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsAmbulance', True), ('dyadic_x2_x1', 'RightOf', True),
     ('monadic_x2', 'IsCar', True)],
    # F6: IsReckless(ego) AND LeftOf(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsReckless', True), ('dyadic_x2_x1', 'LeftOf', True),
     ('monadic_x2', 'IsCar', True)],
    # F7: IsCar(ego) AND Not(IsAtInter(ego)) AND Not(IsInInter(ego)) AND HigherPri(ego, x) AND IsCar(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsAtInter', False),
     ('monadic_x1', 'IsInInter', False), ('dyadic_x1_x2', 'HigherPri', True),
     ('monadic_x2', 'IsCar', True)],
    # F8: IsBus(ego) AND Not(IsAtInter(ego)) AND Not(IsInInter(ego)) AND LeftOf(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsBus', True), ('monadic_x1', 'IsAtInter', False),
     ('monadic_x1', 'IsInInter', False), ('dyadic_x2_x1', 'LeftOf', True),
     ('monadic_x2', 'IsCar', True)],
    # F9: IsAmbulance(ego) AND IsAtInter(ego) AND IsClose(x, ego) AND IsCar(x)
    [('monadic_x1', 'IsAmbulance', True), ('monadic_x1', 'IsAtInter', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # F10: IsCar(ego) AND Not(IsBus(ego)) AND Not(IsAtInter(ego)) AND RightOf(x, ego) AND IsClose(x, ego) AND IsBus(x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsBus', False),
     ('monadic_x1', 'IsAtInter', False), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsBus', True)],
    # F11: IsPedestrian(ego) AND Not(IsOld(ego)) AND Not(IsAtInter(ego)) AND IsClose(ego, x) AND IsPedestrian(x)
    [('monadic_x1', 'IsPedestrian', True), ('monadic_x1', 'IsOld', False),
     ('monadic_x1', 'IsAtInter', False), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsPedestrian', True)],
    # F12: IsCar(ego) AND Not(IsReckless(ego)) AND Not(IsAtInter(ego)) AND HigherPri(ego, x) AND IsClose(ego, x)
    [('monadic_x1', 'IsCar', True), ('monadic_x1', 'IsReckless', False),
     ('monadic_x1', 'IsAtInter', False), ('dyadic_x1_x2', 'HigherPri', True),
     ('dyadic_x1_x2', 'IsClose', True)],
    # F13: IsPedestrian(ego) AND Not(IsOld(ego)) AND Not(IsInInter(ego)) AND NextTo(x, ego) AND IsPedestrian(x)
    [('monadic_x1', 'IsPedestrian', True), ('monadic_x1', 'IsOld', False),
     ('monadic_x1', 'IsInInter', False), ('dyadic_x2_x1', 'NextTo', True),
     ('monadic_x2', 'IsPedestrian', True)],
]

EXTENDED_EXPERT_SUBRULES: Dict[str, List[SubruleSpec]] = {
    "Stop": EXTENDED_STOP_SUBRULES,
    "Slow": EXTENDED_SLOW_SUBRULES,
    "Fast": EXTENDED_FAST_SUBRULES,
}


# ====================================================================
# Language Parameters
# ====================================================================

NUM_MONADIC_PREDICATES = 11
NUM_DYADIC_PREDICATES = 6

# T = total binary slots per Q-sentence = 2·m + 2·d
TOTAL_PREDICATE_SLOTS = (
    2 * NUM_MONADIC_PREDICATES + 2 * NUM_DYADIC_PREDICATES
)  # 34

# log₂(Q) = T = 34,  so  Q = 2^34 total Q-sentences
Q_LOG2 = TOTAL_PREDICATE_SLOTS

# 4^(m+d) = 4^17 = 2^34  (each of the 17 predicates has 4 combos
# for the two arguments)
NUM_Q_SENTENCES = 4 ** (NUM_MONADIC_PREDICATES + NUM_DYADIC_PREDICATES)

# 2^Q attributive constituents, 2^(2^Q) constituents (symbolic only)
LOG2_NUM_ATTRIBUTIVE_CONSTITUENTS = NUM_Q_SENTENCES
LOG2_LOG2_TOTAL_CONSTITUENTS = NUM_Q_SENTENCES


# ====================================================================
# Q-Sentence Slot Mapping
# ====================================================================
#
# Convention:  x1 = ego,  x2 = entity
#
#   Slots  0 -- 10 : M(x1)      ego's monadic predicates
#   Slots 11 -- 21 : M(x2)      entity's monadic predicates
#   Slots 22 -- 27 : P(x1, x2)  dyadic, ego → entity
#   Slots 28 -- 33 : P(x2, x1)  dyadic, entity → ego
#

MONADIC_PREDICATES = [
    'IsCar', 'IsPedestrian', 'IsAmbulance', 'IsBus', 'IsPolice',
    'IsTiro', 'IsReckless', 'IsOld', 'IsYoung', 'IsAtInter', 'IsInInter',
]

DYADIC_PREDICATES = [
    'IsClose', 'CollidingClose', 'LeftOf', 'RightOf', 'NextTo', 'HigherPri',
]

PREDICATE_SLOT_MAP: Dict[Tuple[str, str], int] = {}

for _i, _pred in enumerate(MONADIC_PREDICATES):
    PREDICATE_SLOT_MAP[('monadic_x1', _pred)] = _i

for _i, _pred in enumerate(MONADIC_PREDICATES):
    PREDICATE_SLOT_MAP[('monadic_x2', _pred)] = NUM_MONADIC_PREDICATES + _i

for _i, _pred in enumerate(DYADIC_PREDICATES):
    PREDICATE_SLOT_MAP[('dyadic_x1_x2', _pred)] = (
        2 * NUM_MONADIC_PREDICATES + _i
    )

for _i, _pred in enumerate(DYADIC_PREDICATES):
    PREDICATE_SLOT_MAP[('dyadic_x2_x1', _pred)] = (
        2 * NUM_MONADIC_PREDICATES + NUM_DYADIC_PREDICATES + _i
    )


# ====================================================================
# Subrule → Hypothesis Conversion
# ====================================================================

def subrules_to_hypotheses(
    subrules: List[List[Tuple[str, str, bool]]],
) -> List[Hypothesis]:
    """
    Convert structured subrule specs into Hypothesis objects.

    Each subrule is a list of (slot_category, predicate_name, polarity)
    tuples.  This function maps each to a slot index via
    PREDICATE_SLOT_MAP and builds {slot_idx: polarity}.

    Parameters
    ----------
    subrules : list of SubruleSpec
        Each SubruleSpec is a list of (slot_category, pred_name, polarity).

    Returns
    -------
    list of Hypothesis
        One Dict[int, bool] per subrule.
    """
    hypotheses: List[Hypothesis] = []
    for subrule in subrules:
        h: Hypothesis = {}
        for slot_category, pred_name, polarity in subrule:
            key = (slot_category, pred_name)
            if key not in PREDICATE_SLOT_MAP:
                raise ValueError(
                    f"Unknown predicate slot: {key!r}. "
                    f"Check PREDICATE_SLOT_MAP for valid entries."
                )
            slot_idx = PREDICATE_SLOT_MAP[key]
            h[slot_idx] = polarity
        hypotheses.append(h)
    return hypotheses


ALL_SUBRULES: List[List[Tuple[str, str, bool]]] = (
    STOP_SUBRULES + SLOW_SUBRULES + FAST_SUBRULES
)
"""Flat list of all 12 expert subrules (Slow 3 excluded)."""

ALL_HYPOTHESES: List[Hypothesis] = subrules_to_hypotheses(ALL_SUBRULES)
"""Precomputed Hypothesis objects for all 12 expert subrules."""

EXTENDED_ALL_SUBRULES: List[List[Tuple[str, str, bool]]] = (
    EXTENDED_STOP_SUBRULES + EXTENDED_SLOW_SUBRULES + EXTENDED_FAST_SUBRULES
)
"""Flat list of all 48 extended subrules (Slow 3 from original still excluded)."""

EXTENDED_ALL_HYPOTHESES: List[Hypothesis] = subrules_to_hypotheses(
    EXTENDED_ALL_SUBRULES
)
"""Precomputed Hypothesis objects for all 48 extended subrules."""


# ---- Spatial 30 subrules (from expert_rule_spatial.yaml) ---------------------
#
# 10 Stop + 10 Slow + 10 Fast.  Trimmed from the original 63-rule spatial
# set to remove stop-rule redundancy (no bare CollidingClose or bare
# HigherPri+IsClose — each stop rule requires entity-type or directional
# specificity).  Ego-gate distribution: 57% no ego gate, 30% positional,
# 13% negated.

SPATIAL_STOP_SUBRULES: List[SubruleSpec] = [
    # S1: CollidingClose(ego, x) AND IsCar(x)
    [('dyadic_x1_x2', 'CollidingClose', True), ('monadic_x2', 'IsCar', True)],
    # S2: CollidingClose(ego, x) AND IsPedestrian(x)
    [('dyadic_x1_x2', 'CollidingClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # S3: IsClose(x, ego) AND IsPedestrian(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # S4: IsClose(x, ego) AND IsAmbulance(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsAmbulance', True)],
    # S5: RightOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsCar', True)],
    # S6: NextTo(x, ego) AND IsOld(x)
    [('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsOld', True)],
    # S7: LeftOf(x, ego) AND IsClose(x, ego) AND IsAmbulance(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsAmbulance', True)],
    # S8: IsAtInter(ego) AND IsInInter(x)   [positional ego gate]
    [('monadic_x1', 'IsAtInter', True), ('monadic_x2', 'IsInInter', True)],
    # S9: IsInInter(ego) AND HigherPri(x, ego) AND IsClose(x, ego)   [positional]
    [('monadic_x1', 'IsInInter', True), ('dyadic_x2_x1', 'HigherPri', True),
     ('dyadic_x2_x1', 'IsClose', True)],
    # S10: Not(IsAmbulance(ego)) AND IsClose(x, ego) AND IsAmbulance(x) AND IsAtInter(x)   [negated]
    [('monadic_x1', 'IsAmbulance', False), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsAmbulance', True), ('monadic_x2', 'IsAtInter', True)],
]

SPATIAL_SLOW_SUBRULES: List[SubruleSpec] = [
    # SL1: IsClose(x, ego) AND IsCar(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # SL2: IsClose(ego, x) AND IsPedestrian(x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # SL3: IsClose(x, ego) AND IsOld(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsOld', True)],
    # SL4: IsClose(x, ego) AND IsReckless(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsReckless', True)],
    # SL5: IsClose(ego, x) AND IsYoung(x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsYoung', True)],
    # SL6: LeftOf(x, ego) AND IsClose(x, ego) AND IsCar(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsCar', True)],
    # SL7: RightOf(x, ego) AND IsClose(x, ego) AND IsPedestrian(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsPedestrian', True)],
    # SL8: IsClose(ego, x) AND IsInInter(x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsInInter', True)],
    # SL9: IsAtInter(ego) AND IsClose(x, ego) AND IsCar(x)   [positional]
    [('monadic_x1', 'IsAtInter', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsCar', True)],
    # SL10: Not(IsAmbulance(ego)) AND NextTo(x, ego) AND IsAmbulance(x)   [negated]
    [('monadic_x1', 'IsAmbulance', False), ('dyadic_x2_x1', 'NextTo', True),
     ('monadic_x2', 'IsAmbulance', True)],
]

SPATIAL_FAST_SUBRULES: List[SubruleSpec] = [
    # F1: HigherPri(ego, x) AND IsClose(ego, x)
    [('dyadic_x1_x2', 'HigherPri', True), ('dyadic_x1_x2', 'IsClose', True)],
    # F2: HigherPri(ego, x) AND IsClose(ego, x) AND IsCar(x)
    [('dyadic_x1_x2', 'HigherPri', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsCar', True)],
    # F3: Not(IsAtInter(ego)) AND Not(IsInInter(ego)) AND HigherPri(ego, x) AND IsCar(x)   [positional]
    [('monadic_x1', 'IsAtInter', False), ('monadic_x1', 'IsInInter', False),
     ('dyadic_x1_x2', 'HigherPri', True), ('monadic_x2', 'IsCar', True)],
    # F4: IsAtInter(ego) AND HigherPri(ego, x) AND IsAtInter(x)   [positional]
    [('monadic_x1', 'IsAtInter', True), ('dyadic_x1_x2', 'HigherPri', True),
     ('monadic_x2', 'IsAtInter', True)],
    # F5: IsInInter(ego) AND HigherPri(ego, x) AND IsClose(ego, x)   [positional]
    [('monadic_x1', 'IsInInter', True), ('dyadic_x1_x2', 'HigherPri', True),
     ('dyadic_x1_x2', 'IsClose', True)],
    # F6: Not(IsAtInter(ego)) AND RightOf(x, ego) AND IsClose(ego, x) AND IsCar(x)   [positional]
    [('monadic_x1', 'IsAtInter', False), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # F7: Not(IsInInter(ego)) AND Not(IsAtInter(ego)) AND RightOf(x, ego) AND IsClose(x, ego) AND IsCar(x)   [positional]
    [('monadic_x1', 'IsInInter', False), ('monadic_x1', 'IsAtInter', False),
     ('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsCar', True)],
    # F8: Not(IsAtInter(ego)) AND LeftOf(x, ego) AND IsClose(ego, x) AND IsBus(x)   [positional]
    [('monadic_x1', 'IsAtInter', False), ('dyadic_x2_x1', 'LeftOf', True),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsBus', True)],
    # F9: Not(IsOld(ego)) AND HigherPri(ego, x) AND IsClose(ego, x) AND IsCar(x)   [negated]
    [('monadic_x1', 'IsOld', False), ('dyadic_x1_x2', 'HigherPri', True),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # F10: Not(IsOld(ego)) AND RightOf(x, ego) AND IsClose(ego, x)   [negated]
    [('monadic_x1', 'IsOld', False), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x1_x2', 'IsClose', True)],
]

SPATIAL_ALL_SUBRULES: List[List[Tuple[str, str, bool]]] = (
    SPATIAL_STOP_SUBRULES + SPATIAL_SLOW_SUBRULES + SPATIAL_FAST_SUBRULES
)
"""Flat list of all 30 spatial subrules."""

SPATIAL_ALL_HYPOTHESES: List[Hypothesis] = subrules_to_hypotheses(
    SPATIAL_ALL_SUBRULES
)
"""Precomputed Hypothesis objects for all 30 spatial subrules."""


# ---- Discriminative 30 subrules (from expert_rule_discriminative.yaml) --------
#
# 10 Stop + 10 Slow + 10 Fast.  Designed to maximise semantic vs. random
# differentiation by requiring distinct entity types per subrule.  Nearly
# all subrules are non-ego-gated; each combines a specific attributive
# predicate (IsAmbulance, IsBus, IsPolice, etc.) with spatial predicates.

DISCRIMINATIVE_STOP_SUBRULES: List[SubruleSpec] = [
    # S1: CollidingClose(ego, x) AND IsCar(x)
    [('dyadic_x1_x2', 'CollidingClose', True), ('monadic_x2', 'IsCar', True)],
    # S2: CollidingClose(ego, x) AND IsPedestrian(x)
    [('dyadic_x1_x2', 'CollidingClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # S3: IsClose(x, ego) AND IsAmbulance(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsAmbulance', True)],
    # S4: NextTo(x, ego) AND IsOld(x)
    [('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsOld', True)],
    # S5: RightOf(x, ego) AND IsClose(x, ego) AND IsPolice(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsPolice', True)],
    # S6: LeftOf(x, ego) AND NextTo(x, ego) AND IsPedestrian(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x2_x1', 'NextTo', True),
     ('monadic_x2', 'IsPedestrian', True)],
    # S7: RightOf(x, ego) AND IsClose(x, ego) AND IsReckless(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsReckless', True)],
    # S8: NextTo(x, ego) AND IsBus(x)
    [('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsBus', True)],
    # S9: LeftOf(x, ego) AND IsClose(x, ego) AND IsTiro(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsTiro', True)],
    # S10: IsClose(x, ego) AND IsYoung(x) AND RightOf(x, ego)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsYoung', True),
     ('dyadic_x2_x1', 'RightOf', True)],
]

DISCRIMINATIVE_SLOW_SUBRULES: List[SubruleSpec] = [
    # SL1: IsClose(x, ego) AND IsCar(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True)],
    # SL2: IsClose(ego, x) AND IsPedestrian(x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsPedestrian', True)],
    # SL3: RightOf(x, ego) AND IsClose(x, ego) AND IsAmbulance(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsAmbulance', True)],
    # SL4: LeftOf(x, ego) AND IsClose(x, ego) AND IsBus(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x2_x1', 'IsClose', True),
     ('monadic_x2', 'IsBus', True)],
    # SL5: NextTo(x, ego) AND IsPolice(x)
    [('dyadic_x2_x1', 'NextTo', True), ('monadic_x2', 'IsPolice', True)],
    # SL6: IsClose(x, ego) AND IsOld(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsOld', True)],
    # SL7: IsClose(x, ego) AND IsReckless(x)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsReckless', True)],
    # SL8: IsClose(ego, x) AND IsTiro(x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsTiro', True)],
    # SL9: LeftOf(x, ego) AND NextTo(x, ego) AND IsYoung(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x2_x1', 'NextTo', True),
     ('monadic_x2', 'IsYoung', True)],
    # SL10: IsClose(x, ego) AND IsCar(x) AND HigherPri(x, ego)
    [('dyadic_x2_x1', 'IsClose', True), ('monadic_x2', 'IsCar', True),
     ('dyadic_x2_x1', 'HigherPri', True)],
]

DISCRIMINATIVE_FAST_SUBRULES: List[SubruleSpec] = [
    # F1: HigherPri(ego, x) AND IsClose(ego, x) AND IsCar(x)
    [('dyadic_x1_x2', 'HigherPri', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsCar', True)],
    # F2: HigherPri(ego, x) AND NextTo(x, ego)
    [('dyadic_x1_x2', 'HigherPri', True), ('dyadic_x2_x1', 'NextTo', True)],
    # F3: RightOf(x, ego) AND IsClose(ego, x) AND IsTiro(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsTiro', True)],
    # F4: LeftOf(x, ego) AND IsClose(ego, x) AND IsReckless(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsReckless', True)],
    # F5: IsClose(ego, x) AND IsBus(x) AND HigherPri(ego, x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsBus', True),
     ('dyadic_x1_x2', 'HigherPri', True)],
    # F6: RightOf(x, ego) AND IsClose(ego, x) AND IsPedestrian(x)
    [('dyadic_x2_x1', 'RightOf', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsPedestrian', True)],
    # F7: Not(IsAtInter(ego)) AND Not(IsInInter(ego)) AND HigherPri(ego, x) AND IsCar(x)
    [('monadic_x1', 'IsAtInter', False), ('monadic_x1', 'IsInInter', False),
     ('dyadic_x1_x2', 'HigherPri', True), ('monadic_x2', 'IsCar', True)],
    # F8: IsClose(ego, x) AND IsPolice(x) AND HigherPri(ego, x)
    [('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsPolice', True),
     ('dyadic_x1_x2', 'HigherPri', True)],
    # F9: LeftOf(x, ego) AND IsClose(ego, x) AND IsOld(x)
    [('dyadic_x2_x1', 'LeftOf', True), ('dyadic_x1_x2', 'IsClose', True),
     ('monadic_x2', 'IsOld', True)],
    # F10: Not(IsAtInter(ego)) AND RightOf(x, ego) AND IsClose(ego, x) AND IsAmbulance(x)
    [('monadic_x1', 'IsAtInter', False), ('dyadic_x2_x1', 'RightOf', True),
     ('dyadic_x1_x2', 'IsClose', True), ('monadic_x2', 'IsAmbulance', True)],
]

DISCRIMINATIVE_ALL_SUBRULES: List[List[Tuple[str, str, bool]]] = (
    DISCRIMINATIVE_STOP_SUBRULES + DISCRIMINATIVE_SLOW_SUBRULES
    + DISCRIMINATIVE_FAST_SUBRULES
)
"""Flat list of all 30 discriminative subrules."""

DISCRIMINATIVE_ALL_HYPOTHESES: List[Hypothesis] = subrules_to_hypotheses(
    DISCRIMINATIVE_ALL_SUBRULES
)
"""Precomputed Hypothesis objects for all 30 discriminative subrules."""


def get_hypotheses_for_rule_file(rule_yaml_path: str) -> List[Hypothesis]:
    """Return the appropriate hypothesis list based on the rule YAML filename."""
    if 'discriminative' in rule_yaml_path:
        return DISCRIMINATIVE_ALL_HYPOTHESES
    if 'spatial' in rule_yaml_path:
        return SPATIAL_ALL_HYPOTHESES
    if 'extended' in rule_yaml_path:
        return EXTENDED_ALL_HYPOTHESES
    return ALL_HYPOTHESES


# ====================================================================
# Conversion: GNA Groundings → Q-Sentence
# ====================================================================

def groundings_to_q_sentence(groundings: dict) -> QSentence:
    """
    Convert one entity's GNA grounding dict into a complete Q-sentence.

    The grounding dict (produced by ``gna.ground_predicates_for_vicinity_entity``)
    has four keys, each mapping predicate names to bool values:

        'unary_ego'          → ego's monadic predicates   → slots  0..10  M(x1)
        'unary_entity'       → entity's monadic predicates→ slots 11..21  M(x2)
        'binary_ego_entity'  → dyadic ego→entity          → slots 22..27  P(x1,x2)
        'binary_entity_ego'  → dyadic entity→ego          → slots 28..33  P(x2,x1)

    Returns
    -------
    QSentence
        A 34-element tuple of bools uniquely identifying one Q-sentence.
    """
    slots = [False] * TOTAL_PREDICATE_SLOTS

    _CATEGORY_MAP = {
        'unary_ego':         'monadic_x1',
        'unary_entity':      'monadic_x2',
        'binary_ego_entity': 'dyadic_x1_x2',
        'binary_entity_ego': 'dyadic_x2_x1',
    }

    for grounding_key, slot_category in _CATEGORY_MAP.items():
        for pred_name, value in groundings.get(grounding_key, {}).items():
            map_key = (slot_category, pred_name)
            if map_key in PREDICATE_SLOT_MAP:
                slots[PREDICATE_SLOT_MAP[map_key]] = bool(value)

    return tuple(slots)


# ====================================================================
# Overlap Check (internal helper)
# ====================================================================

def _check_overlap(
    evidence_q_sentences: Set[QSentence],
    hypothesis: Hypothesis,
) -> bool:
    """
    Check whether any evidence Q-sentence already *satisfies* the
    hypothesis — i.e., all predicate slots asserted by the hypothesis
    match the Q-sentence's truth values.

    If overlap exists, the hypothesis is realized by the evidence and
    p(h|e) = 1.
    """
    for qs in evidence_q_sentences:
        if all(qs[slot] == val for slot, val in hypothesis.items()):
            return True
    return False


# ====================================================================
# Constituent Counting
# ====================================================================

def count_compatible_constituents(
    K: int,
    Q_log2: int = Q_LOG2,
) -> int:
    """
    Count constituents compatible with evidence consisting of K distinct
    Q-sentences, using exclusion-based (complementary) counting.

    Mathematical derivation
    -----------------------
    Let Q = 2^{Q_log2} (total Q-sentences).

    The K observed Q-sentences are fixed as existing.  An attributive
    constituent (AC) is *compatible* iff it includes all K of them.

        Compatible ACs   = 2^{Q − K}      (K are fixed, rest free)
        Incompatible ACs = 2^Q − 2^{Q−K}

    A constituent is compatible iff it asserts at least one compatible
    AC.  Excluded constituents (subsets of only incompatible ACs):

        Excluded = 2^{incompatible ACs} = 2^{2^Q − 2^{Q−K}}

    Compatible constituents:

        |{C : C ⊨ e}| = 2^{2^Q} − 2^{2^Q − 2^{Q−K}}

    Parameters
    ----------
    K : int
        Number of distinct Q-sentences observed.

    Returns
    -------
    int
        K itself (the symbolic seed).  The actual count is
        2^{2^Q} − 2^{2^Q − 2^{Q−K}}, which is too large to store
        as an integer for Q_log2 = 34.  Downstream functions use K
        to compute probabilities via the formula above.
    """
    return K


def count_compatible_with_conjunction(
    evidence_q_sentences: Set[QSentence],
    hypothesis: Hypothesis,
    Q_log2: int = Q_LOG2,
) -> dict:
    """
    Count constituents compatible with evidence AND hypothesis jointly.

    This is used to compute  p(h|e) = |{C : C ⊨ e ∧ h}| / |{C : C ⊨ e}|.

    Hypothesis representation
    -------------------------
    A hypothesis fixes Z predicate slots (out of 34) to specific truth
    values.  This defines  H = 2^{34−Z}  hypothesis-compatible
    Q-sentences (those matching the Z asserted slots; the remaining
    34−Z slots are free).

    Step 1 — Overlap check
    ----------------------
    If any of the K evidence Q-sentences already matches all Z
    hypothesis slots, the hypothesis is realized by the evidence:

        p(h|e) = 1    →  return {'overlap': True, ...}

    Step 2 — No-overlap exclusion counting
    ---------------------------------------
    Let Q = 2^{Q_log2},  K = |evidence|,  H = 2^{34−Z}.

    A "good" AC must include all K evidence Q-sentences AND at least
    one of the H hypothesis-compatible Q-sentences.  Since none of
    the K evidence Q-sentences is among the H (no overlap), these
    are disjoint requirements:

        Good ACs = 2^{Q−K} − 2^{Q−K−H}

        (first term: ACs including all K;
         subtract: ACs including all K but zero hypothesis Q-sentences,
         i.e., subsets of the Q−K−H Q-sentences that are neither
         evidence nor hypothesis-compatible)

    Bad ACs (NOT good):

        Bad = 2^Q − Good = 2^Q − 2^{Q−K} + 2^{Q−K−H}

    Excluded constituents (subsets of only bad ACs):

        Excluded_{e∧h} = 2^{Bad}

    Compatible constituents for (e ∧ h):

        C_{e∧h} = 2^{2^Q} − 2^{Bad}

    Returns
    -------
    dict
        {'overlap': bool, 'K': int, 'Z': int, 'H_log2': int, 'Q_log2': int}
        Symbolic parameters for downstream probability computation.
    """
    K = len(evidence_q_sentences)
    Z = len(hypothesis)
    overlap = _check_overlap(evidence_q_sentences, hypothesis)

    return {
        'overlap': overlap,
        'K': K,
        'Z': Z,
        'H_log2': TOTAL_PREDICATE_SLOTS - Z,
        'Q_log2': Q_log2,
    }


# ====================================================================
# Probability Functions
# ====================================================================

def compute_evidence_probability(
    K: int,
    Q_log2: int = Q_LOG2,
) -> float:
    """
    Compute the evidence probability  p(e)  for K distinct Q-sentences.

    Formula
    -------
        p(e) = |{C : C ⊨ e}| / |C|
             = [2^{2^Q} − 2^{2^Q − 2^{Q−K}}] / 2^{2^Q}
             = 1 − 2^{−2^{Q−K}}

    where Q = 2^{Q_log2}.

    Numerical notes
    ---------------
    For Q_log2 = 34 and any K ≥ 1, the inner exponent 2^{Q−K} is
    astronomically large (a number with ~5 billion digits), making
    2^{−2^{Q−K}} ≈ 0 and p(e) ≈ 1.  Float64 cannot distinguish
    different K values.

    For small Q_log2 (toy testing), the computation is exact.

    Comparison across evidence subsets should use K directly:
    larger K → lower p(e).
    """
    Q = 1 << Q_log2
    inner_exp = Q - K

    if inner_exp <= 0:
        return 0.0

    # 2^{inner_exp} can overflow; guard against it
    try:
        if inner_exp > 1023:
            return 1.0
        power = 2.0 ** inner_exp
        if power > 1023:
            return 1.0
        return 1.0 - 2.0 ** (-power)
    except OverflowError:
        return 1.0


def compute_conditional_probability(
    evidence_q_sentences: Set[QSentence],
    hypothesis: Hypothesis,
    Q_log2: int = Q_LOG2,
) -> float:
    """
    Compute the conditional probability  p(h | e).

    Formula
    -------
        p(h|e) = |{C : C ⊨ e ∧ h}| / |{C : C ⊨ e}|

    Case 1 — Overlap (an evidence Q-sentence satisfies the hypothesis):

        p(h|e) = 1.0

    Case 2 — No overlap:

        Let  K = |evidence|,  Z = |hypothesis slots|,  H = 2^{34−Z},
             Q = 2^{Q_log2}.

        Define:
            α = 2^{Q−K}                      (compatible ACs for evidence)
            β = 2^{Q−K−H}                    (ACs with all K but no hypothesis Q-sentence)
            γ = α − β = 2^{Q−K−H} · (2^H − 1)

        Then:
            p(h|e) = (1 − 2^{−γ}) / (1 − 2^{−α})

        For Q_log2 = 34, both α and γ are astronomically large,
        making p(h|e) ≈ 1.

    Parameters
    ----------
    evidence_q_sentences : set of QSentence
        The K distinct Q-sentences observed.
    hypothesis : Hypothesis
        Partial slot assignment {slot_idx: required_value}.

    Returns
    -------
    float
        The conditional probability p(h|e).
    """
    if _check_overlap(evidence_q_sentences, hypothesis):
        return 1.0

    K = len(evidence_q_sentences)
    Z = len(hypothesis)
    Q = 1 << Q_log2
    H = 1 << (TOTAL_PREDICATE_SLOTS - Z)

    alpha = Q - K
    q_k_h = Q - K - H

    if q_k_h < 0:
        # More hypothesis Q-sentences than remaining non-evidence
        # Q-sentences.  gamma ≤ 0; treat as degenerate.
        return 1.0

    # gamma = 2^{q_k_h} · (2^H − 1)
    # Both q_k_h and H can be enormous; guard float computation.
    try:
        if alpha > 1023:
            return 1.0

        denom = 1.0 - 2.0 ** (-(2.0 ** alpha))

        if q_k_h > 1023:
            return 1.0

        gamma = (2.0 ** q_k_h) * ((2.0 ** H) - 1.0)

        if gamma > 1023:
            numer = 1.0
        else:
            numer = 1.0 - 2.0 ** (-gamma)

        if denom == 0.0:
            return 0.0

        return numer / denom
    except OverflowError:
        return 1.0


# ====================================================================
# Objective Functions
# ====================================================================

def compute_objective_simplified(
    evidence_q_sentences: Set[QSentence],
    hypotheses: List[Hypothesis],
    Q_log2: int = Q_LOG2,
) -> float:
    """
    Compute the semantic communication objective using the algebraically
    simplified per-hypothesis term.

    Objective
    ---------
        obj = Σ_i  p(e) · p(h_i|e) · (1 − p(h_i|e))

    Algebraic simplification
    ------------------------
    Substituting and cancelling one (1 − 2^{−α}) factor:

        obj_i  =  (1 − u_i)(u_i − v) / (1 − v)

    where:
        α   = 2^{Q−K}                               (evidence-only parameter)
        β_i = 2^{Q−K−H_i},   H_i = 2^{34−Z_i}      (per-hypothesis)
        γ_i = α − β_i = 2^{Q−K−H_i} · (2^{H_i} − 1)
        u_i = 2^{−γ_i}
        v   = 2^{−α}

    with 0 < v < u_i < 1  (since γ_i < α).

    Overlap case: p(h_i|e) = 1  →  obj_i = 0.

    Approximation for large Q−K:
        u_i, v ≈ 0  →  obj_i ≈ u_i = 2^{−γ_i}.
        Comparison reduces to γ_i values (larger γ = smaller objective).

    Parameters
    ----------
    evidence_q_sentences : set of QSentence
    hypotheses : list of Hypothesis
    Q_log2 : int

    Returns
    -------
    float
        Total objective value (lower is better).
    """
    K = len(evidence_q_sentences)
    Q = 1 << Q_log2
    alpha = Q - K

    try:
        v = 2.0 ** (-alpha) if alpha <= 1023 else 0.0
    except OverflowError:
        v = 0.0

    one_minus_v = 1.0 - v

    objective = 0.0

    for hypothesis in hypotheses:
        if _check_overlap(evidence_q_sentences, hypothesis):
            continue

        Z = len(hypothesis)
        H = 1 << (TOTAL_PREDICATE_SLOTS - Z)
        q_k_h = Q - K - H

        if q_k_h < 0:
            continue

        try:
            # γ_i = 2^{q_k_h} · (2^H − 1)
            if q_k_h > 1023 or H > 1023:
                continue  # u_i = 2^{-γ} underflows to 0

            gamma = (2.0 ** q_k_h) * ((2.0 ** H) - 1.0)

            if gamma > 1023:
                continue

            u = 2.0 ** (-gamma)

            if one_minus_v == 0.0:
                continue

            objective += (1.0 - u) * (u - v) / one_minus_v

        except OverflowError:
            continue

    return objective


def compute_objective_direct(
    evidence_q_sentences: Set[QSentence],
    hypotheses: List[Hypothesis],
    Q_log2: int = Q_LOG2,
) -> float:
    """
    Compute the semantic communication objective directly from the
    probability definitions (no algebraic simplification).

    Objective
    ---------
        obj = Σ_i  p(e) · p(h_i|e) · (1 − p(h_i|e))

    This function computes p(e) and each p(h_i|e) separately, then
    multiplies.  It should produce identical results to
    ``compute_objective_simplified`` but mirrors the mathematical
    definitions more transparently for verification.

    p(e) computation
    ----------------
        Bad ACs for evidence:   B_e = 2^Q − 2^{Q−K}
        Excluded constituents:  2^{B_e}
        Compatible for e:       C_e = 2^{2^Q} − 2^{B_e}
        Total constituents:     C_{total} = 2^{2^Q}

        p(e) = C_e / C_{total} = 1 − 2^{−2^{Q−K}}

    p(h_i|e) computation (no-overlap case)
    ---------------------------------------
        Bad ACs for (e ∧ h):    B_{eh} = B_e + 2^{Q−K−H_i}
        Compatible for (e ∧ h): C_{eh} = 2^{2^Q} − 2^{B_{eh}}

        p(h_i|e) = C_{eh} / C_e

    Parameters
    ----------
    evidence_q_sentences : set of QSentence
    hypotheses : list of Hypothesis
    Q_log2 : int

    Returns
    -------
    float
        Total objective value (lower is better).
    """
    K = len(evidence_q_sentences)
    p_e = compute_evidence_probability(K, Q_log2)

    objective = 0.0

    for hypothesis in hypotheses:
        p_h_e = compute_conditional_probability(
            evidence_q_sentences, hypothesis, Q_log2,
        )
        objective += p_e * p_h_e * (1.0 - p_h_e)

    return objective


# ====================================================================
# Symbolic Objective Key
# ====================================================================
#
# For Q_log2 = 34, all float objective values underflow to 0.
# This function produces a *comparable* tuple so that
# select_optimal_subset can still distinguish subsets.
#
# Mathematical derivation
# -----------------------
# For non-overlapping hypothesis i, the objective term is:
#
#     obj_i ≈ 2^{−γ_i}
#
# where γ_i = 2^{Q−K} − 2^{Q−K−H_i} and H_i = 2^{34−Z_i}.
#
# The total objective (sum over non-overlapping hypotheses) is
# dominated by the term with the *smallest* γ_i.  Therefore
# we want to *maximise* γ_min = min_i γ_i.
#
# Since γ_i = 2^{Q−K} − 2^{Q−K−H_i}, the comparison decomposes
# into two independent effects:
#
#   1. K  (number of distinct evidence Q-sentences):
#      Smaller K  →  larger 2^{Q−K}  →  larger all γ_i  →  better.
#      This effect is *doubly exponential* and dominates everything.
#
#   2. H_min (smallest H among non-overlapping hypotheses):
#      Larger H_min  →  smaller 2^{Q−K−H_min}  →  larger γ_min  →  better.
#      This is the tiebreaker when K is equal.
#
# These two quantities must be separated in the key because they
# have *opposite* relationships with the old proxy (Q−K−H): an
# increase caused by smaller K is good, while an increase caused by
# smaller H is bad.
#
# Ordering logic (minimise the key lexicographically):
#   1. n_non_overlap  — fewer non-overlapping hypotheses is better
#      (overlapping terms contribute exactly 0).
#   2. K  — fewer distinct evidence Q-sentences is better.
#   3. −H values sorted ascending  — among non-overlapping hypotheses,
#      larger H_min is better (its γ is larger, shrinking the
#      dominant objective term).  Negation + minimisation achieves
#      this: minimising −H_min means maximising H_min.
#

def _compute_objective_key(
    evidence_q_sentences: Set[QSentence],
    hypotheses: List[Hypothesis],
    Q_log2: int = Q_LOG2,
) -> Tuple:
    """
    Produce a lexicographically comparable tuple for subset ordering
    when float objective values underflow to 0.

    The key faithfully captures the mathematical objective by
    separating the two independent effects (K and H) that determine
    the dominant objective term.

    Returns
    -------
    tuple
        (n_non_overlap, K, -H_sorted_asc[0], -H_sorted_asc[1], ...)

        Minimising this tuple lexicographically selects the subset
        with the smallest objective value:
          • fewer non-overlapping hypotheses,
          • then fewer distinct Q-sentences (smaller K),
          • then larger H_min among non-overlapping hypotheses.
    """
    K = len(evidence_q_sentences)

    H_values = []
    for hypothesis in hypotheses:
        if _check_overlap(evidence_q_sentences, hypothesis):
            continue
        Z = len(hypothesis)
        H = 1 << (TOTAL_PREDICATE_SLOTS - Z)
        H_values.append(H)

    n_non_overlap = len(H_values)
    H_values.sort()
    negated_H = tuple(-h for h in H_values)

    return (n_non_overlap, K) + negated_H


# ====================================================================
# Subset Selection
# ====================================================================

def select_optimal_subset(
    observations: List[dict],
    hypotheses: List[Hypothesis],
    k: int,
    Q_log2: int = Q_LOG2,
) -> Tuple[List[dict], float]:
    """
    Select the size-k subset of observations that minimizes the
    semantic communication objective.

    Algorithm
    ---------
    Enumerates all C(N, k) subsets of observations.  For each subset:

        1. Collect the Q-sentences of the k selected entities into a set.
        2. K_sub = |distinct Q-sentences in subset|.
        3. Score the subset via ``compute_objective_simplified``.
        4. Also compute the symbolic comparison key via
           ``_compute_objective_key`` to handle float underflow.
        5. Track the subset with the lowest key (lexicographic).

    Complexity: O(C(N, k) · M · K) where M = number of hypotheses.
    For large N this is exponential; a greedy approximation may be
    needed in the future.

    Parameters
    ----------
    observations : list of dict
        N observation dicts, each containing at least a 'q_sentence'
        key with the 34-bool QSentence tuple.
    hypotheses : list of Hypothesis
        Traffic-rule hypothesis expressions (Dict[int, bool]).
    k : int
        Fixed size of the subset to transmit.

    Returns
    -------
    (best_subset, best_score)
        best_subset : list of observation dicts
        best_score  : float objective value (may be 0.0 due to underflow)
    """
    n = len(observations)

    if n <= k:
        qs_set = frozenset(obs['q_sentence'] for obs in observations)
        score = compute_objective_simplified(qs_set, hypotheses, Q_log2)
        return list(observations), score

    best_key: Optional[Tuple] = None
    best_score = float('inf')
    best_subset: Optional[List[dict]] = None

    for indices in combinations(range(n), k):
        subset = [observations[i] for i in indices]
        qs_set = frozenset(obs['q_sentence'] for obs in subset)

        score = compute_objective_simplified(qs_set, hypotheses, Q_log2)
        key = _compute_objective_key(qs_set, hypotheses, Q_log2)

        if best_key is None or key < best_key:
            best_key = key
            best_score = score
            best_subset = subset

    return best_subset, best_score
