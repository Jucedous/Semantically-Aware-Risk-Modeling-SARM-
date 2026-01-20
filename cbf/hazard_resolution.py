from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .cbf_safety_metrics import ObjectState

try:
    from .graphs.feedback_graph import FeedbackGraph, FeedbackEdge
except Exception:
    FeedbackGraph = object 
    FeedbackEdge = object 

HazardRule = Tuple[str, str, float, float, str, str]


class HazardStatus(str, Enum):
    """How a concrete object–object hazard ended up in or out of the final set."""

    LLM_ONLY = "llm_only"

    RULE_CONFIRMED = "rule_confirmed"

    SUPPRESSED_BY_RULE = "suppressed_by_rule_not_dangerous"

    RULE_ONLY = "rule_only"


@dataclass
class RuleGraphDecision:
    """Summary of what the rule graph thinks about a concrete object pair.

    Attributes
    ----------
    label:
        "dangerous", "not_dangerous", or None if the rule graph has no strong
        opinion about this pair.
    confidence:
        A number in [0, 1] indicating how strong the signal is (maximum
        similarity-weighted support from any base rule).
    sources:
        Concept pairs (c1, c2) in the rule graph that contributed the maximum
        support. 
    """

    label: Optional[str]
    confidence: float
    sources: List[Tuple[str, str]]

@dataclass
class RuleGraphConflict:
    """Concept-level conflict between rule-graph edges.

    This describes a situation where the feedback graph contains one
    edge that says (a1, b1) is 'dangerous' and another that says
    (a2, b2) is 'not_dangerous', and the two concept pairs are highly
    aligned according to the similarity graph.
    """

    a1: str
    b1: str
    label1: str
    a2: str
    b2: str
    label2: str
    similarity: float


@dataclass
class HazardMetadata:
    """Metadata for a concrete object–object hazard or potential hazard.

    status:
        How this pair ended up being treated (LLM_ONLY, RULE_CONFIRMED, etc.).
    rule_decision:
        Underlying RuleGraphDecision that led to this status.
    """

    status: HazardStatus
    rule_decision: RuleGraphDecision


@dataclass
class ResolvedHazards:
    """Output of resolve_semantic_hazards.

    Attributes
    ----------
    rules:
        List of hazard rules that should be treated as *active* downstream
        (metrics, semantic hazard graph, explanations, etc.).
    metadata:
        Mapping from unordered (name_a, name_b) pairs to HazardMetadata.
        Pairs that have no hazard at all in the final set may still appear
        here if the rule graph had an opinion about them, which lets the UI
        show suppressed hazards in a debug view if desired.
    conflicts:
        List of concept-level conflicts between rule-graph edges. This does
        not change behavior by itself but can be surfaced in the UI to help
        users debug inconsistent feedback.
    """

    rules: List[HazardRule]
    metadata: Dict[Tuple[str, str], HazardMetadata]
    conflicts: List[RuleGraphConflict] = field(default_factory=list)


def _normalize_pair_key(a: str, b: str) -> Tuple[str, str]:
    """Return a canonical, order-independent key for an object pair."""
    a = str(a)
    b = str(b)
    return (a, b) if a <= b else (b, a)


def _canonical(text: str) -> str:
    """Canonicalize concept strings in the same way as FeedbackGraph."""
    return " ".join(str(text or "").strip().lower().split())


def _get_feedback_edge(
    fg: FeedbackGraph, a: str, b: str
) -> Optional[FeedbackEdge]:
    """Best-effort access to an undirected feedback edge (a, b)."""
    key = getattr(fg, "_key", None)
    if callable(key):
        k = key(a, b)
        return fg.edges.get(k)
    return fg.edges.get((a, b)) or fg.edges.get((b, a))


def _edge_has_kind(edge: FeedbackEdge, kind: str) -> bool:
    kinds = getattr(edge, "kinds", None)
    if isinstance(kinds, set):
        return kind in kinds
    if isinstance(kinds, (list, tuple)):
        return kind in kinds
    return False


def _edge_label(edge: FeedbackEdge) -> Optional[str]:
    """Return 'dangerous' or 'not_dangerous' for a feedback edge.

    We conservatively prefer 'not_dangerous' when both are present.
    """
    if _edge_has_kind(edge, "not_dangerous"):
        return "not_dangerous"
    if _edge_has_kind(edge, "dangerous"):
        return "dangerous"
    return None


def _similarity(fg: FeedbackGraph, c1: str, c2: str) -> float:
    """Symmetric similarity between two concept nodes in the rule graph.

    Returns 1.0 for exact matches, otherwise looks for an edge whose kinds
    include "similar" and uses edge.similarity_score if present; else 0.0.
    """
    c1 = _canonical(c1)
    c2 = _canonical(c2)
    if c1 == c2:
        return 1.0
    edge = _get_feedback_edge(fg, c1, c2)
    if edge is None:
        return 0.0
    if not _edge_has_kind(edge, "similar"):
        return 0.0
    score = getattr(edge, "similarity_score", None)
    if isinstance(score, (float, int)):
        # clamp into [0, 1] just to be safe
        val = float(score)
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return val
    return 0.0


def _concepts_for_object(obj: ObjectState, fg: Optional[FeedbackGraph]) -> Set[str]:
    """Map a concrete object to concept nodes in the rule graph.

    We currently use:
      - the object's kind,
      - the object's name.

    Both are canonicalized; only those concepts that actually appear in the
    feedback graph are returned.
    """
    candidates: Set[str] = set()
    kind = getattr(obj, "kind", None)
    name = getattr(obj, "name", None)

    if kind:
        candidates.add(_canonical(kind))
    if name:
        candidates.add(_canonical(name))

    if fg is None:
        return candidates

    nodes = getattr(fg, "nodes", None)
    if not isinstance(nodes, set):
        return set()

    return {c for c in candidates if c in nodes}


def _iter_base_rule_edges(fg: FeedbackGraph) -> Iterable[FeedbackEdge]:
    """Yield feedback edges that carry 'dangerous' or 'not_dangerous' kinds."""
    edges = getattr(fg, "edges", {})
    for edge in getattr(edges, "values", lambda: [])():
        if _edge_has_kind(edge, "dangerous") or _edge_has_kind(
            edge, "not_dangerous"
        ):
            yield edge

def detect_rule_graph_conflicts(
    fg: Optional[FeedbackGraph],
    *,
    similarity_threshold: float = 0.85,
) -> List[RuleGraphConflict]:
    """Find pairs of rule edges that strongly disagree.

    A conflict is:
      - one 'dangerous' edge and one 'not_dangerous' edge, and
      - their concept pairs are highly aligned under the same similarity
        notion used for object pairs.

    Example that will be flagged:
      - laptop --dangerous--> water source
      - laptop --not_dangerous--> liquid
      - liquid ~ water source with high similarity.
    """
    if fg is None:
        return []

    labeled_edges: List[Tuple[str, FeedbackEdge]] = []

    edges = getattr(fg, "edges", {})
    values = getattr(edges, "values", None)
    if not callable(values):
        return []

    for edge in values():
        has_danger = _edge_has_kind(edge, "dangerous")
        has_not = _edge_has_kind(edge, "not_dangerous")

        if has_danger:
            labeled_edges.append(("dangerous", edge))
        if has_not:
            labeled_edges.append(("not_dangerous", edge))

    conflicts: List[RuleGraphConflict] = []
    n = len(labeled_edges)

    for i in range(n):
        label1, e1 = labeled_edges[i]
        a1 = _canonical(getattr(e1, "a", "") or "")
        b1 = _canonical(getattr(e1, "b", "") or "")
        if not a1 or not b1:
            continue

        for j in range(i + 1, n):
            label2, e2 = labeled_edges[j]
            if label1 == label2:
                continue  # only care about dangerous vs not_dangerous

            a2 = _canonical(getattr(e2, "a", "") or "")
            b2 = _canonical(getattr(e2, "b", "") or "")
            if not a2 or not b2:
                continue

            s1 = _similarity(fg, a1, a2) * _similarity(fg, b1, b2)
            s2 = _similarity(fg, a1, b2) * _similarity(fg, b1, a2)
            sim = max(s1, s2)

            if sim >= similarity_threshold:
                conflicts.append(
                    RuleGraphConflict(
                        a1=a1,
                        b1=b1,
                        label1=label1,
                        a2=a2,
                        b2=b2,
                        label2=label2,
                        similarity=sim,
                    )
                )

    return conflicts


def _classify_pair_with_rule_graph(
    obj_a: ObjectState,
    obj_b: ObjectState,
    fg: Optional[FeedbackGraph],
    base_edges: Optional[Sequence[FeedbackEdge]] = None,
    strong_not_dangerous: float = 0.85,
    strong_dangerous: float = 0.75,
) -> RuleGraphDecision:
    """Infer what the rule graph thinks about this concrete pair of objects.

    Logic (conservative):

    - Take all concept-level hazard edges ("dangerous"/"not_dangerous") in the
      feedback graph.
    - For each such edge (x, y), compute how well (obj_a, obj_b) can be
      matched to (x, y) or (y, x) using identity + similarity edges.
    - The maximum similarity-weighted support for 'dangerous' and
      'not_dangerous' are compared against thresholds.

    Parameters
    ----------
    strong_not_dangerous:
        Threshold in [0,1]; if max support for 'not_dangerous' exceeds this
        and is >= the dangerous support, the pair is treated as not dangerous.
    strong_dangerous:
        Analogous threshold for 'dangerous'.

    Returns
    -------
    RuleGraphDecision
        label:
            "not_dangerous" | "dangerous" | None
        confidence:
            max support value (0–1).
        sources:
            Which concept pairs contributed that support.
    """
    if fg is None:
        return RuleGraphDecision(label=None, confidence=0.0, sources=[])

    concepts_a = _concepts_for_object(obj_a, fg)
    concepts_b = _concepts_for_object(obj_b, fg)
    if not concepts_a or not concepts_b:
        return RuleGraphDecision(label=None, confidence=0.0, sources=[])

    if base_edges is None:
        base_edges = list(_iter_base_rule_edges(fg))

    max_not = 0.0
    max_danger = 0.0
    sources_not: List[Tuple[str, str]] = []
    sources_danger: List[Tuple[str, str]] = []

    for edge in base_edges:
        label = _edge_label(edge)
        if label is None:
            continue
        a = getattr(edge, "a", None)
        b = getattr(edge, "b", None)
        if a is None or b is None:
            continue
        a = _canonical(a)
        b = _canonical(b)

        for ca in concepts_a:
            for cb in concepts_b:
                w1 = _similarity(fg, ca, a) * _similarity(fg, cb, b)
                w2 = _similarity(fg, ca, b) * _similarity(fg, cb, a)
                w = max(w1, w2)
                if w <= 0.0:
                    continue
                if label == "dangerous":
                    if w > max_danger:
                        max_danger = w
                        sources_danger = [(a, b)]
                    elif w == max_danger:
                        sources_danger.append((a, b))
                else:
                    if w > max_not:
                        max_not = w
                        sources_not = [(a, b)]
                    elif w == max_not:
                        sources_not.append((a, b))

    if max_not >= strong_not_dangerous and max_not >= max_danger:
        return RuleGraphDecision(
            label="not_dangerous", confidence=max_not, sources=sources_not
        )
    if max_danger >= strong_dangerous and max_danger > max_not:
        return RuleGraphDecision(
            label="dangerous", confidence=max_danger, sources=sources_danger
        )

    if max_danger == 0.0 and max_not == 0.0:
        return RuleGraphDecision(label=None, confidence=0.0, sources=[])
    if max_danger >= max_not:
        sources = sources_danger
        confidence = max_danger
    else:
        sources = sources_not
        confidence = max_not
    return RuleGraphDecision(label=None, confidence=confidence, sources=sources)


def resolve_semantic_hazards(
    objects: Sequence[ObjectState],
    semantic_rules: Sequence[HazardRule],
    feedback_graph: Optional[FeedbackGraph],
    *,
    strong_not_dangerous: float = 0.85,
    strong_dangerous: float = 0.75,
    create_rule_only_hazards: bool = True,
    default_soft_clearance_m: float = 0.3,
    default_weight: float = 1.0,
) -> ResolvedHazards:
    """Combine LLM semantic hazards with the rule graph.

    Parameters
    ----------
    objects:
        Concrete objects in the current scene.
    semantic_rules:
        Hazards produced by the LLM semantics pipeline and user preference
        enforcement (i.e. after enforce_user_preferences_on_instantiated_rules).
    feedback_graph:
        Rule graph built from user feedback, including similarity edges.
        If None, this function is a no-op that simply returns the input rules.
    strong_not_dangerous, strong_dangerous:
        Thresholds in [0,1] controlling how strong a rule-graph signal must be
        to override or confirm an LLM hazard. Typically you want
        strong_not_dangerous >= strong_dangerous so that "safe" overrides are
        more conservative.
    create_rule_only_hazards:
        If True, pairs that the rule graph marks as dangerous but for which
        the LLM did not propose a hazard will get a synthetic hazard rule
        (status RULE_ONLY). If False, such pairs will simply be ignored.
    default_soft_clearance_m, default_weight:
        Defaults used when creating RULE_ONLY hazards.

    Returns
    -------
    ResolvedHazards
        See dataclass docstring.
    """
    if feedback_graph is None:
        return ResolvedHazards(rules=list(semantic_rules), metadata={}, conflicts=[])

    name_to_obj: Dict[str, ObjectState] = {str(o.name): o for o in objects}
    obj_names = sorted(name_to_obj.keys())

    rule_by_pair: Dict[Tuple[str, str], HazardRule] = {}
    for kind_a, kind_b, clearance, weight, name_a, name_b in semantic_rules:
        key = _normalize_pair_key(str(name_a), str(name_b))
        rule_by_pair[key] = (
            str(kind_a),
            str(kind_b),
            float(clearance),
            float(weight),
            str(name_a),
            str(name_b),
        )

    base_edges = list(_iter_base_rule_edges(feedback_graph))

    final_rules: List[HazardRule] = []
    metadata: Dict[Tuple[str, str], HazardMetadata] = {}

    n = len(obj_names)
    for i in range(n):
        for j in range(i + 1, n):
            name_a = obj_names[i]
            name_b = obj_names[j]
            key = _normalize_pair_key(name_a, name_b)

            obj_a = name_to_obj[name_a]
            obj_b = name_to_obj[name_b]

            decision = _classify_pair_with_rule_graph(
                obj_a,
                obj_b,
                feedback_graph,
                base_edges=base_edges,
                strong_not_dangerous=strong_not_dangerous,
                strong_dangerous=strong_dangerous,
            )

            semantic_rule = rule_by_pair.get(key)
            status: Optional[HazardStatus] = None

            if semantic_rule is not None:
                if decision.label == "not_dangerous":
                    status = HazardStatus.SUPPRESSED_BY_RULE
                else:
                    if decision.label == "dangerous":
                        status = HazardStatus.RULE_CONFIRMED
                    else:
                        status = HazardStatus.LLM_ONLY
                    final_rules.append(semantic_rule)
            else:
                if decision.label == "dangerous" and create_rule_only_hazards:
                    kind_a = getattr(obj_a, "kind", "object") or "object"
                    kind_b = getattr(obj_b, "kind", "object") or "object"
                    new_rule: HazardRule = (
                        str(kind_a),
                        str(kind_b),
                        float(default_soft_clearance_m),
                        float(default_weight),
                        name_a,
                        name_b,
                    )
                    final_rules.append(new_rule)
                    status = HazardStatus.RULE_ONLY

            if status is None:
                # If we have no hazard and no strong decision, skip recording metadata
                # unless there is at least some weak/ambiguous signal.
                if decision.label is None and decision.confidence <= 0.0:
                    continue
                status = HazardStatus.LLM_ONLY

            metadata[key] = HazardMetadata(status=status, rule_decision=decision)

    conflicts = detect_rule_graph_conflicts(
        feedback_graph,
        similarity_threshold=strong_not_dangerous,
    )

    return ResolvedHazards(
        rules=final_rules,
        metadata=metadata,
        conflicts=conflicts,
    )
