from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Set, Optional, Literal

from services.llm import LLMConfig, post_chat_json_messages


EdgeKind = Literal["dangerous", "not_dangerous", "similar"]


@dataclass
class FeedbackEdge:
    a: str
    b: str
    kinds: Set[EdgeKind] = field(default_factory=set)
    soft_clearance_m: Optional[float] = None
    weight: Optional[float] = None
    similarity_score: Optional[float] = None
    rule_ids: Set[str] = field(default_factory=set)


@dataclass
class FeedbackGraph:
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[Tuple[str, str], FeedbackEdge] = field(default_factory=dict)

    def _key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def add_node(self, concept: str) -> None:
        self.nodes.add(concept)

    def add_rule_edge(
        self,
        a: str,
        b: str,
        *,
        dangerous: bool,
        rule_id: Optional[str] = None,
        soft_clearance_m: Optional[float] = None,
        weight: Optional[float] = None,
    ) -> None:
        self.add_node(a)
        self.add_node(b)
        key = self._key(a, b)
        if key not in self.edges:
            self.edges[key] = FeedbackEdge(a=key[0], b=key[1])
        edge = self.edges[key]

        if dangerous:
            edge.kinds.add("dangerous")
            if soft_clearance_m is not None:
                edge.soft_clearance_m = (
                    max(edge.soft_clearance_m, float(soft_clearance_m))
                    if edge.soft_clearance_m is not None
                    else float(soft_clearance_m)
                )
            if weight is not None:
                edge.weight = (
                    max(edge.weight, float(weight))
                    if edge.weight is not None
                    else float(weight)
                )
        else:
            edge.kinds.add("not_dangerous")

        if rule_id is not None:
            edge.rule_ids.add(str(rule_id))

    def add_similarity_edge(
        self,
        a: str,
        b: str,
        *,
        similarity_score: float,
    ) -> None:
        self.add_node(a)
        self.add_node(b)
        key = self._key(a, b)
        if key not in self.edges:
            self.edges[key] = FeedbackEdge(a=key[0], b=key[1])
        edge = self.edges[key]
        edge.kinds.add("similar")
        if edge.similarity_score is None:
            edge.similarity_score = float(similarity_score)
        else:
            edge.similarity_score = max(edge.similarity_score, float(similarity_score))


def _canonical_concept(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


_SIMILARITY_SYSTEM_PROMPT = """
You help build a semantic similarity graph for safety-related concepts.
Given a list of concept strings (like "knife", "human", "kid", "blade", "coke"),
identify which pairs are semantically similar in the context of physical safety
(e.g. human and kid are similar, knife and blade are similar).
Return ONLY a JSON object of the form:
{"similarities": [{"a": "...", "b": "...", "score": 0.0}]}
where score is in [0, 1]. Only include pairs with score >= 0.5.
Use the concept strings exactly as they appear in the input.
""".strip()


_SIMILARITY_THRESHOLD = 0.7


def _llm_similarity_pairs(concepts: List[str], cfg: LLMConfig) -> List[Dict[str, Any]]:
    if len(concepts) < 2:
        return []

    payload = {"concepts": concepts}
    messages = [
        {"role": "system", "content": _SIMILARITY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload)},
    ]
    try:
        data = post_chat_json_messages(cfg, messages) or {}
    except Exception as e:
        print(f"[FeedbackGraph] similarity LLM call failed: {e}")
        return []

    sims = data.get("similarities", [])
    if not isinstance(sims, list):
        return []

    clean: List[Dict[str, Any]] = []
    for item in sims:
        if not isinstance(item, dict):
            continue
        a = str(item.get("a", "")).strip()
        b = str(item.get("b", "")).strip()
        if not a or not b:
            continue
        if a == b:
            continue
        try:
            score = float(item.get("score", 0.0))
        except Exception:
            continue
        clean.append({"a": a, "b": b, "score": score})
    return clean


def build_feedback_graph_from_rules(
    user_rules: List[Dict[str, Any]],
    cfg: LLMConfig,
) -> FeedbackGraph:
    graph = FeedbackGraph()

    for rule in user_rules:
        selectors = rule.get("selectors", {}) or {}
        selA = selectors.get("A") or {}
        selB = selectors.get("B") or {}

        raw_a = str(selA.get("value", "")).strip()
        raw_b = str(selB.get("value", "")).strip()
        if not raw_a or not raw_b:
            continue

        concept_a = _canonical_concept(raw_a)
        concept_b = _canonical_concept(raw_b)
        if not concept_a or not concept_b:
            continue

        override = rule.get("override", {}) or {}
        present = override.get("present", True)
        dangerous = bool(present)

        clr = override.get("soft_clearance_m")
        w = override.get("weight")

        graph.add_rule_edge(
            concept_a,
            concept_b,
            dangerous=dangerous,
            rule_id=rule.get("id"),
            soft_clearance_m=clr,
            weight=w,
        )

    concepts = sorted(graph.nodes)
    sims = _llm_similarity_pairs(concepts, cfg)
    for item in sims:
        a = _canonical_concept(item["a"])
        b = _canonical_concept(item["b"])
        if a not in graph.nodes or b not in graph.nodes:
            continue
        score = float(item["score"])
        if score < _SIMILARITY_THRESHOLD:
            continue
        graph.add_similarity_edge(a, b, similarity_score=score)

    return graph
