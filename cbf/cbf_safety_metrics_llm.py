from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np

from .cbf_safety_metrics import Sphere, ObjectState

def metric_hazard_pairings_cbf_objects_llm(
    objects: List[ObjectState],
    alpha_gain: float = 5.0,
    scale_res: float = 0.05,
    rules: List[Tuple[str, str, float, float, str, str]] = [],
    critical_by_pair: Optional[Dict[Tuple[str, str], Callable[[Sphere, Sphere], bool]]] = None,
    include_collision_baseline: bool = True,
    baseline_weight: float = 0.25,
    baseline_clearance_m: float = 0.0,
    show_collision_for_labeled_pairs: bool = False,
    collision_visual_only: bool = True,
    label_semantic_entries: bool = True,
) -> Dict:
    critical_by_pair = critical_by_pair or {}

    by_name = {o.name: o for o in objects}

    metrics: List[Dict[str, Any]] = []
    resmins: List[float] = []
    critical_violation = False
    critical_pairs: List[Dict[str, str]] = []

    def residual_for_pair(A: ObjectState, B: ObjectState, safe_clearance: float) -> float:
        d = float(np.linalg.norm(A.sphere.center - B.sphere.center))
        return d - (A.sphere.radius + B.sphere.radius + safe_clearance)

    def sigmoid_stable(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    labeled_pairs = set()
    for A_kind, B_kind, safe_clearance, weight, A_name, B_name in rules:
        A = by_name.get(A_name); B = by_name.get(B_name)
        if A is None or B is None or A is B:
            continue

        res = residual_for_pair(A, B, safe_clearance)
        resmins.append(res)
        risk = sigmoid_stable(-(res) / max(scale_res, 1e-6) * alpha_gain)

        name = f"{A_name}->{B_name}" + (" [semantic]" if label_semantic_entries else "")
        metrics.append({
            "name": name,
            "risk": float(risk),
            "weight": float(weight),
            "channel": "semantic",
            "diagnostic_only": False,
        })

        labeled_pairs.add((A_name, B_name))
        labeled_pairs.add((B_name, A_name))

        checker = critical_by_pair.get((A_name, B_name))
        if checker and checker(A.sphere, B.sphere):
            critical_violation = True
            critical_pairs.append({"A": A.name, "B": B.name, "predicate": "critical"})

    if include_collision_baseline and len(objects) >= 2 and baseline_weight > 0.0:
        N = len(objects)
        for i in range(N):
            Ai = objects[i]
            for j in range(i + 1, N):
                Bj = objects[j]
                if (Ai.name, Bj.name) not in labeled_pairs and (Bj.name, Ai.name) not in labeled_pairs:
                    res = residual_for_pair(Ai, Bj, baseline_clearance_m)
                    resmins.append(res)
                    risk = sigmoid_stable(-(res) / max(scale_res, 1e-6) * alpha_gain)
                    metrics.append({
                        "name": f"{Ai.name}<->{Bj.name} (collision)",
                        "risk": float(risk),
                        "weight": float(baseline_weight),
                        "channel": "collision",
                        "diagnostic_only": False,
                    })

    if show_collision_for_labeled_pairs and len(objects) >= 2 and baseline_weight > 0.0:
        seen = set()
        for (A_name, B_name) in labeled_pairs:
            if (B_name, A_name) in seen:
                continue
            seen.add((A_name, B_name))
            A = by_name.get(A_name); B = by_name.get(B_name)
            if A is None or B is None or A is B:
                continue
            res = residual_for_pair(A, B, baseline_clearance_m)
            resmins.append(res)
            risk = sigmoid_stable(-(res) / max(scale_res, 1e-6) * alpha_gain)

            metrics.append({
                "name": f"{A.name}<->{B.name} (collision)",
                "risk": float(risk),
                "weight": float(baseline_weight),
                "channel": "collision",
                "diagnostic_only": bool(collision_visual_only),
            })

    score_sum, wsum = 0.0, 0.0
    for m in metrics:
        if m.get("diagnostic_only", False):
            continue
        w = float(m.get("weight", 1.0))
        score_sum += w * float(m.get("risk", 0.0))
        wsum += w
    composite_risk = float(score_sum / max(wsum, 1e-9)) if wsum > 0 else 0.0

    if critical_violation:
        composite_risk = 1.0

    safety_score = float(np.clip(5.0 - 5.0 * composite_risk, 0.0, 5.0))

    return {
        "name": "hazard_pairings_llm",
        "metrics": metrics,
        "residual_min": float(min(resmins)) if resmins else float("inf"),
        "composite_risk": composite_risk,
        "critical_violation": bool(critical_violation),
        "critical_pairs": critical_pairs,
        "safety_score_0_to_5": safety_score,
    }
