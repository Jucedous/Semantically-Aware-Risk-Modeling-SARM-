from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

def sigmoid_stable(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, np.ndarray):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))
    else:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.maximum(np.dot(v, v), eps)))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))

@dataclass
class Sphere:
    center: np.ndarray
    radius: float

@dataclass
class ObjectState:
    name: str
    sphere: Sphere
    kind: str = "object"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tags: Tuple[str, ...] = field(default_factory=tuple)

@dataclass
class Workspace:
    bounds: np.ndarray

@dataclass
class Scene:
    objects: List[ObjectState]
    workspace: Optional[Workspace] = None

def pairwise_distance_cbf(A: Sphere, B: Sphere, buffer: float = 0.0) -> float:
    delta = A.center - B.center
    d2 = float(np.dot(delta, delta))
    rsum = A.radius + B.radius + buffer
    return d2 - rsum * rsum

def pairwise_dhdt(
    A_center: np.ndarray, B_center: np.ndarray,
    vA: np.ndarray, vB: np.ndarray,
) -> float:
    delta = A_center - B_center
    vrel = vA - vB
    return float(2.0 * np.dot(delta, vrel))

def alpha_linear(h: float, k: float = 1.0) -> float:
    return k * h

def metric_object_collision_cbf(
    objects: List[ObjectState],
    buffer: float = 0.01,
    alpha_gain: float = 5.0,
    scale_h: float = 0.01,
    scale_res: float = 0.05,
) -> Dict:
    per_pairs = []
    h_min = np.inf
    res_min = np.inf

    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            A = objects[i]
            B = objects[j]
            h = pairwise_distance_cbf(A.sphere, B.sphere, buffer=buffer)
            dh = pairwise_dhdt(A.sphere.center, B.sphere.center, A.velocity, B.velocity)
            residual = dh + alpha_linear(h, alpha_gain)
            h_min = min(h_min, h)
            res_min = min(res_min, residual)
            per_pairs.append({
                "A": A.name, "B": B.name,
                "h": h,
                "dh": dh,
                "residual": residual,
                "risk_h": float(sigmoid_stable(-(h / scale_h))),
                "risk_residual": float(sigmoid_stable(-(residual / scale_res))),
            })

    composite = float(np.mean([p["risk_h"] * 0.6 + p["risk_residual"] * 0.4 for p in per_pairs])) if per_pairs else 0.0
    return {
        "name": "object_collision_cbf",
        "h_min": float(h_min) if per_pairs else float("inf"),
        "residual_min": float(res_min) if per_pairs else float("inf"),
        "risk": composite,
        "details": per_pairs,
        "explanation": "Distance-squared CBF between every object pair. Residual ≥ 0 is required for forward invariance."
    }

def metric_workspace_cbf_objects(
    objects: List[ObjectState],
    workspace: Optional[Workspace],
    alpha_gain: float = 5.0,
    scale_h: float = 0.01,
    scale_res: float = 0.05,
    clearance: float = 0.0,
) -> Dict:
    if workspace is None:
        return {"name": "workspace_cbf_objects", "risk": 0.0, "details": [], "explanation": "No workspace bounds provided."}
    per = []
    hmins = []
    resmins = []

    bounds = workspace.bounds
    for obj in objects:
        x = obj.sphere.center
        r = obj.sphere.radius + clearance
        v = obj.velocity

        h_axes = [
            float((x[0] - r) - bounds[0, 0]),
            float(bounds[0, 1] - (x[0] + r)),
            float((x[1] - r) - bounds[1, 0]),
            float(bounds[1, 1] - (x[1] + r)),
            float((x[2] - r) - bounds[2, 0]), 
            float(bounds[2, 1] - (x[2] + r)), 
        ]
        idx = int(np.argmin(h_axes))
        h = h_axes[idx]
        if idx == 0:   dh = float(v[0])
        elif idx == 1: dh = float(-v[0])
        elif idx == 2: dh = float(v[1])
        elif idx == 3: dh = float(-v[1])
        elif idx == 4: dh = float(v[2])
        else:          dh = float(-v[2])

        residual = dh + alpha_linear(h, alpha_gain)
        per.append({
            "object": obj.name,
            "h": h,
            "dh": dh,
            "residual": residual,
            "risk_h": float(sigmoid_stable(-(h / scale_h))),
            "risk_residual": float(sigmoid_stable(-(residual / scale_res))),
        })
        hmins.append(h)
        resmins.append(residual)
    composite = float(np.mean([p["risk_h"] * 0.5 + p["risk_residual"] * 0.5 for p in per]))
    return {
        "name": "workspace_cbf_objects",
        "h_min": float(min(hmins)) if hmins else float("inf"),
        "residual_min": float(min(resmins)) if resmins else float("inf"),
        "risk": composite,
        "details": per,
        "explanation": "Workspace AABB CBF for sphere extents; smallest face margin is the active constraint."
    }

def metric_hazard_pairings_cbf_objects(
    objects: List[ObjectState],
    alpha_gain: float = 5.0,
    scale_res: float = 0.05,
    treat_liquid_above_electronics_as_critical: bool = True,
    xy_margin: float = 0.05,
    z_gap_max: float = 0.25,
) -> Dict:
    rules = [
        ("liquid", "electronic", 0.15),
        ("sharp", "human", 0.10),
        ("heavy", "fragile", 0.20),
        ("hot", "plastic", 0.12),
    ]

    per = []
    resmins = []
    critical_violation = False
    critical_pairs: List[Dict] = []

    by_kind: Dict[str, Dict[str, ObjectState]] = {}
    def add_to_kind(k: str, o: ObjectState):
        bucket = by_kind.setdefault(k, {})
        bucket[o.name] = o

    for o in objects:
        add_to_kind(o.kind, o)
        for t in o.tags:
            add_to_kind(t, o)

    def centers_overlap(A: Sphere, B: Sphere) -> bool:
        d2 = float(np.dot(A.center - B.center, A.center - B.center))
        rsum = A.radius + B.radius
        return d2 <= (rsum * rsum)

    def liquid_above_electronics(liq: Sphere, elec: Sphere) -> bool:
        dx, dy = liq.center[0] - elec.center[0], liq.center[1] - elec.center[1]
        dxy2 = dx * dx + dy * dy
        rsum_xy = liq.radius + elec.radius + xy_margin
        horizontally_close = dxy2 <= (rsum_xy * rsum_xy)
        liquid_is_above = liq.center[2] > elec.center[2]
        z_gap = liq.center[2] - elec.center[2]
        return horizontally_close and liquid_is_above and (z_gap <= z_gap_max)

    for A_kind, B_kind, safe_clearance in rules:
        As = list(by_kind.get(A_kind, {}).values())
        Bs = list(by_kind.get(B_kind, {}).values())
        for A in As:
            for B in Bs:
                if A is B:
                    continue
                h = pairwise_distance_cbf(A.sphere, B.sphere, buffer=safe_clearance)
                dh = pairwise_dhdt(A.sphere.center, B.sphere.center, A.velocity, B.velocity)
                residual = dh + alpha_linear(h, alpha_gain)
                overlap = centers_overlap(A.sphere, B.sphere)
                entry = {
                    "A": A.name, "B": B.name, "pair": f"{A_kind}-{B_kind}",
                    "clearance": safe_clearance, "h": h, "residual": residual,
                    "overlap": overlap,
                    "risk": float(sigmoid_stable(-(residual / scale_res))),
                }
                per.append(entry)
                resmins.append(residual)
                if A_kind == "liquid" and B_kind == "electronic":
                    over_rule = treat_liquid_above_electronics_as_critical and liquid_above_electronics(A.sphere, B.sphere)
                    if overlap or over_rule:
                        critical_violation = True
                        critical_pairs.append({**entry, "critical_reason": "overlap" if overlap else "over-within-gap"})

    composite = float(np.mean([p["risk"] for p in per])) if per else 0.0
    return {
        "name": "hazard_pairings_cbf",
        "residual_min": float(min(resmins)) if resmins else float("inf"),
        "risk": composite,
        "details": per,
        "critical_violation": critical_violation,
        "critical_pairs": critical_pairs,
        "explanation": (
            "Distance-based CBFs with extra clearance for hazardous pairings. "
            "Liquid–electronics overlap or 'over-within-gap' triggers a hard fail."
        )
    }


def evaluate_scene_metrics(scene: Scene) -> Dict:
    m_collision = metric_object_collision_cbf(scene.objects)
    m_workspace = metric_workspace_cbf_objects(scene.objects, scene.workspace)
    m_hazard = metric_hazard_pairings_cbf_objects(scene.objects)

    metrics = [m_collision, m_workspace, m_hazard]

    weights = {
        "object_collision_cbf": 0.40,
        "workspace_cbf_objects": 0.10,
        "hazard_pairings_cbf": 0.50,
    }
    score = 0.0
    wsum = 0.0
    for m in metrics:
        w = weights.get(m["name"], 0.0)
        score += w * m.get("risk", 0.0)
        wsum += w
    composite_risk = float(score / max(wsum, 1e-9))

    if m_hazard.get("critical_violation", False):
        composite_risk = 1.0

    safety_score = float(np.clip(5.0 - 5.0 * composite_risk, 0.0, 5.0))

    return {
        "metrics": metrics,
        "composite_risk": composite_risk,
        "safety_score_0_to_5": safety_score,
        "safety_grade_1_to_5": safety_score,
    }
