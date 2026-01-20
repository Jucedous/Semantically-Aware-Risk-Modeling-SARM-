from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from cbf.cbf_safety_metrics import ObjectState, Sphere

def load_scene(path: str):
    items = json.loads(Path(path).read_text())
    objs = []
    for it in items:
        x, y = it.get("xy", [0.0, 0.0]); z = it.get("z", 0.0); r = float(it.get("r", 0.05))
        objs.append(ObjectState(
            name=str(it["name"]),
            kind=str(it.get("kind", "object")).strip(),
            sphere=Sphere(center=np.array([x, y, z], dtype=float), radius=r),
            tags=tuple(it.get("tags", [])),
        ))
    return objs

def to_llm_payload(objs):
    return [
        dict(
            name=o.name, kind=o.kind, tags=list(o.tags),
            xyz=[float(o.sphere.center[0]), float(o.sphere.center[1]), float(o.sphere.center[2])],
            r=float(o.sphere.radius)
        )
        for o in objs
    ]
