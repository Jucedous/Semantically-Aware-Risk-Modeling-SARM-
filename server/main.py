from __future__ import annotations

import os, json, hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# uvicorn server.main:app --host 127.0.0.1 --port 8000
from services.llm import LLMConfig, read_api_key
from cbf.cbf_safety_metrics import ObjectState, Sphere
from cbf.semantics_runtime import analyze_scene_llm, instantiate_rules, classify_object_kind_llm
from cbf.preferences import PreferenceStore
from cbf.feedback_pipeline import label_and_store_feedback
from cbf.graphs.feedback_graph import build_feedback_graph_from_rules
from cbf.hazard_resolution import resolve_semantic_hazards
from app.rules import enforce_user_preferences_on_instantiated_rules

app = FastAPI(title="Safety Semantics Server", version="1.0")

# ---------
# Models
# ---------
class SceneObjectIn(BaseModel):
    name: str
    kind: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    xyz: List[float] = Field(min_length=3, max_length=3)
    r: float = 0.05

class CompileRequest(BaseModel):
    user_id: str
    objects: List[SceneObjectIn]

class RuleOut(BaseModel):
    A: str
    B: str
    clearance: float
    weight: float

class CompileResponse(BaseModel):
    scene_signature: str
    kinds_by_name: Dict[str, str]
    params: Dict[str, float]
    rules: List[RuleOut]
    debug: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    user_id: str
    feedback: str
    objects: List[SceneObjectIn]

class FeedbackResponse(BaseModel):
    stored_rules: int


store = PreferenceStore()
cfg = LLMConfig(api_key=read_api_key(None))

_COMPILE_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

def _prefs_etag() -> str:
    """Changes when prefs file changes (simple invalidation)."""
    try:
        p = store.path
        if p.exists():
            return str(p.stat().st_mtime_ns)
    except Exception:
        pass
    return "0"

def _scene_signature(objs: List[SceneObjectIn]) -> str:
    """
    Signature should change only when semantics-relevant attributes change.
    Positions do NOT need to trigger recompile.
    """
    payload = []
    for o in sorted(objs, key=lambda x: x.name.lower()):
        payload.append({
            "name": o.name.strip(),
            "kind": (o.kind or "").strip().lower(),
            "tags": sorted([t.strip().lower() for t in (o.tags or [])]),
        })
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _to_object_states(objs: List[SceneObjectIn]) -> List[ObjectState]:
    out: List[ObjectState] = []
    for o in objs:
        kind = (o.kind or "").strip().lower() or "object"
        out.append(ObjectState(
            name=str(o.name),
            kind=kind,
            tags=tuple(o.tags or []),
            sphere=Sphere(center=np.array(o.xyz, dtype=float), radius=float(o.r)),
        ))
    return out

def _compile_semantics(req: CompileRequest) -> Dict[str, Any]:

    objects = _to_object_states(req.objects)

    paired = list(zip(req.objects, objects))
    paired.sort(key=lambda p: str(p[1].name).lower())
    req_objects_sorted = [p[0] for p in paired]
    objects = [p[1] for p in paired]

    for o_in, o_state in zip(req_objects_sorted, objects):
        if not (o_in.kind and o_in.kind.strip()):
            try:
                o_state.kind = classify_object_kind_llm(o_state.name, list(o_state.tags), cfg) or "object"
            except Exception:
                o_state.kind = "object"

    risks = analyze_scene_llm(
        [
            {"name": o.name, "kind": o.kind, "tags": list(o.tags), "r": float(o.sphere.radius)}
            for o in objects
        ],
        cfg,
        include_geometry=False,
    )

    rules_raw, crit_map_raw = instantiate_rules(objects, risks)

    user_rules = store.list_rules(req.user_id)
    rules_pref, crit_map_pref = enforce_user_preferences_on_instantiated_rules(
        objects, rules_raw, crit_map_raw, user_rules
    )

    fg = build_feedback_graph_from_rules(user_rules, cfg)
    resolved = resolve_semantic_hazards(
        objects=objects,
        semantic_rules=rules_pref,
        feedback_graph=fg,
    )

    rules_out: List[Dict[str, Any]] = []
    for (_Ak, _Bk, clr, w, Aname, Bname) in resolved.rules:
        rules_out.append({
            "A": Aname,
            "B": Bname,
            "clearance": float(clr),
            "weight": float(w),
        })

    kinds_by_name = {o.name: o.kind for o in objects}

    debug_meta = {}
    for (a, b), md in resolved.metadata.items():
        key = f"{a}|{b}"
        debug_meta[key] = {
            "status": str(md.status),
            "confidence": float(md.rule_decision.confidence),
            "label": md.rule_decision.label,
            "sources": md.rule_decision.sources,
        }

    return {
        "kinds_by_name": kinds_by_name,
        "rules": rules_out,
        "debug": {
            "metadata": debug_meta,
            "conflicts": [c.__dict__ for c in (resolved.conflicts or [])],
        }
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/compile", response_model=CompileResponse)
def compile_scene(req: CompileRequest):
    scene_sig = _scene_signature(req.objects)
    etag = _prefs_etag()
    cache_key = (req.user_id, scene_sig, etag)

    if cache_key in _COMPILE_CACHE:
        return _COMPILE_CACHE[cache_key]

    try:
        compiled = _compile_semantics(req)
        resp = {
            "scene_signature": scene_sig,
            "kinds_by_name": compiled["kinds_by_name"],
            "params": {"alpha_gain": 5.0, "scale_res": 0.05},
            "rules": compiled["rules"],
            "debug": compiled["debug"],
        }
        _COMPILE_CACHE[cache_key] = resp
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    try:
        objects = _to_object_states(req.objects)
        stored = label_and_store_feedback(
            text=req.feedback,
            objects=objects,
            cfg=cfg,
            user_id=req.user_id,
            store=store,
        )
        for k in list(_COMPILE_CACHE.keys()):
            if k[0] == req.user_id:
                _COMPILE_CACHE.pop(k, None)
        return {"stored_rules": int(stored)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
