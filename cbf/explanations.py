from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Tuple
import requests
from services.llm import post_chat_json_system_user

def _cfg(cfg: Any, k: str, d=None):
    if cfg is None: return d
    if isinstance(cfg, dict): return cfg.get(k, d)
    return getattr(cfg, k, d)

def _objects_summary(objs: List[Any]) -> List[Dict[str,str]]:
    out=[]
    for i,o in enumerate(objs):
        name = str(getattr(o,"name", getattr(o,"id", f"obj{i}")))
        kind = str(getattr(o,"kind","object"))
        out.append({"name":name,"kind":kind})
    return out

def hazard_pair_strings(h: Dict[str,Any]) -> Tuple[str,str]:
    A=B=""
    sel = h.get("selectors")
    if isinstance(sel, dict):
        A = str((sel.get("A") or {}).get("value") or "")
        B = str((sel.get("B") or {}).get("value") or "")
        if A or B: return (A,B)
    Aobj = h.get("A") or {}
    Bobj = h.get("B") or {}
    if isinstance(Aobj, dict) or isinstance(Bobj, dict):
        A = str(Aobj.get("name") or Aobj.get("kind") or Aobj.get("text") or "")
        B = str(Bobj.get("name") or Bobj.get("kind") or Bobj.get("text") or "")
        if A or B: return (A,B)
    pair = h.get("pair")
    if isinstance(pair,(list,tuple)) and len(pair)==2:
        A = str(pair[0]); B = str(pair[1]); return (A,B)
    return ("","")

def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def attach_explanations_to_hazards(
    hazards: List[Dict[str,Any]],
    objects: List[Any],
    cfg: Any,
    only_present: bool = True,
    max_pairs: int = None,
) -> List[Dict[str,Any]]:
    if not hazards: return hazards
    obj_ctx = _objects_summary(objects)
    pairs = []
    for idx,h in enumerate(hazards):
        if only_present and isinstance(h.get("present"), bool) and not h["present"]:
            continue
        A,B = hazard_pair_strings(h)
        if A and B:
            pairs.append({"index": idx, "A": A, "B": B})
    if not pairs: return hazards
    if max_pairs is None:
        max_pairs = int(os.getenv("EXPLAIN_MAX", "40"))
    pairs = pairs[:max_pairs]

    system = (
        "You explain semantic safety risks in brief.\n"
        "Return JSON: {\"explanations\": [{\"index\": int, \"explanation\": string} ...]}.\n"
        "One item per provided pair, using the same index. Keep each explanation under 25 words."
    )
    explanations_map = {}
    for batch in _chunks(pairs, 40):
        user = (
            "Scene objects (names and kinds):\n"
            f"{json.dumps(obj_ctx, ensure_ascii=False)}\n\n"
            "Pairs to explain:\n"
            f"{json.dumps(batch, ensure_ascii=False)}\n"
        )
        try:
            out = post_chat_json_system_user(cfg, system, user, model_env="EXPLAIN_MODEL")
            for it in (out.get("explanations") or []):
                if isinstance(it, dict) and "index" in it and "explanation" in it:
                    explanations_map[int(it["index"])] = str(it["explanation"]).strip()
        except Exception:
            pass

    for idx, expl in explanations_map.items():
        if 0 <= idx < len(hazards):
            hazards[idx]["explanation"] = expl
    return hazards