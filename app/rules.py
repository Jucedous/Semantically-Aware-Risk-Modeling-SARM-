from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from cbf.cbf_safety_metrics import ObjectState

def _match_selector(obj: ObjectState, sel: Dict[str, str]) -> bool:
    by, val = sel.get("by"), sel.get("value")
    if by == "kind": return obj.kind == val
    if by == "name": return obj.name == val
    if by == "tag":  return val in obj.tags
    return False

def _cond_ok(expr: Optional[str], A: ObjectState, B: ObjectState) -> bool:
    if not expr: return True
    Az = float(A.sphere.center[2]); Bz = float(B.sphere.center[2])
    if expr.strip() == "Az > Bz": return Az > Bz
    if expr.strip() == "Az < Bz": return Az < Bz
    return True

def enforce_user_preferences_on_instantiated_rules(
    objects: List[ObjectState],
    rules: List[Tuple[str, str, float, float, str, str]],
    critical_by_pair: Dict[Tuple[str, str], Any],
    user_rules: List[Dict[str, Any]],
) -> Tuple[List[Tuple[str, str, float, float, str, str]], Dict[Tuple[str, str], Any]]:
    if not user_rules:
        return rules, critical_by_pair
    name2obj = {o.name: o for o in objects}
    new_rules: List[Tuple[str, str, float, float, str, str]] = []
    for (Ak, Bk, clr, w, Aname, Bname) in rules:
        A, B = name2obj.get(Aname), name2obj.get(Bname)
        if A is None or B is None:
            new_rules.append((Ak,Bk,clr,w,Aname,Bname))
            continue
        drop = False; clr2, w2 = clr, w
        for R in user_rules:
            selA = R.get("selectors",{}).get("A"); selB = R.get("selectors",{}).get("B")
            if not selA or not selB: continue
            present = R.get("override",{}).get("present", None)
            directional = bool(R.get("directional", False))
            cond = R.get("condition_expr") or R.get("relation")
            def applies(X,Y):
                return _match_selector(X, selA) and _match_selector(Y, selB) and _cond_ok(cond, X, Y)
            matched = applies(A,B) or (not directional and applies(B,A))
            if not matched: continue
            if present is False:
                drop = True
                if isinstance(critical_by_pair, dict):
                    critical_by_pair.pop((Aname,Bname), None)
                    critical_by_pair.pop((Bname,Aname), None)
                break
            elif present is True:
                if "soft_clearance_m" in R.get("override", {}):
                    clr2 = float(R["override"]["soft_clearance_m"])
                if "weight" in R.get("override", {}):
                    w2 = float(R["override"]["weight"])
        if not drop:
            new_rules.append((Ak,Bk,clr2,w2,Aname,Bname))
    existing = {(Aname,Bname) for (_,_,_,_,Aname,Bname) in new_rules}
    for R in user_rules:
        if R.get("override",{}).get("present") is not True:
            continue
        selA = R.get("selectors",{}).get("A"); selB = R.get("selectors",{}).get("B")
        if not selA or not selB: continue
        directional = bool(R.get("directional", False))
        cond = R.get("condition_expr") or R.get("relation")
        clr_new = float(R.get("override",{}).get("soft_clearance_m", 0.0))
        w_new   = float(R.get("override",{}).get("weight", 1.0))
        for A in objects:
            if not _match_selector(A, selA): continue
            for B in objects:
                if A is B: continue
                if not _match_selector(B, selB): continue
                if directional and not _cond_ok(cond, A, B): continue
                if (A.name, B.name) not in existing:
                    new_rules.append((A.kind, B.kind, clr_new, w_new, A.name, B.name))
                    existing.add((A.name,B.name))
    return new_rules, critical_by_pair
