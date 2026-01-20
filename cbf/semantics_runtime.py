from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import json, math, re, os
from services.llm import LLMConfig, post_chat_json_messages
from pathlib import Path

_ALLOWED_FUNCS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "dot": lambda ax, ay, az, bx, by, bz: (ax*bx + ay*by + az*bz),
    "norm": lambda x, y, z: math.sqrt(x*x + y*y + z*z),
    "norm_xy": lambda x, y: math.sqrt(x*x + y*y),
}

_ALLOWED_NAMES = set(_ALLOWED_FUNCS.keys()) | {
    "Ax", "Ay", "Az", "Ar",
    "Bx", "By", "Bz", "Br",
}

class SafeExpr:
    def __init__(self, expr: str):
        import ast
        expr = expr.replace("&&", " and ").replace("||", " or ")
        self._src = expr
        self._ast = ast.parse(expr, mode="eval")

        allowed_nodes = (
            ast.Expression,
            ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Call, ast.Name, ast.Load, ast.Constant, ast.Tuple,
            ast.And, ast.Or, ast.Not,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        )

        for node in ast.walk(self._ast):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Disallowed AST node: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function names allowed")
                if node.func.id not in _ALLOWED_FUNCS:
                    raise ValueError(f"Function '{node.func.id}' not allowed")
                if node.keywords:
                    raise ValueError("Keyword args not allowed")
            if isinstance(node, ast.Name):
                if node.id not in _ALLOWED_NAMES:
                    raise ValueError(f"Name '{node.id}' not allowed")

    def __call__(self, *, Ax, Ay, Az, Ar, Bx, By, Bz, Br) -> Any:
        import ast
        env = {
            "Ax": Ax, "Ay": Ay, "Az": Az, "Ar": Ar,
            "Bx": Bx, "By": By, "Bz": Bz, "Br": Br,
            **_ALLOWED_FUNCS,
        }

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return env[node.id]
            if isinstance(node, ast.UnaryOp):
                v = eval_node(node.operand)
                if   isinstance(node.op, ast.UAdd): return +v
                elif isinstance(node.op, ast.USub): return -v
                elif isinstance(node.op, ast.Not):  return not v
            if isinstance(node, ast.BinOp):
                l, r = eval_node(node.left), eval_node(node.right)
                if   isinstance(node.op, ast.Add):  return l + r
                elif isinstance(node.op, ast.Sub):  return l - r
                elif isinstance(node.op, ast.Mult): return l * r
                elif isinstance(node.op, ast.Div):  return l / r
                elif isinstance(node.op, ast.Pow):  return l ** r
            if isinstance(node, ast.BoolOp):
                vals = [eval_node(v) for v in node.values]
                if isinstance(node.op, ast.And):
                    out = True
                    for v in vals:
                        out = out and v
                        if not out: break
                    return out
                if isinstance(node.op, ast.Or):
                    out = False
                    for v in vals:
                        out = out or v
                        if out: break
                    return out
            if isinstance(node, ast.Compare):
                left = eval_node(node.left)
                for op_node, right_node in zip(node.ops, node.comparators):
                    right = eval_node(right_node)
                    if   isinstance(op_node, ast.Eq):    ok = (left == right)
                    elif isinstance(op_node, ast.NotEq): ok = (left != right)
                    elif isinstance(op_node, ast.Lt):    ok = (left <  right)
                    elif isinstance(op_node, ast.LtE):   ok = (left <= right)
                    elif isinstance(op_node, ast.Gt):    ok = (left >  right)
                    elif isinstance(op_node, ast.GtE):   ok = (left >= right)
                    else: raise ValueError("bad comparator")
                    if not ok: return False
                    left = right
                return True
            if isinstance(node, ast.Call):
                fn = eval_node(node.func)
                args = [eval_node(a) for a in node.args]
                return fn(*args)
            if isinstance(node, ast.Tuple):
                return tuple(eval_node(e) for e in node.elts)
            raise ValueError("bad expression")
        return eval_node(self._ast)

@dataclass
class CompiledRisk:
    sel_a: Tuple[str, str]
    sel_b: Tuple[str, str]
    risk_type: str
    weight: float
    soft_clearance_m: float
    critical_fn: Optional[Callable[[Any, Any], bool]]

def _inline_params_in_expr(expr: str, param_values: Dict[str, float]) -> str:
    out = expr
    for name, val in param_values.items():
        out = re.sub(rf"\b{re.escape(name)}\b", str(float(val)), out)
    return out

def _pair_features(scene_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    feats: List[Dict[str, Any]] = []
    n = len(scene_objs)
    for i in range(n):
        for j in range(i + 1, n):
            A = scene_objs[i]; B = scene_objs[j]
            Ax, Ay, Az = A["xyz"]; Ar = float(A["r"])
            Bx, By, Bz = B["xyz"]; Br = float(B["r"])
            dx, dy, dz = Ax - Bx, Ay - By, Az - Bz
            d_xy = math.sqrt(dx*dx + dy*dy)
            d_xyz = math.sqrt(dx*dx + dy*dy + dz*dz)
            overlap = d_xyz <= (Ar + Br)
            A_above_B = Az > Bz
            B_above_A = Bz > Az
            feats.append(dict(
                A_name=A["name"], A_kind=A["kind"], A_r=Ar,
                B_name=B["name"], B_kind=B["kind"], B_r=Br,
                d_xy=d_xy, d_xyz=d_xyz, dz=dz,
                overlap=bool(overlap),
                A_above_B=bool(A_above_B),
                B_above_A=bool(B_above_A),
            ))
    return feats


def _pair_semantic_pairs(scene_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a stable, geometry-independent candidate pair list.

    This is used to keep LLM semantic hazard compilation consistent across runs,
    even if object spawn/physics positions differ.
    """
    feats: List[Dict[str, Any]] = []
    n = len(scene_objs)
    for i in range(n):
        for j in range(i + 1, n):
            A = scene_objs[i]; B = scene_objs[j]
            feats.append(dict(
                A_name=A.get("name", ""),
                A_kind=A.get("kind", "object"),
                A_tags=list(A.get("tags") or []),
                A_r=float(A.get("r", 0.05)),
                B_name=B.get("name", ""),
                B_kind=B.get("kind", "object"),
                B_tags=list(B.get("tags") or []),
                B_r=float(B.get("r", 0.05)),
            ))
    return feats

_PAIR_SYSTEM_PROMPT = """You are a safety analyst for object-only manipulation scenes.

You will receive a list of candidate object pairs with precomputed features.
For EACH candidate pair, decide whether the pair is a **SEMANTICALLY HAZARDOUS CATEGORY** in general
(e.g., if the object types imply a plausible risk in typical settings), regardless of current spacing.

Return STRICT JSON:

{
  "assessments": [
    {
      "A_name": "<string>",
      "B_name": "<string>",
      "present": true | false,              // true = semantically hazardous category (independent of distance)
      "risk_type": "<short label or ''>",   // e.g., "spill", "cut", "crush", "heat", ...
      "weight": <number 0..inf>,            // semantic importance of this hazard category
      "soft_clearance_m": <number>,         // clearance margin the CBF should prefer (still geometric)
      "critical_condition": {
        "expression": "<boolean formula using ONLY Ax,Ay,Az,Ar,Bx,By,Bz,Br and funcs: abs,min,max,sqrt,norm,norm_xy,dot>"
      } | null,
      "reason": "<one-line justification>"
    }
  ]
}

IMPORTANT:
- `present` is about SEMANTIC hazard (category), NOT whether the pair is currently close.
- Even if far apart now, if the pair type is a plausible hazard in typical usage, set present=true and assign a reasonable weight and clearance.
- Use `critical_condition` ONLY for clear, geometry-based red lines (e.g., A above B within small gap or overlap).

STRICT RULES:
- Decide per-pair; do not skip a pair silently. If no semantic hazard, set present=false, weight=0, soft_clearance_m=0.
- Expressions MUST use ONLY the allowed variables/functions (no custom names).
- No commentary outside JSON.
"""

_PAIR_USER_PROMPT = """Scene objects:
{scene_json}

Candidate pairs with features:
{pair_json}

Task:
- For EACH candidate pair above, return one assessment object in the same order.
- If the pair is a SEMANTICALLY hazardous category, set present=true and propose weight and soft_clearance_m
  (e.g., liquids vs electronics, sharp vs human, heavy vs fragile, etc. — do not hard-code rules; infer from names/kinds/tags).
- If a CRITICAL condition is appropriate, provide a boolean expression using ONLY Ax,Ay,Az,Ar,Bx,By,Bz,Br and allowed funcs.
- If NOT semantically hazardous, present=false, weight=0, soft_clearance_m=0, critical_condition=null.
"""

_PAIR_SYSTEM_PROMPT_SEMANTIC_ONLY = """You are a safety analyst for object-only manipulation scenes.

You will receive a list of candidate object pairs described only by semantic cues (names/kinds/tags),
NOT current geometry. Your task is to decide whether the pair is a **SEMANTICALLY HAZARDOUS CATEGORY**
(e.g., if the object types imply a plausible risk in typical settings), regardless of spacing.

Return STRICT JSON:

{
  "assessments": [
    {
      "A_name": "<string>",
      "B_name": "<string>",
      "present": true | false,
      "risk_type": "<short label or ''>",
      "weight": <number 0..inf>,
      "soft_clearance_m": <number>,
      "critical_condition": {
        "expression": "<boolean formula using ONLY Ax,Ay,Az,Ar,Bx,By,Bz,Br and funcs: abs,min,max,sqrt,norm,norm_xy,dot>"
      } | null,
      "reason": "<one-line justification>"
    }
  ]
}

IMPORTANT:
- `present` is about SEMANTIC hazard (category), NOT whether the pair is currently close.
- Even if far apart now, if the pair type is a plausible hazard in typical usage, set present=true.
- If you include a `critical_condition`, write it GENERICALLY as a geometry predicate over the variables.
  (Ax,Ay,Az are A's center coordinates in meters; Ar is radius; likewise Bx,By,Bz,Br.)

STRICT RULES:
- Decide per-pair; do not skip a pair silently. If no semantic hazard, set present=false, weight=0, soft_clearance_m=0.
- Expressions MUST use ONLY the allowed variables/functions (no custom names).
- No commentary outside JSON.
"""

_PAIR_USER_PROMPT_SEMANTIC_ONLY = """Scene objects (semantic fields only):
{scene_json}

Candidate pairs:
{pair_json}

Task:
- For EACH candidate pair above, return one assessment object in the same order.
- If the pair is a SEMANTICALLY hazardous category, set present=true and propose weight and soft_clearance_m.
- If NOT semantically hazardous, present=false, weight=0, soft_clearance_m=0, critical_condition=null.
"""

def analyze_scene_llm(scene_objs: List[Dict[str, Any]], cfg: LLMConfig, *, include_geometry: bool = True) -> List[CompiledRisk]:
    """Ask the LLM to identify semantic hazard categories for object pairs.

    Parameters
    ----------
    include_geometry:
        If True, include distance/overlap features derived from xyz in the prompt.
        If False (recommended for consistency), only semantic descriptors are sent,
        making outputs stable across runs even if object positions differ.
    """

    # Build a stable candidate pair list.
    if include_geometry:
        pairs = _pair_features(scene_objs)
        scene_payload = scene_objs
        system_base = _PAIR_SYSTEM_PROMPT
        user_template = _PAIR_USER_PROMPT
    else:
        pairs = _pair_semantic_pairs(scene_objs)
        # Strip any geometry from the scene payload to keep prompts stable.
        scene_payload = [
            {
                "name": o.get("name", ""),
                "kind": o.get("kind", "object"),
                "tags": list(o.get("tags") or []),
                "r": float(o.get("r", 0.05)),
            }
            for o in scene_objs
        ]
        system_base = _PAIR_SYSTEM_PROMPT_SEMANTIC_ONLY
        user_template = _PAIR_USER_PROMPT_SEMANTIC_ONLY

    pair_count = len(pairs)

    strict_system = system_base + f"""

CONSTRAINTS:
- There are EXACTLY {pair_count} candidate pairs in this request.
- You MUST return `assessments` with EXACTLY {pair_count} items, in the SAME ORDER as given.
- For any pair with NO semantic hazard, set: present=false, weight=0, soft_clearance_m=0, critical_condition=null.
"""

    messages = [
        {"role": "system", "content": strict_system},
        {"role": "user", "content": user_template.format(
            scene_json=json.dumps(scene_payload, indent=2),
            pair_json=json.dumps(pairs, indent=2),
        )},
    ]
    data = post_chat_json_messages(cfg, messages)

    assessments = data.get("assessments", [])
    if not isinstance(assessments, list) or len(assessments) != pair_count:
        msg = f"LLM returned {len(assessments) if isinstance(assessments,list) else 'non-list'} assessments; expected {pair_count}."
        raise RuntimeError(msg)

    def quantize_clearance(x: float, step: float = 0.05) -> float:
        x = max(0.0, float(x))
        return round(x / step) * step

    out: List[CompiledRisk] = []

    for assess, pair in zip(assessments, pairs):
        A_name, B_name = pair["A_name"], pair["B_name"]
        present = bool(assess.get("present", False))

        # Deterministic post-processing to reduce tiny numeric jitter across runs.
        try:
            weight = float(assess.get("weight", 0.0))
        except Exception:
            weight = 0.0
        try:
            soft_clearance = float(assess.get("soft_clearance_m", 0.0))
        except Exception:
            soft_clearance = 0.0

        if not present:
            weight = 0.0
            soft_clearance = 0.0
        else:
            if weight <= 0.0:
                weight = 1.0
            weight = round(weight, 3)
            soft_clearance = quantize_clearance(soft_clearance, step=0.05)

        cc = assess.get("critical_condition")

        crit_fn = None
        if present and isinstance(cc, dict) and isinstance(cc.get("expression"), str):
            expr_src = cc["expression"]
            expr_src = _inline_params_in_expr(expr_src, {"soft_clearance_m": soft_clearance})
            expr = SafeExpr(expr_src)
            def make_checker(e: SafeExpr):
                def _chk(sA, sB):
                    return bool(e(
                        Ax=float(sA.center[0]), Ay=float(sA.center[1]), Az=float(sA.center[2]), Ar=float(sA.radius),
                        Bx=float(sB.center[0]), By=float(sB.center[1]), Bz=float(sB.center[2]), Br=float(sB.radius),
                    ))
                return _chk
            crit_fn = make_checker(expr)

        if present and weight > 0.0:
            out.append(CompiledRisk(
                sel_a=("name", A_name),
                sel_b=("name", B_name),
                risk_type=str(assess.get("risk_type", "")).strip(),
                weight=weight,
                soft_clearance_m=soft_clearance,
                critical_fn=crit_fn,
            ))

    return out


def _matches(obj, sel: Tuple[str, str]) -> bool:
    by, val = sel
    val = val.lower()
    if by == "kind":
        return obj.kind.lower() == val
    if by == "name":
        return obj.name.lower() == val
    if by == "tag":
        return any(t.lower() == val for t in obj.tags)
    return False

def instantiate_rules(objects, compiled: List[CompiledRisk]):
    rules: List[Tuple[str, str, float, float, str, str]] = []
    crits: Dict[Tuple[str, str], Callable[[Any, Any], bool]] = {}
    for cr in compiled:
        As = [o for o in objects if _matches(o, cr.sel_a)]
        Bs = [o for o in objects if _matches(o, cr.sel_b)]
        for A in As:
            for B in Bs:
                if A is B:
                    continue
                rules.append((A.kind, B.kind, float(cr.soft_clearance_m), float(cr.weight), A.name, B.name))
                if cr.critical_fn:
                    crits[(A.name, B.name)] = cr.critical_fn
    return rules, crits


# ---------------------------------------------------------------------------
# Persistent kind cache (reduces LLM variability across runs)
# ---------------------------------------------------------------------------

_KIND_CACHE_PATH = Path(__file__).resolve().parents[1] / "models" / "kind_cache.json"
_KIND_CACHE: Dict[str, str] = {}


def _kind_cache_key(name: str, tags: List[str]) -> str:
    n = " ".join((name or "").strip().lower().split())
    ts = [" ".join((t or "").strip().lower().split()) for t in (tags or []) if (t or "").strip()]
    ts = sorted(set(ts))
    return n + "|" + ",".join(ts)


def _load_kind_cache() -> None:
    global _KIND_CACHE
    try:
        if _KIND_CACHE_PATH.exists():
            data = json.loads(_KIND_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # only keep simple str->str mappings
                clean = {}
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str) and k and v:
                        clean[k] = v
                _KIND_CACHE = clean
    except Exception:
        # cache is optional; ignore read errors
        pass


def _save_kind_cache() -> None:
    try:
        _KIND_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _KIND_CACHE_PATH.write_text(json.dumps(_KIND_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # best-effort only
        pass


_load_kind_cache()

def classify_object_kind_llm(name: str, tags: List[str], cfg: LLMConfig) -> str:
    cache_key = _kind_cache_key(name, list(tags or []))
    cached = _KIND_CACHE.get(cache_key)
    if isinstance(cached, str) and 1 <= len(cached) <= 64:
        return cached

    try:
        sys_prompt = (
            "You classify ONE object name into a short, general 'kind' label for safety semantics.\n"
            "Return STRICT JSON only: {\"kind\": \"<lowercase kind>\"}.\n"
            "Prefer hazard-relevant kinds when obvious: liquid, electronic, human, sharp, flammable, heat, fragile,\n"
            "heavy, gas, chemical, toxic, explosive, tool, container, food, drink, battery, cable, robot, paper.\n"
            "If ambiguous, return 'object'. No extra fields or commentary."
        )
        user_payload = {"name": str(name), "tags": list(tags or [])}
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        data = post_chat_json_messages(cfg, messages)
        k = str(data.get("kind", "")).strip().lower()
        if 1 <= len(k) <= 64:
            _KIND_CACHE[cache_key] = k
            _save_kind_cache()
            return k
    except Exception:
        pass
    return "object"