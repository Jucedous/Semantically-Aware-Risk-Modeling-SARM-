from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Optional, Tuple
import requests
from pathlib import Path
from cbf.preferences import PreferenceStore
from services.llm import post_chat_json_system_user, compose_completions_url as _compose_completions_url, read_api_key


def _cfg_field(cfg: Any, name: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _objects_summary(objects: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, o in enumerate(objects):
        name = str(getattr(o, "name", getattr(o, "id", f"obj{i}")))
        kind = str(getattr(o, "kind", "object"))
        out.append({"name": name, "kind": kind})
    return out


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _normalize_selector(sel: Dict[str, str], objects: List[Any]) -> Dict[str, str]:
    by = (sel.get("by") or "").strip().lower()
    val = str(sel.get("value") or "").strip()
    if not val:
        return {"by": "kind", "value": "object"}
    names = {_canon(getattr(o, "name", "")): getattr(o, "name", "") for o in objects}
    kinds = {_canon(getattr(o, "kind", "")): getattr(o, "kind", "") for o in objects}
    cval = _canon(val)
    if cval in names:
        return {"by": "name", "value": names[cval]}
    if cval in kinds:
        return {"by": "kind", "value": kinds[cval]}
    for k, human in names.items():
        if cval and k and cval in k:
            return {"by": "name", "value": human}
    for k, human in kinds.items():
        if cval and k and cval in k:
            return {"by": "kind", "value": human}
    if by in ("name", "kind", "tag"):
        return {"by": by, "value": val}
    return {"by": "kind", "value": val}


def _iter_balanced_json_objects(text: str):
    stack: List[str] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    yield text[start : i + 1]
                    start = None


def _clean_json_quiet(s: str) -> str:
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _extract_json_block(text: str) -> Dict[str, Any]:
    candidates: List[str] = []
    for obj_text in _iter_balanced_json_objects(text):
        candidates.append(obj_text)
    if not candidates:
        raise ValueError("No JSON object found in model output.")
    def score(s: str) -> Tuple[int, int]:
        return (0 if ('"present"' in s) else 1, -len(s))
    candidates.sort(key=score)
    last_err: Optional[Exception] = None
    for c in candidates:
        try:
            return json.loads(_clean_json_quiet(c))
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not parse JSON from model output. Last error: {last_err}")


_CHAT_SYSTEM = (
    "You convert user preferences into a single JSON object with keys:\n"
    '  "present": true or false\n'
    '  "pairs": [ { "A": {"text": string}, "B": {"text": string} } ]\n'
    "\n"
    "INTERPRETATION RULES:\n"
    "- Treat the user's sentence as a NORMATIVE OVERRIDE, not a factual claim.\n"
    "- If the user says the relation is allowed/safe/not dangerous/OK together => present=false.\n"
    "- If the user says the relation is dangerous/avoid/prohibit/not allowed together => present=true.\n"
    "- Respect explicit negation words (e.g., 'not', 'should not', 'no', 'never').\n"
    "- Do NOT use outside world knowledge to flip the user's intent.\n"
    "\n"
    "Return ONLY the JSON."
)


_CHAT_USER_TEMPLATE = (
    "User feedback:\n{feedback}\n\n"
    "Available objects (names and kinds):\n{objects}\n"
)


_CONFIRM_SYSTEM = """
You check whether a boolean hazard label matches the USER'S INTENT in a feedback sentence.

Meaning of the boolean `present`:
- present = true  => the relation between A and B should be treated as DANGEROUS / NOT ALLOWED.
- present = false => the relation should be treated as SAFE / ALLOWED / NOT DANGEROUS.

CRITICAL:
- Base your judgment ONLY on what the user wrote.
- Ignore your own world knowledge (e.g., that knives are usually dangerous around humans).
- If user wording clearly suggests "safe / not dangerous / should not be dangerous" => user_intent_present=false.
- If user wording clearly suggests "dangerous / should be avoided / must not happen" => user_intent_present=true.

You will receive:
- the user sentence,
- the model's current `present` label.

Return STRICT JSON:

{
  "user_intent_present": true or false,
  "needs_confirmation": true or false,
  "confirmation_question": "<short question to show the user or ''>"
}

Rules:
- Set user_intent_present based on the user's sentence ONLY.
- If user_intent_present == model_present, then needs_confirmation=false and confirmation_question="".
- If they differ, needs_confirmation=true and confirmation_question should be a SHORT yes/no question like:
  - "This looks dangerous to me. Do you want to treat it as dangerous?"
  - or "Do you really mean this should NOT be treated as dangerous?"
"""


def _confirm_present_with_llm(text: str, model_present: bool, cfg: Any) -> Dict[str, Any]:
    payload = {
        "feedback": text.strip(),
        "model_present": bool(model_present),
    }
    user_prompt = json.dumps(payload, ensure_ascii=False)
    out = post_chat_json_system_user(cfg, _CONFIRM_SYSTEM, user_prompt)
    ui = out.get("user_intent_present")
    if isinstance(ui, str):
        ui = ui.strip().lower() in ("true", "1", "yes", "y")
    elif not isinstance(ui, bool):
        ui = model_present
    nc = out.get("needs_confirmation")
    if isinstance(nc, str):
        nc = nc.strip().lower() in ("true", "1", "yes", "y")
    elif not isinstance(nc, bool):
        nc = False
    q = out.get("confirmation_question")
    if not isinstance(q, str):
        q = ""
    q = q.strip()
    return {
        "user_intent_present": bool(ui),
        "needs_confirmation": bool(nc),
        "confirmation_question": q,
    }


_PROMPT_TEMPLATE = """Return ONLY JSON with fields:
{
  "present": true|false,
  "pairs": [
    { "A": {"text": "<name or kind>"}, "B": {"text": "<name or kind>"} }
  ]
}
User feedback:
{feedback}

Available objects (names and kinds):
{objects}
"""


def _guess_pairs_from_text(text: str, objects: List[Any]) -> List[Dict[str, Any]]:
    t = text.lower()
    names = [(getattr(o, "name", ""), getattr(o, "name", "").lower()) for o in objects]
    kinds = [(getattr(o, "kind", ""), getattr(o, "kind", "").lower()) for o in objects]
    mentions: List[str] = []
    for human, l in names:
        if human and l and l in t and human not in mentions:
            mentions.append(human)
    for human, l in kinds:
        if human and l and l in t and human not in mentions:
            mentions.append(human)
    pairs: List[Dict[str, Any]] = []
    if len(mentions) >= 2:
        pairs.append({"A": {"text": mentions[0]}, "B": {"text": mentions[1]}})
    return pairs


def label_and_store_feedback(
    text: str,
    objects: List[Any],
    cfg: Any,
    user_id: str,
    store: Optional[PreferenceStore] = None,
    model_fallback: str = "gpt-3.5-turbo-instruct",
) -> int:
    if not text or not text.strip():
        raise ValueError("Empty feedback.")
    objects_ctx = _objects_summary(objects)
    mode = os.environ.get("FEEDBACK_API_MODE", "prompt").lower()
    data_json: Dict[str, Any]
    if mode == "direct" or text.lstrip().startswith("{"):
        data_json = _extract_json_block(text)
    else:
        try:
            user_prompt = _CHAT_USER_TEMPLATE.format(
                feedback=text.strip(),
                objects=json.dumps(objects_ctx, ensure_ascii=False, indent=2),
            )
            data_json = post_chat_json_system_user(cfg, _CHAT_SYSTEM, user_prompt)
        except Exception as e_chat:
            try:
                base = (
                    os.environ.get("FEEDBACK_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("LLM_API_URL")
                    or _cfg_field(cfg, "api_url")
                    or _cfg_field(cfg, "base_url")
                )
                url = _compose_completions_url(base)
                api_key = read_api_key(cfg)
                if not api_key:
                    raise RuntimeError("Missing OpenAI API key.")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                prompt = _PROMPT_TEMPLATE.format(
                    feedback=text.strip(),
                    objects=json.dumps(objects_ctx, ensure_ascii=False, indent=2),
                )
                model = (
                    os.environ.get("FEEDBACK_MODEL")
                    or os.environ.get("LLM_FEEDBACK_MODEL")
                    or _cfg_field(cfg, "model")
                    or model_fallback
                )
                body = {
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": 512,
                }
                timeout_s = int(_cfg_field(cfg, "timeout_s", 60))
                resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
                resp.raise_for_status()
                payload = resp.json()
                text_out = ""
                if isinstance(payload, dict) and "choices" in payload and payload["choices"]:
                    c0 = payload["choices"][0]
                    text_out = (
                        c0.get("text")
                        or (c0.get("message") or {}).get("content")
                        or ""
                    )
                if not text_out:
                    text_out = payload.get("output_text", "")
                if not text_out:
                    raise RuntimeError(f"Unexpected completions payload: {payload}")
                data_json = _extract_json_block(text_out)
            except Exception as e_comp:
                raise RuntimeError(
                    f"Feedback labeling failed (chat:{e_chat}) (completions:{e_comp})"
                )
    if "present" not in data_json:
        raise ValueError("Model output missing 'present'.")
    model_present = bool(data_json["present"])
    if mode == "direct" or text.lstrip().startswith("{"):
        user_intent_present = model_present
        needs_confirmation = False
        confirmation_question = ""
    else:
        try:
            confirm_info = _confirm_present_with_llm(text, model_present, cfg)
            user_intent_present = bool(
                confirm_info.get("user_intent_present", model_present)
            )
            needs_confirmation = bool(
                confirm_info.get("needs_confirmation", False)
            )
            confirmation_question = str(
                confirm_info.get("confirmation_question", "") or ""
            ).strip()
        except Exception:
            user_intent_present = model_present
            needs_confirmation = False
            confirmation_question = ""
    present = model_present
    pairs = data_json.get("pairs") or []
    if not isinstance(pairs, list):
        raise ValueError("Model output 'pairs' must be a list.")
    if not pairs:
        pairs = _guess_pairs_from_text(text, objects)
        if not pairs:
            raise RuntimeError("No A/B pairs found.")
    rules_to_write: List[Dict[str, Any]] = []
    for p in pairs:
        A = p.get("A") or {}
        B = p.get("B") or {}
        A_text = str(
            A.get("text") or A.get("name") or A.get("kind") or ""
        ).strip()
        B_text = str(
            B.get("text") or B.get("name") or B.get("kind") or ""
        ).strip()
        if not A_text or not B_text:
            continue
        rule = {
            "user_id": user_id,
            "selectors": {
                "A": _normalize_selector({"by": "name", "value": A_text}, objects),
                "B": _normalize_selector({"by": "name", "value": B_text}, objects),
            },
            "directional": False,
            "relation": None,
            "condition_expr": None,
            "override": {
                "present": present,
                "model_present": model_present,
                "user_intent_present": user_intent_present,
                "needs_confirmation": needs_confirmation,
                "confirmation_question": confirmation_question,
            },
            "reason": text.strip(),
        }
        rules_to_write.append(rule)
    if not rules_to_write:
        raise RuntimeError("No valid rules to save.")
    store = store or PreferenceStore()
    return store.add_rules(rules_to_write)
