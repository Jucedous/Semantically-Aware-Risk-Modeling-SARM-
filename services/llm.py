from __future__ import annotations
import json, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

def _cfg_field(cfg: Any, name: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def read_api_key(cfg: Any = None) -> str:
    for env in ("OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    v = _cfg_field(cfg, "api_key")
    if v:
        return v
    root = _project_root()
    candidates = [
        root / "config" / "openai_key.txt",
        root / "cbf" / "openai_key.txt",
        root / "cbf_extracted" / "config" / "openai_key.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                t = p.read_text(encoding="utf-8").strip()
                if not t:
                    continue
                if "OPENAI_API_KEY" in t and "=" in t:
                    return t.split("=", 1)[1].strip().strip('"').strip("'")
                return t
        except Exception:
            continue
    raise RuntimeError("Missing OpenAI API key.")

@dataclass
class LLMConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    api_url: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.0
    timeout_s: int = 60

def post_chat_json_messages(cfg: Any, messages: List[Dict[str, str]], model_env: Optional[str] = None) -> Dict[str, Any]:
    api_key = _cfg_field(cfg, "api_key") or read_api_key(cfg)
    api_url = _cfg_field(cfg, "api_url", "https://api.openai.com/v1/chat/completions")
    model = None
    if model_env:
        m = os.environ.get(model_env)
        if m:
            model = m.strip()
    if not model:
        model = _cfg_field(cfg, "model", "gpt-4o-mini")
    temperature = float(_cfg_field(cfg, "temperature", 0.0))
    timeout_s = int(_cfg_field(cfg, "timeout_s", 60))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": temperature,
        "top_p": float(_cfg_field(cfg, "top_p", 1.0)),
        "presence_penalty": float(_cfg_field(cfg, "presence_penalty", 0.0)),
        "frequency_penalty": float(_cfg_field(cfg, "frequency_penalty", 0.0)),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }

    use_disk_cache = str(os.environ.get("LLM_DISK_CACHE", "")).strip().lower() in ("1", "true", "yes")
    cache_file = None
    if use_disk_cache:
        try:
            import hashlib
            cache_root = os.environ.get("LLM_CACHE_DIR")
            cache_dir = Path(cache_root) if cache_root else (_project_root() / "models" / "llm_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            key_payload = {
                "model": body.get("model"),
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "presence_penalty": body.get("presence_penalty"),
                "frequency_penalty": body.get("frequency_penalty"),
                "response_format": body.get("response_format"),
                "messages": body.get("messages"),
            }
            key_raw = json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            h = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
            cache_file = cache_dir / f"{h}.json"

            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    # ignore cache read errors and fall back to live call
                    cache_file = None
        except Exception:
            cache_file = None

    r = requests.post(api_url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    data = json.loads(content)

    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return data

def post_chat_json_system_user(cfg: Any, system_prompt: str, user_prompt: str, model_env: Optional[str] = None) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return post_chat_json_messages(cfg, messages, model_env=model_env)

def norm_base_v1(base: Optional[str]) -> Optional[str]:
    if not base:
        return None
    b = base.strip().rstrip("/")
    b = re.sub(r"/chat/completions/?$", "", b)
    b = re.sub(r"/completions/?$", "", b)
    if not re.search(r"/v\d+($|/)", b):
        b += "/v1"
    return b.rstrip("/")

def compose_completions_url(base: Optional[str]) -> str:
    base = norm_base_v1(base) or "https://api.openai.com/v1"
    return base + "/completions"
