from __future__ import annotations
import json, time, uuid, os
from pathlib import Path
from typing import Any, Dict, List, Optional

def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def _default_path() -> Path:
    return (_root_dir() / "models" / "user_prefs.json").resolve()

def _profile_path(profile: str) -> Path:
    return (_root_dir() / "models" / f"user_prefs_{profile}.json").resolve()

class PreferenceStore:
    def __init__(self, path: Optional[Path] = None):
        mode = os.getenv("PREFS_MODE", "default").strip().lower()
        path_env = os.getenv("PREFS_PATH", "").strip()
        profile = os.getenv("PREFS_PROFILE", "").strip()
        if path is not None:
            self.path = Path(path).resolve()
            self.enabled = True
        elif mode in ("off", "none", "disabled", "0", "false", "no"):
            self.path = _default_path()
            self.enabled = False
        elif path_env:
            self.path = Path(path_env).expanduser().resolve()
            self.enabled = True
        elif profile:
            self.path = _profile_path(profile)
            self.enabled = True
        else:
            self.path = _default_path()
            self.enabled = True

    def _ensure_parent(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        try:
            if not self.path.exists():
                return {"rules": []}
            txt = self.path.read_text(encoding="utf-8").strip()
            if not txt:
                return {"rules": []}
            obj = json.loads(txt)
            if not isinstance(obj, dict):
                return {"rules": []}
            obj.setdefault("rules", [])
            return obj
        except Exception:
            return {"rules": []}

    def _write(self, obj: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure_parent()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add_rules(self, rules: List[Dict[str, Any]]) -> int:
        if not self.enabled:
            return 0
        db = self._read()
        for r in rules:
            r.setdefault("id", str(uuid.uuid4()))
            r.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            db.setdefault("rules", []).append(r)
        self._write(db)
        return len(rules)

    def list_rules(self, user_id: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return [r for r in self._read().get("rules", []) if r.get("user_id") == user_id]

    def clear_user(self, user_id: str) -> int:
        if not self.enabled:
            return 0
        db = self._read()
        before = len(db.get("rules", []))
        db["rules"] = [r for r in db.get("rules", []) if r.get("user_id") != user_id]
        self._write(db)
        return before - len(db["rules"])
