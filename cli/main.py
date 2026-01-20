from __future__ import annotations
import sys
from pathlib import Path
import argparse

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.cli import main as _run_app

def _default_scene_path() -> str:
    return str(_ROOT / "models" / "scene1.json")

def main(scene_path: str | None = None):
    path = scene_path or _default_scene_path()
    _run_app(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive LLM safety runner")
    parser.add_argument("scene", nargs="?", default=_default_scene_path())
    args = parser.parse_args()
    main(args.scene)
