import sys
from pathlib import Path
import argparse

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cli.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive LLM safety (with LLM-based feedback pipeline)")
    parser.add_argument("scene", nargs="?", default=str(_ROOT / "models" / "scene1.json"))
    args = parser.parse_args()
    main(args.scene)
