from __future__ import annotations
from typing import Any, Dict

LABEL_KW: Dict[str, Any] = dict(color="black", fontsize=10, ha="center", va="center")
TITLE_KW: Dict[str, Any] = dict(color="black", fontsize=13, weight="bold")
INFO_KW: Dict[str, Any] = dict(color="black", fontsize=10, family="monospace", ha="left", va="top")

ARENA_FACE_COLOR = (0.96, 0.97, 0.99)
ARENA_EDGE_COLOR = (0.75, 0.78, 0.85)
GRID_ALPHA = 0.35

DEFAULT_OBJECT_STYLE: Dict[str, Dict[str, Any]] = {
    "electronic": dict(ec="black", lw=1.5, fc=(0.80, 0.90, 1.00), alpha=0.9),
    "liquid": dict(ec="black", lw=1.5, fc=(0.65, 0.86, 0.92), alpha=0.9),
    "human": dict(ec="black", lw=1.5, fc=(0.90, 0.75, 0.75), alpha=0.9),
    "sharp": dict(ec="black", lw=1.5, fc=(0.95, 0.90, 0.70), alpha=0.9),
    "fragile": dict(ec="black", lw=1.5, fc=(0.92, 0.85, 0.96), alpha=0.9),
    "heavy": dict(ec="black", lw=1.5, fc=(0.85, 0.85, 0.85), alpha=0.9),
    "object": dict(ec="black", lw=1.5, fc=(0.88, 0.88, 0.95), alpha=0.9),
}