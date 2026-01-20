from __future__ import annotations

import os
import getpass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from services.llm import LLMConfig, read_api_key
from cbf.semantics_runtime import (
    analyze_scene_llm,
    instantiate_rules,
    classify_object_kind_llm,
)
from cbf.cbf_safety_metrics_llm import metric_hazard_pairings_cbf_objects_llm
from cbf.cbf_safety_metrics import ObjectState, Sphere
from .ui_helpers import DraggableCircle, set_arena_bounds
from cbf.preferences import PreferenceStore
from cbf.feedback_pipeline import label_and_store_feedback
from cbf.explanations import attach_explanations_to_hazards
from cbf.graphs.hazard_graph import (
    SceneHazardGraph,
    build_scene_hazard_graph_from_rules,
)
from cbf.graphs.feedback_graph import FeedbackGraph, build_feedback_graph_from_rules
from cbf.hazard_resolution import (
    resolve_semantic_hazards,
    HazardStatus,
    HazardMetadata,
)

from .scene_io import load_scene, to_llm_payload
from .rules import enforce_user_preferences_on_instantiated_rules


_ROOT = Path(__file__).resolve().parents[1]


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class InteractiveLLMApp:
    def __init__(self, scene_path: str):
        self.objects: List[ObjectState] = load_scene(scene_path)
        self.alpha_gain: float = 5.0
        self.scale_res: float = 0.05

        self.scene_version: int = 0
        self.rules_version: int = -1
        self._last_out: Optional[Dict[str, Any]] = None
        self._explain_map: Dict[Tuple[str, str], str] = {}

        self.hazard_graph: Optional[SceneHazardGraph] = None
        self.feedback_graph: Optional[FeedbackGraph] = None
        self.hazard_metadata: Dict[Tuple[str, str], HazardMetadata] = {}

        api_key = read_api_key(None)
        self.cfg = LLMConfig(api_key=api_key)

        self.fig = plt.figure(figsize=(15.2, 9.6))
        self.fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.86)

        gs = self.fig.add_gridspec(
            nrows=7,
            ncols=2,
            width_ratios=[3, 2],
            height_ratios=[24, 1, 1, 1, 1, 1, 1],
        )

        self.ax = self.fig.add_subplot(gs[0, 0])
        set_arena_bounds(self.ax, self.objects)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Interactive LLM-Driven Safety (drag circles)")

        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.axis("off")

        self.ax_btn_requery = self.fig.add_subplot(gs[1, 1])
        self.ax_btn_reset = self.fig.add_subplot(gs[2, 1])
        self.ax_add_box = self.fig.add_subplot(gs[3, 1])
        self.ax_btn_add = self.fig.add_subplot(gs[4, 1])
        self.ax_rm_box = self.fig.add_subplot(gs[5, 1])
        self.ax_btn_close = self.fig.add_subplot(gs[6, 1])

        self.btn_requery = Button(self.ax_btn_requery, label="Requery LLM")
        self.btn_reset = Button(self.ax_btn_reset, label="Reset Scene")
        self.btn_close = Button(self.ax_btn_close, label="Close Window")

        self.tb_add = TextBox(self.ax_add_box, label="Add name: ", initial="")
        self.btn_add = Button(self.ax_btn_add, label="Add")
        self.tb_rm = TextBox(
            self.ax_rm_box, label="Remove name (Enter): ", initial=""
        )

        self.btn_requery.on_clicked(self.on_requery_llm)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_close.on_clicked(self.on_close)
        self.btn_add.on_clicked(self.on_add_object)
        self.tb_add.on_submit(lambda _t: self.on_add_object(None))
        self.tb_rm.on_submit(lambda _t: self.on_remove_object())

        self.user_id = getpass.getuser()
        self.store = PreferenceStore()

        self.ax_fb_box = self.fig.add_axes([0.06, 0.90, 0.58, 0.055])
        self.tb_feedback = TextBox(
            self.ax_fb_box, label="Feedback: ", initial=""
        )
        self.ax_fb_submit = self.fig.add_axes([0.66, 0.90, 0.16, 0.055])
        self.btn_fb_submit = Button(self.ax_fb_submit, label="Submit & Requery")
        self.ax_fb_clear = self.fig.add_axes([0.84, 0.90, 0.12, 0.055])
        self.btn_fb_clear = Button(self.ax_fb_clear, label="Clear My Rules")

        self.btn_fb_submit.on_clicked(self.on_submit_feedback)
        self.tb_feedback.on_submit(lambda _t: self.on_submit_feedback(None))
        self.btn_fb_clear.on_clicked(self.on_clear_rules)

        self.initial_positions: Dict[str, np.ndarray] = {
            o.name: o.sphere.center.copy() for o in self.objects
        }
        self.draggables: List[DraggableCircle] = []

        self._info_lines: List[str] = []
        self._info_scroll: int = 0
        self._info_max_lines: int = 30

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll_info)

        main_pos = self.ax.get_position()
        bottom_margin = 0.02
        gap = 0.06
        graph_height = max(main_pos.y0 - gap - bottom_margin, 0.08)
        graph_bottom = bottom_margin
        graph_left = main_pos.x0
        graph_total_width = main_pos.width

        hazard_width = graph_total_width * 0.48
        feedback_width = graph_total_width * 0.48
        spacing = graph_total_width * 0.04

        self.ax_graph = self.fig.add_axes(
            [graph_left, graph_bottom, hazard_width, graph_height]
        )
        self.ax_graph.set_title("Semantic hazard graph")
        self.ax_graph.axis("off")

        self.ax_feedback_graph = self.fig.add_axes(
            [graph_left + hazard_width + spacing, graph_bottom, feedback_width, graph_height]
        )
        self.ax_feedback_graph.set_title("User feedback graph")
        self.ax_feedback_graph.axis("off")

        self._draw_all()
        self.update_info_panel()

    def _classify_all_kinds(self):
        for o in self.objects:
            try:
                k = classify_object_kind_llm(o.name, list(o.tags), self.cfg)
                o.kind = k or "object"
            except Exception as e:
                print(f"[Classify] {o.name}: {e} -> 'object'")
                o.kind = "object"

    def _rebuild_semantics(self):
        """Run LLM semantics, apply prefs, combine with rule graph."""
        risks = analyze_scene_llm(to_llm_payload(self.objects), self.cfg)

        rules_raw, crit_map_raw = instantiate_rules(self.objects, risks)

        user_rules = self.store.list_rules(self.user_id)
        rules_pref, crit_map_pref = enforce_user_preferences_on_instantiated_rules(
            self.objects, rules_raw, crit_map_raw, user_rules
        )

        self.feedback_graph = build_feedback_graph_from_rules(
            user_rules, self.cfg
        )

        resolved = resolve_semantic_hazards(
            objects=self.objects,
            semantic_rules=rules_pref,
            feedback_graph=self.feedback_graph,
        )

        self.rules = resolved.rules
        self.hazard_metadata = resolved.metadata
        # Node positions for graph linking + figure-level link artists
        self._hazard_node_positions: Dict[str, Tuple[float, float]] = {}
        self._feedback_node_positions: Dict[str, Tuple[float, float]] = {}
        self._link_artists: List[Any] = []

        self.crit_map = crit_map_pref

        self.hazard_graph = build_scene_hazard_graph_from_rules(
            self.objects, self.rules
        )

        if os.getenv("EXPLAIN_AUTO", "1") != "0":
            hazards_for_expl = [
                {
                    "selectors": {
                        "A": {"by": "name", "value": Aname},
                        "B": {"by": "name", "value": Bname},
                    }
                }
                for (_Ak, _Bk, _clr, _w, Aname, Bname) in self.rules
            ]
            try:
                only_present = os.getenv("EXPLAIN_ONLY_PRESENT", "1") != "0"
                enriched = attach_explanations_to_hazards(
                    hazards=hazards_for_expl,
                    objects=self.objects,
                    cfg=self.cfg,
                    only_present=only_present,
                )
                self._explain_map = {}
                for h in enriched:
                    sel = h.get("selectors", {})
                    a = ((sel.get("A") or {}).get("value") or "").strip()
                    b = ((sel.get("B") or {}).get("value") or "").strip()
                    ex = (h.get("explanation") or "").strip()
                    if a and b and ex:
                        self._explain_map[(a, b)] = ex
            except Exception as e:
                print(f"[Explain] failed: {e}")
                self._explain_map = {}

    def _compute_safety_now(self):
        return metric_hazard_pairings_cbf_objects_llm(
            objects=self.objects,
            alpha_gain=self.alpha_gain,
            scale_res=self.scale_res,
            rules=self.rules,
            critical_by_pair=self.crit_map,
            include_collision_baseline=True,
            baseline_weight=0.25,
            baseline_clearance_m=0.0,
            show_collision_for_labeled_pairs=True,
            collision_visual_only=True,
            label_semantic_entries=True,
        )

    # ------------------------------------------------------------------
    # Drawing / events
    # ------------------------------------------------------------------

    def _draw_all(self):
        for d in getattr(self, "draggables", []):
            d.remove()
        self.draggables = []
        for o in self.objects:
            self.draggables.append(
                DraggableCircle(self.ax, o, on_change=self.update_info_panel)
            )
        self.fig.canvas.draw_idle()

    def on_close(self, _evt):
        try:
            plt.close(self.fig)
        except Exception:
            pass

    def on_reset(self, _evt):
        for o in self.objects:
            if o.name in self.initial_positions:
                orig = self.initial_positions[o.name]
                o.sphere.center[0], o.sphere.center[1] = float(orig[0]), float(
                    orig[1]
                )
        for d in self.draggables:
            x, y = d.obj.sphere.center[0], d.obj.sphere.center[1]
            d.set_position(x, y)
        self.update_info_panel()

    def on_requery_llm(self, _evt):
        self._classify_all_kinds()
        self._rebuild_semantics()
        self.rules_version = self.scene_version
        self.update_info_panel()

    def on_add_object(self, _evt):
        name = self.tb_add.text.strip()
        if not name:
            return
        if any(o.name.lower() == name.lower() for o in self.objects):
            print(f"[Add] '{name}' already exists.")
            return
        obj = ObjectState(
            name=name,
            kind="object",
            sphere=Sphere(
                center=np.array([0.0, 0.0, 0.0], dtype=float),
                radius=0.05,
            ),
            tags=(),
        )
        self.objects.append(obj)
        self.initial_positions[name] = obj.sphere.center.copy()
        set_arena_bounds(self.ax, self.objects)
        self.draggables.append(
            DraggableCircle(self.ax, obj, on_change=self.update_info_panel)
        )
        self.scene_version += 1
        self.update_info_panel()
        self.tb_add.set_val("")

    def on_remove_object(self):
        name = self.tb_rm.text.strip()
        if not name:
            return
        idx = next(
            (i for i, o in enumerate(self.objects) if o.name.lower() == name.lower()),
            None,
        )
        if idx is None:
            print(f"[Remove] No object named '{name}'.")
            return
        removed = self.objects.pop(idx)
        self.initial_positions.pop(removed.name, None)
        di = next(
            (
                i
                for i, d in enumerate(self.draggables)
                if d.obj.name.lower() == name.lower()
            ),
            None,
        )
        if di is not None:
            self.draggables[di].remove()
            self.draggables.pop(di)
        set_arena_bounds(self.ax, self.objects)
        self.scene_version += 1
        self.update_info_panel()
        self.tb_rm.set_val("")

    def on_submit_feedback(self, _evt):
        text = self.tb_feedback.text.strip()
        if not text:
            print("[Feedback] empty input -> ignored")
            return
        try:
            n = label_and_store_feedback(
                text, self.objects, self.cfg, self.user_id, store=self.store
            )
            print(f"[Feedback] stored {n} rule(s) for user '{self.user_id}'.")
            self.tb_feedback.set_val("")
            self.on_requery_llm(None)
        except Exception as e:
            print(f"[Feedback] labeling failed: {e}")

    def on_clear_rules(self, _evt):
        n = self.store.clear_user(self.user_id)
        print(f"[Feedback] cleared {n} saved rule(s) for user '{self.user_id}'.")
        self.update_info_panel()

    def update_info_panel(self):
        scene_in_sync = self.scene_version == self.rules_version
        if scene_in_sync:
            out = self._compute_safety_now()
            self._last_out = out
        else:
            out = self._last_out or {
                "name": "hazard_pairings_llm",
                "metrics": [],
                "residual_min": float("inf"),
                "composite_risk": 0.0,
                "critical_violation": False,
                "critical_pairs": [],
                "safety_score_0_to_5": 5.0,
            }

        score = float(out.get("safety_score_0_to_5", 0.0))
        crit = bool(out.get("critical_violation", False))
        title = f"Safety Score: {score:.2f} / 5.00"
        if crit:
            title += "  —  CRITICAL ⚠️"
        if not scene_in_sync:
            title += "  —  (scene changed; press Requery LLM)"
        self.ax.set_title(title)

        for d in self.draggables:
            cx, cy = d.artist.center
            d.label.set_position((cx, cy))

        info_lines: List[str] = []

        info_lines.append("Rules (LLM + prefs):")
        if scene_in_sync:
            if not getattr(self, "rules", []):
                info_lines.append(" • (none)")
            else:
                for (Ak, Bk, clr, w, Aname, Bname) in self.rules:
                    key = _pair_key(Aname, Bname)
                    meta = getattr(self, "hazard_metadata", {}).get(key)
                    status_suffix = ""
                    if meta is not None and getattr(meta, "status", None) is not None:
                        try:
                            status_label = getattr(meta.status, "value", str(meta.status))
                        except Exception:
                            status_label = str(meta.status)
                        status_suffix = f" [{status_label}]"
                    info_lines.append(
                        f" • {Aname}({Ak}) → {Bname}({Bk}) | clr={clr:.3f}m, w={w:.2f}{status_suffix}"
                    )
                    if os.getenv("EXPLAIN_SHOW", "1") != "0":
                        ex = self._explain_map.get((Aname, Bname))
                        if ex:
                            info_lines.append(f"    why: {ex}")
        else:
            info_lines.append(" • (stale; press Requery LLM)")

        metrics = out.get("metrics", [])
        hazards = [m for m in metrics if m.get("channel") == "semantic"]
        collisions = [
            m
            for m in metrics
            if m.get("channel") == "collision"
            and not m.get("diagnostic_only", False)
        ]
        visual = [
            m
            for m in metrics
            if m.get("channel") == "collision"
            and m.get("diagnostic_only", False)
        ]

        info_lines.append("")
        info_lines.append(f"Hazard (semantic) [{len(hazards)}]:")
        for m in hazards:
            info_lines.append(f" • {m['name']}: risk={m['risk']:.3g}")

        info_lines.append("")
        info_lines.append(f"Collision baseline [{len(collisions)}]:")
        for m in collisions:
            info_lines.append(f" • {m['name']}: risk={m['risk']:.3g}")

        if visual:
            info_lines.append("")
            info_lines.append(f"Collision (visual) [{len(visual)}]:")
            for m in visual:
                info_lines.append(f" • {m['name']}: risk={m['risk']:.3g}")

        info_lines.append("")
        info_lines.append("Semantic hazard graph (objects):")
        if self.hazard_graph is None:
            info_lines.append(" • (not built yet; press Requery LLM)")
        elif not self.hazard_graph.edges:
            info_lines.append(" • (no semantic hazard edges)")
        else:
            for (a, b), edge in sorted(self.hazard_graph.edges.items()):
                key = _pair_key(edge.obj_a, edge.obj_b)
                meta = getattr(self, "hazard_metadata", {}).get(key)
                status_suffix = ""
                if meta is not None and getattr(meta, "status", None) is not None:
                    try:
                        status_label = getattr(meta.status, "value", str(meta.status))
                    except Exception:
                        status_label = str(meta.status)
                    status_suffix = f" [{status_label}]"
                info_lines.append(
                    f" • {edge.obj_a} --(clr={edge.soft_clearance_m:.3f}m, "
                    f"w={edge.weight:.2f})-- {edge.obj_b}{status_suffix}"
                )

        suppressed = [
            (a, b, meta)
            for (a, b), meta in getattr(self, "hazard_metadata", {}).items()
            if getattr(meta, "status", None) == HazardStatus.SUPPRESSED_BY_RULE
        ]
        if suppressed:
            info_lines.append("")
            info_lines.append(
                f"Suppressed semantic hazards by rules [{len(suppressed)}]:"
            )
            for (a, b, meta) in sorted(suppressed, key=lambda item: item[0:2]):
                decision = getattr(meta, "rule_decision", None)
                label = getattr(decision, "label", None)
                conf = getattr(decision, "confidence", None)
                sources = getattr(decision, "sources", None) or []

                if sources:
                    src_str = "; ".join(
                        f"({c1}, {c2})" for (c1, c2) in sources
                    )
                else:
                    src_str = "unknown rule edge(s)"

                parts = [f"{a} -- {b}"]
                if label:
                    parts.append(f"overridden as {label}")
                if isinstance(conf, (float, int)):
                    parts.append(f"conf≈{float(conf):.2f}")
                parts.append(f"by rule(s): {src_str}")

                info_lines.append(" • " + " | ".join(parts))

        residual_min = float(out.get("residual_min", float("inf")))
        composite_risk = float(out.get("composite_risk", 0.0))
        info_lines.append("")
        info_lines.append(f"residual_min: {residual_min:.3f}")
        info_lines.append(f"composite_risk: {composite_risk:.3g}")

        self._info_lines = info_lines
        max_start = max(len(self._info_lines) - self._info_max_lines, 0)
        self._info_scroll = max(0, min(self._info_scroll, max_start))

        self._render_info_panel()
        self._draw_hazard_graph()
        self._draw_feedback_graph()
        self._draw_graph_links()

    def _on_scroll_info(self, event):
        if event.inaxes is not self.ax_info:
            return
        step = -3 if event.button == "up" else 3
        if step < 0:
            self._info_scroll = max(self._info_scroll + step, 0)
        else:
            max_start = max(len(self._info_lines) - self._info_max_lines, 0)
            self._info_scroll = min(self._info_scroll + step, max_start)
        self._render_info_panel()

    def _render_info_panel(self):
        self.ax_info.clear()
        self.ax_info.axis("off")

        if not self._info_lines:
            self.fig.canvas.draw_idle()
            return

        start = self._info_scroll
        end = min(start + self._info_max_lines, len(self._info_lines))
        visible = self._info_lines[start:end]

        line_height = 0.035
        y = 0.98
        for line in visible:
            self.ax_info.text(
                0.02,
                y,
                line,
                ha="left",
                va="top",
                family="monospace",
                fontsize=9,
                transform=self.ax_info.transAxes,
            )
            y -= line_height
            if y < 0:
                break

        if len(self._info_lines) > self._info_max_lines:
            self.ax_info.text(
                0.98,
                0.02,
                f"{start + 1}-{end}/{len(self._info_lines)}",
                ha="right",
                va="bottom",
                fontsize=7,
                alpha=0.6,
                transform=self.ax_info.transAxes,
            )

        self.fig.canvas.draw_idle()

    def _draw_hazard_graph(self):
        self.ax_graph.clear()
        self.ax_graph.set_title("Semantic hazard graph")
        self.ax_graph.axis("off")

        hg = self.hazard_graph
        if hg is None or not hg.nodes:
            self.ax_graph.text(
                0.5,
                0.5,
                "(no semantic hazards)",
                ha="center",
                va="center",
                fontsize=8,
                transform=self.ax_graph.transAxes,
            )
            self._hazard_node_positions = {}
            self.fig.canvas.draw_idle()
            return

        nodes = sorted(hg.nodes)
        n = len(nodes)
        if n == 0:
            self._hazard_node_positions = {}
            self.fig.canvas.draw_idle()
            return

        radius = 0.9
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions: Dict[str, Tuple[float, float]] = {
            node: (radius * float(np.cos(a)), radius * float(np.sin(a)))
            for node, a in zip(nodes, angles)
        }
        self._hazard_node_positions = positions

        if hg.edges:
            weights = [e.weight for e in hg.edges.values()]
            w_min, w_max = min(weights), max(weights)
            if w_max > w_min:
                def norm(w):
                    return (w - w_min) / (w_max - w_min)
            else:
                def norm(_w):
                    return 0.5
        else:
            def norm(_w):
                return 0.5

        for (a, b), edge in hg.edges.items():
            if a not in positions or b not in positions:
                continue
            x0, y0 = positions[a]
            x1, y1 = positions[b]
            w_norm = norm(edge.weight)
            lw = 0.5 + 3.0 * w_norm
            self.ax_graph.plot(
                [x0, x1],
                [y0, y1],
                linewidth=lw,
                alpha=0.4,
                color="0.4",
                zorder=1,
            )

        meta = getattr(self, "hazard_metadata", {}) or {}
        for (name_a, name_b), m in meta.items():
            if getattr(m, "status", None) != HazardStatus.SUPPRESSED_BY_RULE:
                continue
            if name_a not in positions or name_b not in positions:
                continue

            x0, y0 = positions[name_a]
            x1, y1 = positions[name_b]

            self.ax_graph.plot(
                [x0, x1],
                [y0, y1],
                linewidth=1.0,
                linestyle="--",
                color="0.6",
                alpha=0.7,
                zorder=0.5,
            )

            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            dx, dy = x1 - x0, y1 - y0
            length = float(np.hypot(dx, dy))
            if length > 1e-6:
                ux, uy = dx / length, dy / length
                px, py = -uy, ux
            else:
                ux, uy = 1.0, 0.0
                px, py = 0.0, 1.0
            size = 0.08
            self.ax_graph.plot(
                [mx - size * ux, mx + size * ux],
                [my - size * uy, my + size * uy],
                linewidth=1.5,
                color="red",
                alpha=0.9,
                zorder=4,
            )
            self.ax_graph.plot(
                [mx - size * px, mx + size * px],
                [my - size * py, my + size * py],
                linewidth=1.5,
                color="red",
                alpha=0.9,
                zorder=4,
            )

        for node, (x, y) in positions.items():
            self.ax_graph.scatter(
                [x],
                [y],
                s=40,
                zorder=2,
                color="white",
                edgecolors="black",
            )
            angle = np.arctan2(y, x)
            lx = (radius + 0.16) * np.cos(angle)
            ly = (radius + 0.16) * np.sin(angle)
            self.ax_graph.text(
                lx,
                ly,
                node,
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )

        self.ax_graph.set_xlim(-1.4, 1.4)
        self.ax_graph.set_ylim(-1.4, 1.4)
        self.ax_graph.set_aspect("equal", "box")

        self.fig.canvas.draw_idle()


    def _draw_feedback_graph(self):
        self.ax_feedback_graph.clear()
        self.ax_feedback_graph.set_title("User feedback graph")
        self.ax_feedback_graph.axis("off")

        fg = self.feedback_graph
        if fg is None or not getattr(fg, "nodes", None):
            self.ax_feedback_graph.text(
                0.5,
                0.5,
                "(no feedback rules yet)",
                ha="center",
                va="center",
                fontsize=8,
                transform=self.ax_feedback_graph.transAxes,
            )
            self.fig.canvas.draw_idle()
            return
        
        nodes = sorted(fg.nodes)
        n = len(nodes)
        if n == 0:
            self._feedback_node_positions = {}
            self.fig.canvas.draw_idle()
            return

        radius = 0.9
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions: Dict[str, Tuple[float, float]] = {
            node: (radius * float(np.cos(a)), radius * float(np.sin(a)))
            for node, a in zip(nodes, angles)
        }
        self._feedback_node_positions = positions

        for (a, b), edge in fg.edges.items():
            if a not in positions or b not in positions:
                continue
            x0, y0 = positions[a]
            x1, y1 = positions[b]

            kinds = edge.kinds
            if "dangerous" in kinds:
                lw = 2.5
                ls = "-"
                color = "tab:red"
            elif "not_dangerous" in kinds:
                lw = 1.5
                ls = "-"
                color = "tab:green"
            elif "similar" in kinds:
                lw = 1.0
                ls = "--"
                color = "0.5"
            else:
                lw = 1.0
                ls = "-"
                color = "0.5"

            self.ax_feedback_graph.plot(
                [x0, x1],
                [y0, y1],
                linewidth=lw,
                linestyle=ls,
                color=color,
                alpha=0.7,
                zorder=1,
            )

            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            labels: List[str] = []
            if "dangerous" in kinds:
                labels.append("dangerous")
                if getattr(edge, "soft_clearance_m", None) is not None:
                    labels.append(f"clr={edge.soft_clearance_m:.2f}")
                if getattr(edge, "weight", None) is not None:
                    labels.append(f"w={edge.weight:.2f}")
            if "not_dangerous" in kinds and "dangerous" not in kinds:
                labels.append("not_dangerous")
            if "similar" in kinds and getattr(edge, "similarity_score", None) is not None:
                labels.append(f"sim={edge.similarity_score:.2f}")

            if labels:
                self.ax_feedback_graph.text(
                    mx,
                    my,
                    ",".join(labels),
                    ha="center",
                    va="center",
                    fontsize=7,
                    alpha=0.8,
                )

        for node, (x, y) in positions.items():
            self.ax_feedback_graph.scatter(
                [x],
                [y],
                s=40,
                zorder=2,
                color="white",
                edgecolors="black",
            )
            angle = np.arctan2(y, x)
            lx = (radius + 0.18) * np.cos(angle)
            ly = (radius + 0.18) * np.sin(angle)
            self.ax_feedback_graph.text(
                lx,
                ly,
                node,
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )

        self.ax_feedback_graph.set_xlim(-1.4, 1.4)
        self.ax_feedback_graph.set_ylim(-1.4, 1.4)
        self.ax_feedback_graph.set_aspect("equal", "box")

        self.fig.canvas.draw_idle()

    def _draw_graph_links(self):
        """Draw figure-level lines connecting scene-graph objects to
        rule-graph concepts when a rule overrides a semantic hazard.

        We only link suppressed hazards (SUPPRESSED_BY_RULE).
        """
        for artist in getattr(self, "_link_artists", []):
            try:
                artist.remove()
            except Exception:
                pass
        self._link_artists = []

        if not self._hazard_node_positions or not self._feedback_node_positions:
            self.fig.canvas.draw_idle()
            return

        meta = getattr(self, "hazard_metadata", {}) or {}
        if not meta:
            self.fig.canvas.draw_idle()
            return

        fig = self.fig
        ax_scene = self.ax_graph
        ax_rule = self.ax_feedback_graph

        for (name_a, name_b), m in meta.items():
            if getattr(m, "status", None) != HazardStatus.SUPPRESSED_BY_RULE:
                continue

            decision = getattr(m, "rule_decision", None)
            sources = getattr(decision, "sources", []) if decision is not None else []
            if not sources:
                continue

            if (
                name_a not in self._hazard_node_positions
                or name_b not in self._hazard_node_positions
            ):
                continue

            obj_a_pos = self._hazard_node_positions[name_a]
            obj_b_pos = self._hazard_node_positions[name_b]

            c1, c2 = sources[0]
            c1 = str(c1)
            c2 = str(c2)

            if (
                c1 not in self._feedback_node_positions
                or c2 not in self._feedback_node_positions
            ):
                continue

            con_a_pos = self._feedback_node_positions[c1]
            con_b_pos = self._feedback_node_positions[c2]

            oa_disp = ax_scene.transData.transform(obj_a_pos)
            ob_disp = ax_scene.transData.transform(obj_b_pos)
            ca_disp = ax_rule.transData.transform(con_a_pos)
            cb_disp = ax_rule.transData.transform(con_b_pos)

            oa_fig = fig.transFigure.inverted().transform(oa_disp)
            ob_fig = fig.transFigure.inverted().transform(ob_disp)
            ca_fig = fig.transFigure.inverted().transform(ca_disp)
            cb_fig = fig.transFigure.inverted().transform(cb_disp)

            for (x0, y0), (x1, y1) in ((oa_fig, ca_fig), (ob_fig, cb_fig)):
                line = plt.Line2D(
                    [x0, x1],
                    [y0, y1],
                    transform=fig.transFigure,
                    linewidth=0.8,
                    linestyle="--",
                    color="red",
                    alpha=0.6,
                    zorder=0.1,
                )
                fig.lines.append(line)
                self._link_artists.append(line)

        self.fig.canvas.draw_idle()

