"""Render eval results as a scored report against the project's targets.

Turns results/eval_summary.json into something you can read at a glance and paste into
a write-up, instead of a flat wall of key: value lines.

    python scripts/report.py
    python scripts/report.py --json results/eval_summary.json --markdown report.md
    python scripts/report.py --compare results/run_a.json results/run_b.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from shadowweave.console import C, bar, header, rule, status, table  # noqa: E402
from shadowweave.utils import load_config  # noqa: E402

# (label, metric key, target, comparison) — the targets from the project spec.
TARGETS = [
    ("World model IOU @ 5s", "iou_5s", 0.60, "ge"),
    ("Shadow IOU @ 5s", "shadow_iou_5s", 0.40, "ge"),
    ("Collision rate", "collision_rate_per_step", 0.10, "le"),
    ("Falling-object lead time", "falling_lead_time_s", 3.0, "ge"),
    ("Pipeline latency p95 (ms)", "latency_p95_ms", 100.0, "le"),
    ("Calibration error", "calibration_error", 0.10, "le"),
]


def _meets(value: float, target: float, cmp: str) -> bool:
    return value >= target if cmp == "ge" else value <= target


def render(summary: dict, cfg) -> str:
    out: list[str] = [header("ShadowWeave — evaluation report")]

    # ── targets ────────────────────────────────────────────────────────
    rows = []
    for label, key, target, cmp in TARGETS:
        if key not in summary:
            rows.append([label, f"{C.dim}not measured{C.reset}",
                         f"{'≥' if cmp == 'ge' else '≤'} {target:g}", status(None)])
            continue
        v = summary[key]
        rows.append([label, f"{v:.3f}", f"{'≥' if cmp == 'ge' else '≤'} {target:g}",
                     status(_meets(v, target, cmp))])
    out += ["", rule("Targets"), "",
            table(rows, ["metric", "value", "target", ""], aligns=["l", "r", "r", "r"])]

    # ── per-horizon vs baselines ───────────────────────────────────────
    horizons = list(cfg.world_model.prediction_horizons)
    rows = []
    for h in horizons:
        label = f"{h:g}s"
        if f"iou_{label}" not in summary:
            continue
        model = summary[f"iou_{label}"]
        persist = summary.get(f"baseline_persistence_iou_{label}")
        gain = summary.get(f"model_gain_over_best_baseline_{label}")
        rows.append([
            label,
            f"{model:.3f}",
            "  –  " if persist is None else f"{persist:.3f}",
            "  –  " if gain is None else
            f"{C.green if gain > 0 else C.red}{gain:+.3f}{C.reset}",
            f"{summary.get(f'shadow_iou_{label}', float('nan')):.3f}",
            bar(model),
        ])
    if rows:
        out += ["", rule("Prediction quality by horizon"), "",
                table(rows, ["horizon", "model", "persist", "gain", "shadow", ""],
                      aligns=["l", "r", "r", "r", "r", "l"])]
        missing = [f"{h:g}s" for h in horizons if f"iou_{h:g}s" not in summary]
        if missing:
            out.append(f"\n  {C.yellow}note{C.reset}: {', '.join(missing)} not scored — "
                       f"raise eval.steps_per_episode above horizon x fps")
        out.append(f"\n  {C.dim}gain = model minus the strongest baseline. Negative means "
                   f"the model is not\n  beating a trivial predictor, whatever its raw IOU."
                   f"{C.reset}")

    # ── falling objects ────────────────────────────────────────────────
    if "falling_detection_rate" in summary:
        det = summary["falling_detection_rate"]
        fa = summary.get("falling_false_alarm_rate")
        margin = summary.get("falling_detection_margin")
        rows = [["detection rate", f"{det:.3f}", bar(det)]]
        if fa is not None:
            rows.append(["false-alarm rate", f"{fa:.3f}", bar(fa)])
        if margin is not None:
            rows.append(["margin (det - fa)", f"{margin:+.3f}", bar(max(margin, 0.0))])
        if "falling_lead_time_s" in summary:
            rows.append(["mean lead time", f"{summary['falling_lead_time_s']:.2f}s", ""])
        out += ["", rule("Falling-object anticipation"), "",
                table(rows, ["metric", "value", ""], aligns=["l", "r", "l"]),
                f"\n  {C.dim}a high detection rate with a high false-alarm rate means the "
                f"model flags\n  everything, not that it anticipates anything.{C.reset}"]

    # ── navigation ─────────────────────────────────────────────────────
    nav = [(k, summary[k]) for k in
           ("collision_rate", "collision_rate_per_step", "path_efficiency", "n_episodes")
           if k in summary]
    if nav:
        out += ["", rule("Navigation"), "",
                table([[k, f"{v:.3f}"] for k, v in nav], ["metric", "value"])]

    lat = [(k, summary[k]) for k in ("latency_p50_ms", "latency_p95_ms") if k in summary]
    if lat:
        out += ["", rule("Latency"), "",
                table([[k, f"{v:.2f} ms"] for k, v in lat], ["metric", "value"])]

    passed = sum(1 for _, k, t, c in TARGETS if k in summary and _meets(summary[k], t, c))
    measured = sum(1 for _, k, _, _ in TARGETS if k in summary)
    out += ["", rule(""),
            f"  {C.bold}{passed}/{measured} measured targets met{C.reset}"
            f"  {C.dim}({len(TARGETS) - measured} not measured){C.reset}", ""]
    return "\n".join(out)


def render_markdown(summary: dict, cfg) -> str:
    lines = ["# ShadowWeave evaluation report", "", "## Targets", "",
             "| metric | value | target | status |", "|---|---|---|---|"]
    for label, key, target, cmp in TARGETS:
        if key not in summary:
            lines.append(f"| {label} | not measured | {'≥' if cmp=='ge' else '≤'} {target:g} | — |")
            continue
        v = summary[key]
        ok = "✅" if _meets(v, target, cmp) else "❌"
        lines.append(f"| {label} | {v:.3f} | {'≥' if cmp=='ge' else '≤'} {target:g} | {ok} |")

    lines += ["", "## Prediction quality by horizon", "",
              "| horizon | model IOU | persistence | gain | shadow IOU |", "|---|---|---|---|---|"]
    for h in cfg.world_model.prediction_horizons:
        label = f"{h:g}s"
        if f"iou_{label}" not in summary:
            continue
        g = summary.get(f"model_gain_over_best_baseline_{label}")
        lines.append(
            f"| {label} | {summary[f'iou_{label}']:.3f} | "
            f"{summary.get(f'baseline_persistence_iou_{label}', float('nan')):.3f} | "
            f"{'n/a' if g is None else f'{g:+.3f}'} | "
            f"{summary.get(f'shadow_iou_{label}', float('nan')):.3f} |")

    lines += ["", "## All metrics", "", "| key | value |", "|---|---|"]
    lines += [f"| `{k}` | {v:.4f} |" for k, v in sorted(summary.items())]
    return "\n".join(lines) + "\n"


def render_compare(paths: list[str], cfg) -> str:
    runs = []
    for p in paths:
        runs.append((pathlib.Path(p).stem, json.loads(pathlib.Path(p).read_text())))
    keys = sorted({k for _, s in runs for k in s})
    rows = []
    for k in keys:
        row = [k]
        vals = [s.get(k) for _, s in runs]
        best = max((v for v in vals if v is not None), default=None)
        for v in vals:
            if v is None:
                row.append("  –  ")
            elif v == best and len([x for x in vals if x is not None]) > 1:
                row.append(f"{C.green}{v:.3f}{C.reset}")
            else:
                row.append(f"{v:.3f}")
        rows.append(row)
    return "\n".join([header("Run comparison"), "",
                      table(rows, ["metric"] + [n for n, _ in runs])])


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a ShadowWeave evaluation report")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--markdown", type=str, default=None, help="also write a markdown file")
    ap.add_argument("--compare", nargs="+", default=None, help="compare several result files")
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)

    if args.compare:
        print(render_compare(args.compare, cfg))
        return

    path = pathlib.Path(args.json or (pathlib.Path(cfg.eval.results_dir) / "eval_summary.json"))
    if not path.exists():
        print(f"No results at {path}. Run:  python -m shadowweave.eval.run_eval")
        raise SystemExit(1)

    summary = json.loads(path.read_text())
    print(render(summary, cfg))

    if args.markdown:
        out = pathlib.Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(summary, cfg))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
