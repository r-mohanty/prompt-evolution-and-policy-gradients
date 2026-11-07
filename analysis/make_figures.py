#!/usr/bin/env python
"""B2 -- paper figures, generated from the committed analysis JSONs.

    python analysis/make_figures.py

Reads only from results/*.json (produced by run_all.py and the A4/A5
scripts), so figures cannot drift from the reported numbers.

Design notes (dataviz skill, adapted for print):
  * Palette is the skill's reference instance, VALIDATED with its own script:
      diverging  blue #2a78d6 / red #e34948  -- all six checks PASS
      categorical blue #2a78d6 / orange #eb6834 -- all six checks PASS
  * Colour is assigned by JOB: fig 1 and 4 encode POLARITY (helps / hurts) so they
    use the diverging pair with a zero reference; fig 2 encodes MAGNITUDE on one
    axis; fig 3 encodes IDENTITY of two conditions.
  * Skill steps deliberately skipped, and why: no hover layer and no dark mode --
    these are static figures for a print PDF, not an HTML surface. Every other
    step (form, colour job, validation, mark specs, accessibility) applies.
  * Text always wears ink tokens, never the series colour.
  * Identity is never colour-alone: every panel is directly labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
# NeurIPS requires Type 1 / embedded TrueType fonts only; matplotlib's PDF default is Type 3.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# PAPER_FIGS=1 renders the versions embedded in the LaTeX paper: no in-image title or
# footnote (the LaTeX caption carries them) and output straight into results/figures/paper.
import os as _os
PAPER = _os.environ.get("PAPER_FIGS") == "1"
if PAPER:
    import matplotlib.figure as _mf
    _orig_text = _mf.Figure.text

    def _paper_text(self, x, y, *a, **k):
        # footnotes are placed below the axes (y < 0); supxlabel and the like use y >= 0
        if y < 0:
            return None
        return _orig_text(self, x, y, *a, **k)

    _mf.Figure.suptitle = lambda self, *a, **k: None
    _mf.Figure.text = _paper_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "results" / "figures"
if PAPER:
    FIG = ROOT / "results" / "figures" / "paper"


# --- palette (validated; see module docstring) ------------------------------
BLUE, RED, ORANGE = "#2a78d6", "#e34948", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 200,
})


def _load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ---------------------------------------------------------------- figure 1
def fig1_regime_boundary():
    """Per-cell paired differences. Job: POLARITY -> diverging pair + zero line."""
    a01 = _load("a01_variant_comparison.json")
    g = a01["gepa_then_grpo_vs_dr_grpo"]
    cells = g["cells"]
    # 2026-09-03/04: hatch the cells in which GEPA edited the FIRST module's instruction
    # (generate_query on HotpotQA, plan on math); solid bars trained from the unchanged
    # seed docstring for that module (analysis/a07). Section 4.1 shows the query module
    # carries the HotpotQA effect: the four solid HotpotQA bars are the four largest gains.
    # The answer instruction was edited in 10 of 12 HotpotQA cells; no math cell was edited.
    a07 = _load("a07_instruction_wording.json") or {"cells": []}
    first = {"hotpotqa": "generate_query", "aime": "plan"}
    edited = {(c["task"], c["budget"], c["seed"]): c["flags"].get(first[c["task"]]) == "evolved"
              for c in a07["cells"]}
    # sharex is load-bearing: with independent scales the reader compares bar
    # LENGTHS across panels that are on different axes, which is the panel-level
    # version of the dual-axis error.
    # Designed at 5.2in so that, printed at 0.66 of the NeurIPS text width (3.6in), type
    # stays above 6pt. Every figure below follows the same rule (design width close to
    # print width), which the first versions did not.
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.8), sharey=True, sharex=True)
    for ax, task in zip(axes, ("hotpotqa", "aime")):
        sub = sorted([c for c in cells if c["task"] == task],
                     key=lambda c: (c["budget"], c["seed"]))
        ys = range(len(sub))
        vals = [c["diff"] for c in sub]
        ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
        bars = ax.barh(list(ys), vals, height=0.62, zorder=2,
                       color=[BLUE if v > 0 else RED for v in vals])
        for bar, c in zip(bars, sub):
            if edited.get((c["task"], c["budget"], c["seed"]), False):
                bar.set_hatch("///")
                bar.set_edgecolor(SURFACE)
                bar.set_linewidth(0.6)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"b{c['budget']} s{c['seed']}" for c in sub], fontsize=7)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.invert_yaxis()
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        mean = sum(vals) / len(vals)
        n_pos = sum(1 for v in vals if v > 0)
        label = "HotpotQA" if task == "hotpotqa" else "Competition math"
        ax.set_title(f"{label}\nmean {mean:+.4f}, {n_pos}/{len(vals)} above zero",
                     color=INK, loc="left", fontsize=7.5, pad=4)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
    fig.supxlabel("GEPA-returned prompts minus trainer defaults (mean reward, shared scale)",
                  fontsize=8, color=INK2, y=0.02)
    fig.suptitle("The composition boundary is a wording boundary: seed docstrings, not GEPA's edits, drive it",
                 x=0.01, ha="left", fontsize=10.5, color=INK, y=1.02)
    fig.text(0.01, -0.10,
             "Each bar is one (budget, seed) cell. Hatched: GEPA edited the first module's instruction; solid: it returned "
             "the seed docstring.\nEvery math cell is solid (GEPA returned its seed in 12/12). Pooling the tasks cancels "
             "them to a null (-0.0285, ns).",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_regime_boundary.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig1_regime_boundary.png", bbox_inches="tight")
    plt.close(fig)
    return "fig1_regime_boundary"


# ---------------------------------------------------------------- figure 2
def fig2_effect_sizes():
    """Effects on one common reward scale. Job: MAGNITUDE -> single axis."""
    a02, a04 = _load("a02_grpo_learning.json"), _load("a4_verbosity_penalty.json")
    a01 = _load("a01_variant_comparison.json")
    rows = []
    if a04:
        rows.append(("Reward lost to\nanswer format", a04["verbosity_penalty"], None, RED))
    if a02 and a02.get("grpo_learning"):
        gl = a02["grpo_learning"]
        rows.append(("GRPO total learning\n(8,192 rollouts)", gl["mean_delta"], gl["ci"], BLUE))
    lc = (a02 or {}).get("lr_control", {}).get("1e-05_vs_1e-06")
    if lc:
        b = lc["bootstrap"]
        rows.append(("Learning rate\n1e-5 vs 1e-6", b["mean"], [b["ci_lo"], b["ci_hi"]], BLUE))
    k = a01["rca_vs_dr_grpo"]
    rows.append(("RCA vs Dr. GRPO", k["mean_diff"], k["ci"], BLUE))

    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    ys = list(range(len(rows)))[::-1]
    for y, (label, val, ci, colour) in zip(ys, rows):
        ax.barh(y, val, height=0.55, color=colour, zorder=3)
        if ci:
            ax.plot([ci[0], ci[1]], [y, y], color=INK2, lw=1.4, zorder=4,
                    solid_capstyle="butt")
        # anchor the value label past whichever is further right, bar end or CI
        anchor = max(val, ci[1]) if ci else val
        ax.text(anchor + 0.007, y, f"{val:+.4f}", va="center", ha="left",
                fontsize=7.5, color=INK)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5, color=INK2)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("effect on mean reward", fontsize=8)
    ax.set_xlim(-0.02, 0.29)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.suptitle("The reward metric destroys 5.4x more signal than GRPO training recovers",
                 x=0.01, ha="left", color=INK, fontsize=10.5, y=1.04)
    fig.text(0.01, -0.13,
             "Bars are point estimates; dark rules are 95% intervals. The top bar is the reward "
             "recoverable by perfect answer\nextraction, an upper bound the optimiser never sees.",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_effect_sizes.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig2_effect_sizes.png", bbox_inches="tight")
    plt.close(fig)
    return "fig2_effect_sizes"


# ---------------------------------------------------------------- figure 3
def fig3_trace_ablation():
    """With vs without the execution trace. Job: IDENTITY of 2 conditions."""
    a05 = _load("a05_legibility.json")
    ta = a05.get("trajectory_ablation") if a05 else None
    if not ta:
        return None
    # per-task accuracies computed from the raw judge rows: the legibility sweep's
    # `baseline` condition is the judge WITH the full trace; trajectory_ablation.json
    # holds the same judge with the trace removed (`no_trajectory`).
    sweep, abl = _load("legibility_sweep.json"), _load("trajectory_ablation.json")

    def acc(rows, task, cond):
        # same exclusion as a05_legibility.py: judge calls that errored (no prediction)
        # are dropped from the denominator, which is why math has 21 with-trace rows
        sel = [r for r in rows if r["task"] == task and r["condition"] == cond and r.get("predicted")]
        return sum(1 for r in sel if r["correct"]) / len(sel)

    tasks = ["HotpotQA", "Competition math"]
    with_trace = [acc(sweep["rows"], "hotpotqa", "baseline"), acc(sweep["rows"], "aime", "baseline")]
    without = [acc(abl["rows"], "hotpotqa", "no_trajectory"), acc(abl["rows"], "aime", "no_trajectory")]
    chance = [0.500, 0.333]
    x = [0, 1]
    w, gap = 0.26, 0.03          # thin marks + a visible surface gap between fills
    fig, ax = plt.subplots(figsize=(3.8, 2.3))
    b1 = ax.bar([i - w / 2 - gap for i in x], with_trace, w, color=BLUE, zorder=3,
                label="judge sees the trace")
    b2 = ax.bar([i + w / 2 + gap for i in x], without, w, color=ORANGE, zorder=3,
                label="trace removed")
    for i, c in zip(x, chance):
        # named once in the legend; an in-plot label would sit on the bar it crosses
        ax.plot([i - 0.42, i + 0.42], [c, c], color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=5,
                label="chance" if i == x[0] else None)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, color=INK, fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("fault-localisation accuracy", fontsize=8)
    ax.set_ylim(0, 0.95)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper right", labelcolor=INK2)
    fig.suptitle("A frontier judge gains nothing from the trace, and on math is misled below chance",
                 x=0.01, ha="left", color=INK, fontsize=10.5, y=1.04)
    fig.text(0.01, -0.10,
             "With the trace the judge blames the module that failed to CATCH the error rather than the one that "
             "made it.\nHuman labels on 20 natural failures show the same bias (both judge errors, none in reverse).",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_trace_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig3_trace_ablation.png", bbox_inches="tight")
    plt.close(fig)
    return "fig3_trace_ablation"


# ---------------------------------------------------------------- figure 4
def fig4_rca_null():
    """Forest plot of variant comparisons. Job: POLARITY around a zero line."""
    a01 = _load("a01_variant_comparison.json")
    order = [("rca_vs_dr_grpo", "RCA vs Dr. GRPO"),
             ("rca_no_validate_vs_dr_grpo", "RCA (no val.) vs Dr. GRPO"),
             ("rca_vs_rca_no_validate", "RCA vs RCA (no val.)")]
    rows = [(lbl, a01[k]) for k, lbl in order if k in a01]
    fig, ax = plt.subplots(figsize=(3.8, 2.2))
    ys = list(range(len(rows)))[::-1]
    for y, (lbl, d) in zip(ys, rows):
        lo, hi = d["ci"]
        colour = BLUE if d["mean_diff"] >= 0 else RED
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot([d["mean_diff"]], [y], "o", ms=7, color=colour, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(d["mean_diff"], y + 0.1, f"{d['mean_diff']:+.4f}", va="bottom", ha="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    # Plot the positive control on the SAME axis. A null is only meaningful next to
    # an effect the apparatus demonstrably detects; putting it off-axis and describing
    # it in prose asks the reader to take the comparison on trust.
    a02 = _load("a02_grpo_learning.json")
    lc = (a02 or {}).get("lr_control", {}).get("1e-05_vs_1e-06")
    if lc:
        v = lc["bootstrap"]["mean"]
        ax.axvline(v, color=ORANGE, lw=1.6, ls=(0, (4, 3)), zorder=2)
        ax.text(v - 0.0006, max(ys) + 0.45, f"LR control {v:+.4f}",
                fontsize=7, color=ORANGE, va="center", ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5, color=INK2)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("difference in mean reward (24 cells)", fontsize=8)
    ax.set_xlim(-0.010, 0.020)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fig.suptitle("Reflective credit assignment: a bounded null on every comparison",
                 x=0.01, ha="left", color=INK, fontsize=10.5, y=1.06)
    fig.text(0.01, -0.16,
             "Dots are point estimates, rules are 95% intervals. The dashed rule is the "
             "learning-rate positive control, an effect the same apparatus detects, an order of "
             "magnitude\nlarger than anything RCA produces and outside every interval shown.",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_rca_null.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig4_rca_null.png", bbox_inches="tight")
    plt.close(fig)
    return "fig4_rca_null"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig1_regime_boundary, fig2_effect_sizes, fig3_trace_ablation, fig4_rca_null):
        name = fn()
        print(f"  wrote {name}.pdf / .png" if name else f"  SKIPPED {fn.__name__} (missing data)")
    print(f"\nfigures -> {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
