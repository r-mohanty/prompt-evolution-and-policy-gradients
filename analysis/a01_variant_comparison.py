#!/usr/bin/env python
"""A01 -- the paper's central result: does reflective credit assignment (RCA)
beat uniform-credit Dr. GRPO?

Design (cluster-robust): training steps within a run are autocorrelated, so
treating them as independent observations inflates t wildly. Instead we reduce
each run to a single mean reward, form ONE difference per
(task, budget, seed) cell, and run a one-sample t-test across cells. Each cell
contributes one degree of freedom; n = number of cells.

Run:  python analysis/a01_variant_comparison.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_runs import ROOT, load_runs, mean, paired_test  # noqa: E402

# The main grid holds these hyperparameters fixed; LR-sweep runs are analysed
# separately in a02 and must not contaminate this comparison.
BASE = {"lr": 1e-6, "gs": 8, "kl": 0.04}
# The pre-registered grid. Anything else (e.g. a stray budget=32, 4-step Phase-1
# smoke test found in the archives) is excluded: it never pairs, but it would
# otherwise contaminate the absolute per-variant means.
GRID_BUDGETS = {128, 256, 512, 2048, 8192}

# RESOLVED 2026-09-02 (A1). All 12 HotpotQA `gepa_then_grpo` cells have been
# re-run with the answer-span rule on BOTH reward paths; the AIME arm was never
# affected (it extracts via oxed{}). The cohort is now uniform and the
# comparison is valid. Each cell's provenance is pinned in extract_runs.py so a
# stale format-bugged copy on any pod volume cannot be reinstated.
GEPA_COHORT_MIXED = False
COMPARISONS = [
    ("rca", "dr_grpo"),
    ("rca_no_validate", "dr_grpo"),
    ("gepa_then_grpo", "dr_grpo"),
    ("rca", "rca_no_validate"),
    # On-policy GEPA arm (2026-09-03): budget-2048 cells only, so these pair on 6 cells.
    # Per-task detail lives in a06_onpolicy_gepa.py.
    ("gepa_onpolicy_then_grpo", "dr_grpo"),
    ("gepa_onpolicy_then_grpo", "gepa_then_grpo"),
]


def main() -> None:
    runs = [r for r in load_runs()
            if r.rewards and r.lr == BASE["lr"] and r.gs == BASE["gs"] and r.kl == BASE["kl"]
            and r.budget in GRID_BUDGETS]

    by: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in runs:
        by[r.cell][r.variant] = r.mean_reward

    print("=" * 96)
    print("A01 -- CLUSTER-ROBUST VARIANT COMPARISON")
    print("     one mean-difference per (task,budget,seed) cell; t-test across cells")
    print("=" * 96)
    print(f"loaded {len(runs)} runs at lr={BASE['lr']:g} gs={BASE['gs']} kl={BASE['kl']}, "
          f"{len(by)} distinct cells\n")

    out = {}
    for a, b in COMPARISONS:
        diffs, cells = [], []
        for cell, d in sorted(by.items()):
            if a in d and b in d:
                diffs.append(d[a] - d[b])
                cells.append(cell)
        if not diffs:
            print(f"{a} vs {b}: no paired cells")
            continue
        res = paired_test(diffs)
        label = f"{a} vs {b}"
        print(res.line(label))
        if GEPA_COHORT_MIXED and "gepa_then_grpo" in (a, b):
            print("      ^^ INVALID: mixes format-bugged runs with 1 patched re-run. "
                  "Meaningless until A1 completes.")
        out[f"{a}_vs_{b}"] = {
            "n_cells": res.n, "mean_diff": res.mean_diff, "se": res.se, "t": res.t,
            "ci": [res.ci_lo, res.ci_hi], "n_better": res.n_positive,
            "significant": res.significant,
            "cells": [{"task": c[0], "budget": c[1], "seed": c[2], "diff": d}
                      for c, d in zip(cells, diffs)],
        }

    # Per-variant absolute means, for context in the writeup.
    print("\nabsolute mean reward by variant x task:")
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in runs:
        agg[(r.task, r.variant)].append(r.mean_reward)
    for (task, variant), vals in sorted(agg.items()):
        print(f"  {task:9s} {variant:16s} mean={mean(vals):.4f}  (n={len(vals)} runs)")
        out.setdefault("absolute", {})[f"{task}_{variant}"] = {
            "mean": mean(vals), "n_runs": len(vals)}

    dest = ROOT / "results" / "a01_variant_comparison.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
