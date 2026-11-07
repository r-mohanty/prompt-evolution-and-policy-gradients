#!/usr/bin/env python
"""A06 -- on-policy GEPA arm (2026-09-03): does the regime boundary survive when the
prompts are evolved on the policy itself?

The paper's gepa_then_grpo arm evolved prompts with Claude Haiku 4.5 as GEPA's task
model and transplanted them into Qwen3-8B (Section 3.2; named as a confound in
Section 6). gepa_onpolicy_then_grpo evolves them with Qwen3-8B itself, served by vLLM,
and is otherwise identical. Only the budget-2048 cells were run (3 seeds x 2 tasks).

Per task this reports, at budget 2048, the run means of the three arms and two paired
differences over seeds: on-policy minus default-prompt Dr. GRPO (does prompt evolution
still help/hurt?) and on-policy minus transplanted (did the transplant matter?).
n=3 per task, so the t-test is weak by construction; the per-seed differences are
printed and are the honest evidence.

    python analysis/a06_onpolicy_gepa.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from lib_runs import ROOT, load_runs, paired_test

BUDGET = 2048
BASE = {"lr": 1e-6, "gs": 8, "kl": 0.04}
ARMS = ["dr_grpo", "gepa_then_grpo", "gepa_onpolicy_then_grpo", "docstring_then_grpo"]


def main() -> None:
    runs = [r for r in load_runs()
            if r.rewards and r.budget == BUDGET and r.variant in ARMS
            and r.lr == BASE["lr"] and r.gs == BASE["gs"] and r.kl == BASE["kl"]]
    by: dict[tuple[str, int], dict[str, object]] = defaultdict(dict)
    for r in runs:
        by[(r.task, r.seed)][r.variant] = r

    print("=" * 92)
    print(f"A06 -- ON-POLICY GEPA vs TRANSPLANTED GEPA vs DEFAULT PROMPTS, budget={BUDGET}")
    print("=" * 92)
    out: dict = {"budget": BUDGET, "per_task": {}}
    for task in ("hotpotqa", "aime"):
        cells = sorted(k for k in by if k[0] == task)
        print(f"\n{task}: {len(cells)} seeds present")
        print(f"  {'seed':>4s}  {'dr_grpo':>9s}  {'transplant':>10s}  {'on-policy':>9s}  {'docstring':>9s}  "
              f"{'onp-dr':>8s}  {'doc-dr':>8s}  {'onp-tra':>8s}  {'zero% onp':>9s}")
        d_onp_dr, d_onp_tra, d_doc_dr, d_doc_tra, rows = [], [], [], [], []
        for (_, seed) in cells:
            d = by[(task, seed)]
            g = d.get("dr_grpo"); t = d.get("gepa_then_grpo"); o = d.get("gepa_onpolicy_then_grpo"); c = d.get("docstring_then_grpo")
            f = lambda r: f"{r.mean_reward:9.4f}" if r else f"{'--':>9s}"
            row = {"seed": seed,
                   "dr_grpo": g.mean_reward if g else None,
                   "transplanted": t.mean_reward if t else None,
                   "onpolicy": o.mean_reward if o else None,
                   "docstring": c.mean_reward if c else None,
                   "onpolicy_zero_frac": o.zero_reward_frac if o else None,
                   "docstring_zero_frac": c.zero_reward_frac if c else None,
                   "dr_grpo_zero_frac": g.zero_reward_frac if g else None}
            rows.append(row)
            dd = f"{o.mean_reward - g.mean_reward:+8.4f}" if (o and g) else f"{'--':>8s}"
            dc = f"{c.mean_reward - g.mean_reward:+8.4f}" if (c and g) else f"{'--':>8s}"
            dt = f"{o.mean_reward - t.mean_reward:+8.4f}" if (o and t) else f"{'--':>8s}"
            z = f"{100 * o.zero_reward_frac:8.1f}%" if o else f"{'--':>9s}"
            print(f"  {seed:>4d}  {f(g)}  {f(t):>10s}  {f(o)}  {f(c)}  {dd}  {dc}  {dt}  {z}")
            if o and g:
                d_onp_dr.append(o.mean_reward - g.mean_reward)
            if o and t:
                d_onp_tra.append(o.mean_reward - t.mean_reward)
            if c and g:
                d_doc_dr.append(c.mean_reward - g.mean_reward)
            if c and t:
                d_doc_tra.append(c.mean_reward - t.mean_reward)
        res = {"rows": rows}
        for name, diffs in (("onpolicy_minus_dr_grpo", d_onp_dr), ("onpolicy_minus_transplanted", d_onp_tra),
                            ("docstring_minus_dr_grpo", d_doc_dr), ("docstring_minus_transplanted", d_doc_tra)):
            if len(diffs) >= 2:
                pr = paired_test(diffs)
                print("  " + pr.line(f"{task} {name}"))
                res[name] = {"n": pr.n, "mean_diff": pr.mean_diff, "se": pr.se, "t": pr.t,
                             "ci": [pr.ci_lo, pr.ci_hi], "n_positive": pr.n_positive,
                             "significant": pr.significant, "diffs": diffs}
            elif diffs:
                print(f"  {task} {name}: single cell, diff={diffs[0]:+.4f}")
                res[name] = {"n": 1, "mean_diff": diffs[0], "diffs": diffs}
            else:
                print(f"  {task} {name}: no paired cells")
        out["per_task"][task] = res

    dest = ROOT / "results" / "a06_onpolicy_gepa.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
