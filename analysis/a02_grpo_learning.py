#!/usr/bin/env python
"""A02 -- how much does GRPO actually learn, and why can't credit assignment help?

Three results the paper leans on:
  (1) GRPO's TOTAL learning, from the longest runs (budget 8192 = 1024 steps),
      comparing the first tenth of training to the last, paired across runs.
      This is the denominator that makes every other effect interpretable.
  (2) The zero-gradient rate: steps where all rollouts in a group score
      identically produce r_i - mean(r) = 0 for every i, so there is no
      advantage for credit assignment to redistribute, however accurate it is.
  (3) The learning-rate positive control, which proves the apparatus CAN
      detect a real effect under the same scoring.

Run:  python analysis/a02_grpo_learning.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_runs import ROOT, block_bootstrap_mean, load_runs, mean, paired_test  # noqa: E402

GRID_BUDGETS = {128, 256, 512, 2048, 8192}

N_BINS = 10


def binned(rewards: list[float], n: int = N_BINS) -> list[float]:
    w = len(rewards) / n
    return [mean(rewards[int(k * w):int((k + 1) * w)]) for k in range(n)]


def main() -> None:
    runs = [r for r in load_runs() if r.rewards and r.budget in GRID_BUDGETS]
    out: dict = {}

    # ---- (1) GRPO's total learning ------------------------------------
    print("=" * 92)
    print("A02.1 -- DOES GRPO LEARN? first-tenth vs last-tenth, longest runs (budget 8192)")
    print("=" * 92)
    long_runs = [r for r in runs if r.variant == "dr_grpo" and r.budget == 8192
                 and r.lr == 1e-6 and len(r.rewards) >= 100]
    deltas = []
    for r in sorted(long_runs, key=lambda r: r.name):
        b = binned(r.rewards)
        d = b[-1] - b[0]
        deltas.append(d)
        print(f"  {r.name[:52]:<52s} steps={len(r.rewards):5d}  "
              f"bin1={b[0]:.4f} -> bin10={b[-1]:.4f}   delta={d:+.4f}")
    if deltas:
        res = paired_test(deltas)
        print("\n  " + res.line("GRPO total learning"))
        out["grpo_learning"] = {"n_runs": res.n, "mean_delta": res.mean_diff, "se": res.se,
                                "t": res.t, "ci": [res.ci_lo, res.ci_hi],
                                "significant": res.significant}

    # ---- (2) zero-gradient rate ---------------------------------------
    print("\n" + "=" * 92)
    print("A02.2 -- ZERO-GRADIENT RATE (steps where mean_reward==0 => no advantage to redistribute)")
    print("=" * 92)
    print("  NOTE: a lower bound. mean_reward==0 implies every rollout scored 0, hence zero")
    print("        advantage; but any group with IDENTICAL non-zero rewards is also zero-gradient")
    print("        and is not detectable from the logged mean alone.\n")
    z: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in runs:
        if r.lr == 1e-6:
            z[(r.task, r.variant)].append(r.zero_reward_frac)
    for (task, variant), fr in sorted(z.items()):
        print(f"  {task:9s} {variant:16s} zero-gradient steps = {100*mean(fr):5.1f}%  (n={len(fr)} runs)")
        out.setdefault("zero_gradient", {})[f"{task}_{variant}"] = {
            "frac": mean(fr), "n_runs": len(fr)}

    # ---- (3) learning-rate positive control ---------------------------
    print("\n" + "=" * 92)
    print("A02.3 -- POSITIVE CONTROL: does a real intervention register under this scoring?")
    print("=" * 92)
    print("  Only ONE (task,budget,seed) cell was swept, so the cluster-robust design used")
    print("  above is unavailable (n=1 cell). Steps within a run are autocorrelated, so a")
    print("  step-paired t-test would badly overstate significance. We therefore report a")
    print("  MOVING-BLOCK BOOTSTRAP over the per-step differences instead.")
    print()
    by_lr: dict[tuple, dict[float, list[float]]] = defaultdict(dict)
    for r in runs:
        if r.variant == "dr_grpo" and r.gs == 8 and r.kl == 0.04:
            by_lr[r.cell][r.lr] = r.rewards
    for hi in (1e-5, 3e-5):
        for cell, d in sorted(by_lr.items()):
            if hi not in d or 1e-6 not in d:
                continue
            n = min(len(d[hi]), len(d[1e-6]))
            diffs = [d[hi][i] - d[1e-6][i] for i in range(n)]
            naive = paired_test(diffs)
            bs = block_bootstrap_mean(diffs)
            print(f"  lr={hi:g} vs 1e-06  cell={cell}  n_steps={n}")
            print(f"     naive step-paired  : mean={naive.mean_diff:+.4f} t={naive.t:+.2f}  "
                  f"<-- INFLATED, autocorrelated steps treated as independent")
            print(f"     block bootstrap    : mean={bs['mean']:+.4f} SE={bs['se']:.4f} "
                  f"95%CI=[{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}]  "
                  f"{'EXCLUDES ZERO' if bs['excludes_zero'] else 'includes zero'}")
            out.setdefault("lr_control", {})[f"{hi:g}_vs_1e-06"] = {
                "cell": list(cell), "n_steps": n,
                "naive_t": naive.t, "naive_mean": naive.mean_diff,
                "bootstrap": bs,
                "caveat": "single cell; cluster-robust test not possible",
            }

    dest = ROOT / "results" / "a02_grpo_learning.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
