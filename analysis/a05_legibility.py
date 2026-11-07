#!/usr/bin/env python
"""does fault localisation degrade as the trajectory becomes less legible?

Tests the Legibility Hypothesis. Two mechanistically distinct routes are kept
separate:

  P2  -- the observer LOSES ACCESS to an existing trace
         (strip_tool_logs, summarize_traces)
  P2b -- no faithful trace EXISTS to begin with, because the reasoning was
         compressed past verbalisation (latent_reasoning_proxy*). This is the
         architectural-legibility axis motivated by Coconut/latent-CoT.
         `latent_reasoning_proxy_gold` is the adversarial version: the module
         whose reasoning is destroyed is the GOLD FAULTY one.

`distractor_module` is a control for a third thing entirely: added NOISE rather
than removed information.

Reported both raw and BALANCED (mean per-class recall), because FaultLine's gold
labels are structurally skewed toward downstream modules -- see a03.

Run:  python analysis/a05_legibility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_runs import ROOT, mcnemar  # noqa: E402

RES = ROOT / "results"
BASE = "baseline"


def main() -> None:
    src = RES / "legibility_sweep.json"
    if not src.is_file():
        raise SystemExit("no legibility_sweep.json -- run scripts/faultline/run_legibility_sweep.py first")
    data = json.loads(src.read_text())
    rows = data["rows"]

    by_cond: dict[str, dict[int, bool]] = {}
    gold_of: dict[int, str] = {}
    task_of: dict[int, str] = {}
    for r in rows:
        if r.get("error"):
            continue
        by_cond.setdefault(r["condition"], {})[r["instance_idx"]] = bool(r["correct"])
        gold_of[r["instance_idx"]] = r["gold"]
        task_of[r["instance_idx"]] = r["task"]

    classes = sorted(set(gold_of.values()))
    idx_by_class: dict[str, list[int]] = {}
    for i, g in gold_of.items():
        idx_by_class.setdefault(g, []).append(i)

    def balanced(m: dict[int, bool]) -> float:
        recs = []
        for _cls, idxs in idx_by_class.items():
            sub = [i for i in idxs if i in m]
            if sub:
                recs.append(sum(m[i] for i in sub) / len(sub))
        return sum(recs) / len(recs) if recs else float("nan")

    print("=" * 96)
    print("A05 -- LEGIBILITY DEGRADATION SWEEP")
    print(f"     judge={data['judge_model']}  n_instances={data['n_instances']}  "
          f"cost=${data.get('total_cost_usd_at_completion', 0):.2f}")
    print("=" * 96)
    n_err = sum(1 for r in rows if r.get("error"))
    if n_err:
        print(f"  ({n_err} judge calls errored and are excluded)")
    print(f"{'condition':<30s} {'raw':>7s} {'balanced':>9s} {'n':>4s}   vs baseline (paired McNemar)")

    out: dict = {"judge_model": data["judge_model"], "conditions": {}}
    base_m = by_cond.get(BASE, {})
    order = [BASE, "strip_tool_logs", "summarize_traces", "distractor_module",
             "latent_reasoning_proxy", "latent_reasoning_proxy_gold"]
    for cond in order:
        m = by_cond.get(cond)
        if not m:
            continue
        raw = sum(m.values()) / len(m)
        bal = balanced(m)
        line = f"{cond:<30s} {raw:7.3f} {bal:9.3f} {len(m):4d}"
        rec = {"raw_acc": raw, "balanced_acc": bal, "n": len(m)}
        if cond != BASE and base_m:
            mc = mcnemar(base_m, m)   # a=baseline, b=condition
            rec["vs_baseline"] = mc
            line += (f"   baseline_wins={mc['a_wins']:2d} cond_wins={mc['b_wins']:2d} "
                     f"chi2={mc['chi2']:5.2f} "
                     f"{'DEGRADES (sig)' if mc['significant'] and mc['a_wins'] > mc['b_wins'] else ('sig' if mc['significant'] else 'ns')}")
        print(line)
        out["conditions"][cond] = rec

    print()
    print("  P2 (access removed) vs P2b (no faithful trace exists):")
    for a, b in [("summarize_traces", "latent_reasoning_proxy_gold"),
                 ("latent_reasoning_proxy", "latent_reasoning_proxy_gold")]:
        if a in by_cond and b in by_cond:
            mc = mcnemar(by_cond[a], by_cond[b])
            print(f"    {a:26s} vs {b:28s} {a[:10]}_wins={mc['a_wins']:2d} "
                  f"{b[:10]}_wins={mc['b_wins']:2d} chi2={mc['chi2']:5.2f} "
                  f"{'SIGNIFICANT' if mc['significant'] else 'ns'}")
            out.setdefault("p2_vs_p2b", {})[f"{a}_vs_{b}"] = mc


    # ---- A05b: does the judge use the trajectory AT ALL? -----------------
    abl_path = RES / "trajectory_ablation.json"
    if abl_path.is_file():
        abl = json.loads(abl_path.read_text())
        notr = {r["instance_idx"]: bool(r["correct"]) for r in abl["rows"] if not r.get("error")}
        print()
        print("=" * 96)
        print("A05b -- TRAJECTORY ABLATION: the trace removed ENTIRELY")
        print("=" * 96)
        print("  Motivation: no condition degraded accuracy, and discordant counts were tiny")
        print("  (1-5 of ~69 flipping). The explanation is not that the degradations were weak")
        print("  but that the judge barely uses the trace -- so remove it completely and see.")
        print()
        base_raw = sum(base_m.values()) / len(base_m)
        notr_raw = sum(notr.values()) / len(notr)
        print(f"  full trajectory : raw={base_raw:.3f}  balanced={balanced(base_m):.3f}  n={len(base_m)}")
        print(f"  NO trajectory   : raw={notr_raw:.3f}  balanced={balanced(notr):.3f}  n={len(notr)}")
        mc = mcnemar(notr, base_m)
        print(f"  paired          : no_traj_wins={mc['a_wins']} full_traj_wins={mc['b_wins']} "
              f"chi2={mc['chi2']:.2f} {'SIGNIFICANT' if mc['significant'] else 'ns'}")
        print()
        print("  per task (the effect is entirely AIME):")
        for t in sorted(set(task_of.values())):
            b = {i: v for i, v in base_m.items() if task_of.get(i) == t}
            n = {i: v for i, v in notr.items() if task_of.get(i) == t}
            if not b or not n:
                continue
            m2 = mcnemar(n, b)
            print(f"    {t:9s} full={sum(b.values())/len(b):.3f}  none={sum(n.values())/len(n):.3f}   "
                  f"paired {m2['a_wins']}-{m2['b_wins']} chi2={m2['chi2']:.2f} "
                  f"{'SIGNIFICANT' if m2['significant'] else 'ns'}")
        out["trajectory_ablation"] = {
            "full_raw": base_raw, "none_raw": notr_raw,
            "full_balanced": balanced(base_m), "none_balanced": balanced(notr),
            "paired_overall": mc,
        }

    dest = RES / "a05_legibility.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
