#!/usr/bin/env python
"""A05b -- does the legibility judge actually USE the execution trace?

The legibility sweep (a05) found no degradation across five conditions, with
tiny discordant counts (1-5 instances flipping out of ~69). The natural
explanation is that the judge is not reading the trajectory at all: it may be
inferring the faulty module from the question / gold answer / wrong answer
alone. If so, the flat degradation curve and the near-chance baseline are the
same fact, and "legibility" was never being measured.

This tests it directly by REMOVING the trajectory entirely. If accuracy is
unchanged, the trace is decorative.

Run:  python scripts/faultline/run_trajectory_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import dspy  # noqa: E402
from reflectrl.pipelines.legibility_judge import LegibilityJudge, configure_judge_lm  # noqa: E402
from reflectrl.pipelines.metering import METER, cost_tag  # noqa: E402

JUDGE_MODEL = "anthropic/claude-sonnet-5"
JUDGE_MAX_TOKENS = 4096
HOTPOTQA_MODULES = ["generate_query", "generate_answer"]
AIME_MODULES = ["plan", "solve", "verify"]
OUT = ROOT / "results" / "trajectory_ablation.json"


def run_one(i, inst, judge, judge_lm):
    modules = HOTPOTQA_MODULES if inst["task"] == "hotpotqa" else AIME_MODULES
    gold = inst["target_module"]
    t0 = time.monotonic()
    try:
        with cost_tag(f"traj-ablation-{i}"):
            # Bypass judge_failed_trajectory: pass an EMPTY trajectory so the
            # judge sees only question / gold / predicted answer.
            with dspy.context(lm=judge_lm):
                pred = judge(modules=modules, trajectory=[], question=inst["question"],
                             gold_answer=inst["gold_answer"], predicted_answer=inst["faulty_answer"])
        p = str(pred.failing_module).strip()
        return {"instance_idx": i, "task": inst["task"], "gold": gold, "condition": "no_trajectory",
                "predicted": p, "correct": p == gold, "latency_s": time.monotonic() - t0}
    except Exception as e:  # noqa: BLE001
        return {"instance_idx": i, "task": inst["task"], "gold": gold, "condition": "no_trajectory",
                "predicted": None, "correct": False, "error": f"{type(e).__name__}: {e}"}


def main():
    insts = json.loads((ROOT / "results" / "faultline_instances.json").read_text())["instances"]
    dspy.configure(lm=dspy.LM(JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS, api_key=os.environ["ANTHROPIC_API_KEY"]))
    judge = LegibilityJudge()
    judge_lm = configure_judge_lm(JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS)
    rows, lock, done = [], threading.Lock(), 0
    print(f"{len(insts)} instances, trajectory REMOVED entirely")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_one, i, x, judge, judge_lm) for i, x in enumerate(insts)]
        for f in as_completed(futs):
            with lock:
                rows.append(f.result()); done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(insts)}  ${METER.total_usd:.2f}", flush=True)
    OUT.write_text(json.dumps({"judge_model": JUDGE_MODEL, "n": len(insts),
                               "total_cost_usd_at_completion": METER.total_usd, "rows": rows}, indent=2))
    ok = [r for r in rows if not r.get("error")]
    print(f"\nno_trajectory raw accuracy = {sum(r['correct'] for r in ok)/len(ok):.3f}  (n={len(ok)})")
    print(f"wrote {OUT.relative_to(ROOT)}  (${METER.total_usd:.2f})")


if __name__ == "__main__":
    main()
