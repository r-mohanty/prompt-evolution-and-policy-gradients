#!/usr/bin/env python
"""A3 / Step 1.6 + 2.3 -- legibility-degradation sweep.

Runs the SAME legibility judge used for FaultLine's flat_reflection scoring
across every legibility condition, measuring how fault-localisation accuracy
degrades as the trajectory becomes less legible.

Conditions (see reflectrl/pipelines/legibility.py for the P2 vs P2b distinction):
  baseline                  -- trajectory as-is
  strip_tool_logs           -- P2: retrieved context/passages removed
  summarize_traces          -- P2: every module's reasoning truncated uniformly
  distractor_module         -- noise rather than missing information
  latent_reasoning_proxy    -- P2b: ONE module compressed past faithful
                               verbalisation (first module; neutral placement)
  latent_reasoning_proxy_gold -- P2b, adversarial: the compressed module is the
                               GOLD FAULTY one. This is the sharp test of the
                               architectural-legibility hypothesis -- if the
                               culprit never verbalised its reasoning, can the
                               judge still find it?

API only, no GPU.  Run:  python scripts/faultline/run_legibility_sweep.py
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
from reflectrl.pipelines.legibility_judge import LegibilityJudge, configure_judge_lm, judge_failed_trajectory  # noqa: E402
from reflectrl.pipelines.metering import METER, cost_tag  # noqa: E402

JUDGE_MODEL = "anthropic/claude-sonnet-5"   # matches score_credit_methods.py
# 2026-09-01: the first sweep lost 61/132 AIME judgments to
# "AdapterParseError: LM returned an empty or null response" at max_tokens=1024 --
# AIME's 3-module math traces plus ChainOfThought reasoning plus 3 output fields
# do not fit. Raised to 4096; HotpotQA was almost unaffected (2 errors) so this
# only changes whether the AIME arm exists at all.
JUDGE_MAX_TOKENS = 4096
HOTPOTQA_MODULES = ["generate_query", "generate_answer"]
AIME_MODULES = ["plan", "solve", "verify"]
NUM_WORKERS = 6
CHECKPOINT_EVERY = 10

CONDITIONS = [
    ("baseline", {}),
    ("strip_tool_logs", {}),
    ("summarize_traces", {}),
    ("distractor_module", {}),
    ("latent_reasoning_proxy", {}),
    ("latent_reasoning_proxy_gold", {}),   # kwargs filled per-instance
]

OUT = ROOT / "results" / "legibility_sweep.json"
CKPT = ROOT / "results" / "legibility_sweep_checkpoint.json"


def load_instances() -> list[dict]:
    data = json.loads((ROOT / "results" / "faultline_instances.json").read_text())
    return data["instances"]


def run_one(i: int, inst: dict, judge, judge_lm) -> list[dict]:
    modules = HOTPOTQA_MODULES if inst["task"] == "hotpotqa" else AIME_MODULES
    gold = inst["target_module"]
    rows = []
    for cond, kwargs in CONDITIONS:
        real_cond, kw = cond, dict(kwargs)
        if cond == "latent_reasoning_proxy_gold":
            real_cond = "latent_reasoning_proxy"
            kw["target_module"] = gold
        tag = f"legibility-{cond}-{i}"
        t0 = time.monotonic()
        try:
            with cost_tag(tag):
                pred = judge_failed_trajectory(
                    judge=judge, judge_lm=judge_lm, modules=modules,
                    trajectory=inst["faulty_trajectory"], question=inst["question"],
                    gold_answer=inst["gold_answer"], predicted_answer=inst["faulty_answer"],
                    condition=real_cond, **kw,
                )
            predicted = str(pred.failing_module).strip()
            rows.append({"instance_idx": i, "task": inst["task"], "gold": gold,
                         "condition": cond, "predicted": predicted,
                         "correct": predicted == gold,
                         "confidence": float(getattr(pred, "confidence", 0.0) or 0.0),
                         "latency_s": time.monotonic() - t0})
        except Exception as e:  # noqa: BLE001 -- a failure is data, not a crash
            rows.append({"instance_idx": i, "task": inst["task"], "gold": gold,
                         "condition": cond, "predicted": None, "correct": False,
                         "error": f"{type(e).__name__}: {e}",
                         "latency_s": time.monotonic() - t0})
    return rows


def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    instances = load_instances()
    print(f"{len(instances)} instances x {len(CONDITIONS)} conditions "
          f"= {len(instances)*len(CONDITIONS)} judge calls (model={JUDGE_MODEL})")

    dspy.configure(lm=dspy.LM(JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS, api_key=os.environ["ANTHROPIC_API_KEY"]))
    judge = LegibilityJudge()
    judge_lm = configure_judge_lm(JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS)

    all_rows: list[dict] = []
    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(run_one, i, inst, judge, judge_lm): i
                for i, inst in enumerate(instances)}
        for fut in as_completed(futs):
            rows = fut.result()
            with lock:
                all_rows.extend(rows)
                done += 1
                if done % 5 == 0 or done == len(instances):
                    print(f"  {done}/{len(instances)} instances, spend=${METER.total_usd:.2f}", flush=True)
                if done % CHECKPOINT_EVERY == 0:
                    CKPT.write_text(json.dumps(all_rows, indent=2))

    OUT.write_text(json.dumps({
        "judge_model": JUDGE_MODEL,
        "n_instances": len(instances),
        "conditions": [c for c, _ in CONDITIONS],
        "total_cost_usd_at_completion": METER.total_usd,
        "rows": all_rows,
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}  (spend ${METER.total_usd:.2f})")


if __name__ == "__main__":
    main()
