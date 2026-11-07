"""
Step 1.6 -- Judge model bake-off harness: Haiku 4.5 vs Sonnet 5.

Collects real failed HotpotQA trajectories (baseline pipeline, no
optimization), runs both candidate judge models on each (baseline
legibility condition -- no degradation, so this is a clean starting
comparison), and writes a review sheet. The human-labeling half of the
actual bake-off (>=90% agreement against human judgment) still needs a
person -- see paper/legibility_judge_rubric.md. This script produces the
`your_label` column empty and ready to fill in; it does NOT decide the
bake-off winner on its own (inter-judge agreement is reported as a
descriptive signal only, not a substitute for human ground truth).

Run manually:
    python scripts/faultline/judge_bakeoff.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

import dspy

from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset
from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric
from reflectrl.pipelines.aime_pipeline import load_devset as load_aime_devset
from reflectrl.pipelines.legibility_judge import LegibilityJudge, configure_judge_lm, judge_failed_trajectory
from reflectrl.pipelines.metering import METER, cost_tag

RESULTS_DIR = WORKSHOP_ROOT / "results"

TASK_MODEL = "anthropic/claude-haiku-4-5"
CANDIDATE_JUDGES = {
    "haiku-4.5": "anthropic/claude-haiku-4-5",
    "sonnet-5": "anthropic/claude-sonnet-5",
}
N_CANDIDATES_TO_SCAN = 30  # examples to run looking for failures
K_FAILED_TRAJECTORIES = 15  # cap, per the plan's K-failed-trajectories convention
FAILURE_THRESHOLD = 0.5  # score below this counts as "failed" for this harness

# 2026-09-02: parameterised by task. AIME is the higher-value target for the human
# agreement gate: the trace-ablation result is a NULL on HotpotQA (0.745 with and
# without the trace) but strikingly negative on AIME (0.095 with vs 0.364 without,
# below the 0.333 chance rate). Only human labels distinguish "the trace misleads an
# accurate judge" from "the judge is bad at math attribution". AIME's 3 modules and
# more balanced failure distribution also make Cohen's kappa interpretable, whereas
# HotpotQA's 2-module, heavily-skewed structure can let a constant predictor clear a
# 90% agreement gate -- the trap that caused the FaultLine retraction.
# max_tokens: the TASK model's budget when generating trajectories.
# judge_max_tokens: the JUDGE's budget. 2026-09-02 -- AIME crashed at the default 1024:
# the judge's chain-of-thought over a 3-module math trace exhausted the budget before it
# emitted `failing_module`/`confidence`, DSPy fell back ChatAdapter -> JSONAdapter, that
# truncated too, and AdapterParseError propagated and killed the run. This is the same
# failure that cost 46% of the AIME arm in run_legibility_sweep.py, fixed there by
# raising to 4096; the fix is carried over here.
TASKS = {
    "hotpotqa": {"modules": ["generate_query", "generate_answer"],
                 "max_tokens": 1024, "judge_max_tokens": 1024},
    "aime": {"modules": ["plan", "solve", "verify"],
             "max_tokens": 4096, "judge_max_tokens": 4096},
}


def collect_failed_trajectories(task: str = "hotpotqa"):
    cfg = TASKS[task]
    dspy.configure(lm=dspy.LM(TASK_MODEL, max_tokens=cfg["max_tokens"],
                              api_key=os.environ["ANTHROPIC_API_KEY"]))
    if task == "hotpotqa":
        _, devset = load_devset(n_train=0, n_dev=N_CANDIDATES_TO_SCAN)
        pipeline = HotpotQAPipeline(k=5)
        run_one = lambda ex: pipeline(question=ex.question, context=ex.context)
        score_one = em_f1_metric
    else:
        _, devset = load_aime_devset(n_train=0, n_dev=N_CANDIDATES_TO_SCAN)
        pipeline = AIMEStylePipeline()
        run_one = lambda ex: pipeline(question=ex.question)
        score_one = answer_check_metric

    failed = []
    with cost_tag(f"judge-bakeoff-trajectory-collection-{task}"):
        for ex in devset:
            pred = run_one(ex)
            score = score_one(ex, pred)
            if score < FAILURE_THRESHOLD:
                failed.append({"example": ex, "pred": pred, "score": score})
            if len(failed) >= K_FAILED_TRAJECTORIES:
                break
    print(f"[{task}] Collected {len(failed)} failed trajectories "
          f"(scanned up to {N_CANDIDATES_TO_SCAN}, threshold={FAILURE_THRESHOLD})")
    return failed


def run_bakeoff(failed, task: str = "hotpotqa"):
    judge = LegibilityJudge()
    rows = []
    for judge_name, judge_model in CANDIDATE_JUDGES.items():
        judge_lm = configure_judge_lm(judge_model, max_tokens=TASKS[task]["judge_max_tokens"])
        print(f"\n=== Running judge: {judge_name} ({judge_model}) ===")
        for i, item in enumerate(failed):
            ex, pred = item["example"], item["pred"]
            try:
              with cost_tag(f"judge-bakeoff-{task}-{judge_name}"):
                result = judge_failed_trajectory(
                    judge=judge,
                    judge_lm=judge_lm,
                    modules=TASKS[task]["modules"],
                    trajectory=pred.trajectory,
                    question=ex.question,
                    gold_answer=ex.answer,
                    predicted_answer=pred.answer,
                    condition="baseline",
                )
            except Exception as e:  # noqa: BLE001 -- one unparseable judgement must not
                # abort a whole collection run; record it and carry on.
                print(f"  [{judge_name}] trajectory {i}: judge FAILED ({type(e).__name__}) -- recorded as error")
                rows.append({
                    "trajectory_id": i, "judge": judge_name,
                    "question": ex.question, "gold_answer": ex.answer,
                    "predicted_answer": pred.answer, "em_f1": item["score"],
                    "trajectory": pred.trajectory,
                    "judge_failing_module": None, "judge_confidence": None,
                    "judge_reasoning": None, "error": f"{type(e).__name__}: {e}",
                    "your_label": "",
                })
                continue
            rows.append({
                "trajectory_id": i,
                "judge": judge_name,
                "question": ex.question,
                "gold_answer": ex.answer,
                "predicted_answer": pred.answer,
                "em_f1": item["score"],
                # 2026-09-02: the trajectory was passed to the judge but never persisted,
                # which made HUMAN labelling impossible -- a rater cannot attribute module
                # blame without seeing what the query retrieved. Saved now so the >=90%
                # human-agreement gate (Step 1.6) can actually be evaluated.
                "trajectory": pred.trajectory,
                "judge_failing_module": result.failing_module,
                "judge_confidence": result.confidence,
                "judge_reasoning": result.reasoning,
                "your_label": "",  # <-- filled in by a human labeller
            })
    return rows


def compute_inter_judge_agreement(rows):
    by_traj = {}
    for r in rows:
        # Skip error rows. Otherwise two failed judgements both read None and
        # `None == None` would silently score as AGREEMENT.
        if r.get("judge_failing_module") is None:
            continue
        by_traj.setdefault(r["trajectory_id"], {})[r["judge"]] = r["judge_failing_module"]
    judges = list(CANDIDATE_JUDGES.keys())
    if len(judges) != 2:
        return None
    agree = sum(1 for v in by_traj.values() if len(v) == 2 and v[judges[0]] == v[judges[1]])
    total = sum(1 for v in by_traj.values() if len(v) == 2)
    return {"n_compared": total, "n_agree": agree, "agreement_rate": agree / total if total else None}


def run(task: str = "hotpotqa"):
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = collect_failed_trajectories(task)
    if not failed:
        print("No failed trajectories found in this scan -- widen N_CANDIDATES_TO_SCAN or raise FAILURE_THRESHOLD.")
        return

    rows = run_bakeoff(failed, task)
    inter_judge = compute_inter_judge_agreement(rows)

    out = {
        "task": task,
        "modules": TASKS[task]["modules"],
        "task_model": TASK_MODEL,
        "candidate_judges": CANDIDATE_JUDGES,
        "n_failed_trajectories": len(failed),
        "inter_judge_agreement": inter_judge,
        "note": (
            "inter_judge_agreement is a descriptive signal (do the two models agree "
            "with EACH OTHER), not the actual bake-off criterion. The plan's real "
            "criterion is >=90% agreement with HUMAN labels -- fill in the 'your_label' "
            "column in rows below, then recompute agreement against that column before "
            "freezing a judge model."
        ),
        "rows": rows,
        "total_cost_usd_at_completion": METER.total_usd,
    }
    out_path = RESULTS_DIR / (f"judge_bakeoff_{task}.json" if task != "hotpotqa"
                              else "judge_bakeoff.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nInter-judge agreement (descriptive only, not the real bake-off criterion): {inter_judge}")
    print(f"Results written to {out_path} -- fill in 'your_label' for each row, then rerun the agreement calc.")
    print(f"Total running spend: ${METER.total_usd:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Judge bake-off; --task aime is the higher-value gate.")
    ap.add_argument("--task", default="hotpotqa", choices=sorted(TASKS))
    ap.add_argument("--k", type=int, default=None, help="override K_FAILED_TRAJECTORIES")
    ap.add_argument("--scan", type=int, default=None, help="override N_CANDIDATES_TO_SCAN")
    a = ap.parse_args()
    if a.k:
        K_FAILED_TRAJECTORIES = a.k
    if a.scan:
        N_CANDIDATES_TO_SCAN = a.scan
    run(a.task)
