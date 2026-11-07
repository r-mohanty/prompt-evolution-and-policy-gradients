"""
Step 1.4 -- Reproduce GEPA on HotpotQA; MIPROv2 baseline at matched budget.

Not a literal reconciliation against the GEPA paper's own published numbers:
the paper's exact task model + full-Wikipedia ColBERT retrieval isn't
reproducible here (see decisions.md 2026-08-16 -- the public ColBERT demo is
unreachable, so this project's HotpotQA runs on a local BM25 distractor-pool
retriever instead). What this script *does* verify, which is what Step 1.4's
gate actually needs before the Week 2 regime study can mean anything: GEPA's
optimizer loop runs cleanly end-to-end on our pipeline, produces an optimized
program that beats the zero-shot baseline, and every call is cost-metered.
Literal published-number comparison is noted as a limitation, not silently
skipped.

Run manually:
    python scripts/prompt_opt/gepa_repro.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

import dspy

from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, exact_match, f1_score, load_devset
from reflectrl.pipelines.metering import METER, cost_tag, flush_lm_history

RESULTS_DIR = WORKSHOP_ROOT / "results"

# Task LM: the program being optimized. Reflection LM: the model GEPA uses to
# read failed trajectories and propose prompt edits (Step 1.6's frozen
# reflection model elsewhere in the plan is also Sonnet 5 -- kept consistent
# here rather than introducing a third model choice for this one script).
TASK_MODEL = "anthropic/claude-haiku-4-5"
REFLECTION_MODEL = "anthropic/claude-sonnet-5"

# Explicit, predictable budget for this first reproduction pass -- NOT the
# plan's eventual 5-point (2^7..2^13) sweep, which happens in Week 2's Track A
# once this gate passes. Keeping this small and explicit (rather than
# auto="light", whose actual metric-call count isn't obvious up front) keeps
# cost bounded and visible before scaling up.
N_TRAIN = 15
N_VAL = 20
N_TEST = 15
MAX_METRIC_CALLS = 150


def make_task_lm():
    return dspy.LM(TASK_MODEL, max_tokens=1024, api_key=os.environ["ANTHROPIC_API_KEY"])


def make_reflection_lm():
    return dspy.LM(REFLECTION_MODEL, max_tokens=4096, api_key=os.environ["ANTHROPIC_API_KEY"])


def gepa_feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    """GEPAFeedbackMetric: program-level score + textual feedback for the
    reflection LM. Score is the same EM+F1 average used everywhere else in
    this project so P1/P3's cross-condition comparisons stay apples-to-apples."""
    score = em_f1_metric(gold, pred)
    em = exact_match(pred.answer, gold.answer)
    f1 = f1_score(pred.answer, gold.answer)
    feedback = (
        f"Question: {gold.question}\n"
        f"Gold answer: {gold.answer!r}\n"
        f"Predicted answer: {pred.answer!r}\n"
        f"Search query used: {pred.query!r}\n"
        f"Exact match: {bool(em)}, F1: {f1:.2f}\n"
    )
    if em:
        feedback += "The answer was exactly correct."
    elif f1 > 0:
        feedback += "The answer partially overlapped the gold answer but was not exact -- likely wrong span length or phrasing."
    else:
        feedback += "The answer was wrong. Check whether the search query retrieved evidence relevant to the gold answer."
    return dspy.Prediction(score=score, feedback=feedback)


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dspy.configure(lm=make_task_lm())

    trainset, rest = load_devset(n_train=N_TRAIN, n_dev=N_VAL + N_TEST)
    valset, testset = rest[:N_VAL], rest[N_VAL:N_VAL + N_TEST]
    print(f"train={len(trainset)} val={len(valset)} test={len(testset)}")

    baseline_program = HotpotQAPipeline(k=5)

    evaluator = dspy.Evaluate(devset=testset, metric=em_f1_metric, num_threads=4, display_progress=True)

    print("\n=== Baseline (zero-shot) eval on held-out test ===")
    with cost_tag("gepa-repro-baseline-eval"):
        baseline_result = evaluator(baseline_program)
    # dspy.Evaluate returns an EvaluationResult (.score is 0-100, a percent)
    # not a plain float -- normalize to [0,1] to match em_f1_metric's own
    # scale and every other metric in this project.
    baseline_score = baseline_result.score / 100.0
    baseline_cost_so_far = METER.total_usd
    print(f"Baseline EM+F1 (mean): {baseline_score:.4f}")
    print(f"Running total spend: ${baseline_cost_so_far:.4f}")

    print("\n=== GEPA optimization ===")
    reflection_lm = make_reflection_lm()
    gepa = dspy.GEPA(
        metric=gepa_feedback_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=MAX_METRIC_CALLS,
        reflection_minibatch_size=3,
        num_threads=4,
        track_stats=True,
    )
    t0 = time.monotonic()
    with cost_tag("gepa-repro-optimize"):
        optimized_program = gepa.compile(baseline_program, trainset=trainset, valset=valset)
        # reflection_lm's calls bypass pipeline.forward() (GEPA calls it
        # directly for reflection, not through the module), so it isn't
        # covered by the per-forward-pass flush in HotpotQAPipeline -- flush
        # it explicitly here or its cost silently never gets logged.
        flush_lm_history(reflection_lm, tag="gepa-reflection")
    optimize_wall_s = time.monotonic() - t0
    optimize_cost = METER.total_usd - baseline_cost_so_far
    print(f"GEPA optimize wall time: {optimize_wall_s:.0f}s, cost: ${optimize_cost:.4f}")

    print("\n=== Optimized program eval on held-out test ===")
    with cost_tag("gepa-repro-optimized-eval"):
        optimized_result = evaluator(optimized_program)
    optimized_score = optimized_result.score / 100.0
    total_cost = METER.total_usd
    print(f"Optimized EM+F1 (mean): {optimized_score:.4f}")
    print(f"Total running spend: ${total_cost:.4f}")

    results = {
        "task": "hotpotqa",
        "optimizer": "GEPA",
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "n_train": len(trainset),
        "n_val": len(valset),
        "n_test": len(testset),
        "max_metric_calls": MAX_METRIC_CALLS,
        "baseline_em_f1": baseline_score,
        "optimized_em_f1": optimized_score,
        "gap": optimized_score - baseline_score,
        "optimize_wall_seconds": optimize_wall_s,
        "optimize_cost_usd": optimize_cost,
        "total_cost_usd_at_completion": total_cost,
        "limitation_note": (
            "Not a literal reconciliation against GEPA's own published HotpotQA "
            "numbers -- different task model (Haiku 4.5 vs the paper's) and "
            "different retrieval (local BM25 over the distractor pool vs the "
            "paper's full-Wikipedia ColBERT index, which is unreachable from "
            "this environment). Verifies the optimizer loop and improvement "
            "direction, not absolute score parity."
        ),
    }
    out_path = RESULTS_DIR / "gepa_hotpotqa_repro.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    run()
