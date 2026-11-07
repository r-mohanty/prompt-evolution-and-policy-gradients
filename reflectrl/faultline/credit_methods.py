"""
Step 2.6 -- Credit-assignment methods scored against FaultLine instances.

Four methods, per the plan's §4/§6: LM reflection (flat, default), LM
reflection (recursive/programmatic -- RLM-style, separate module, see
credit_recursive_reflector.py), Shapley-style ablate-and-rescore, uniform
baseline. Each implements the same interface:

    score(instance: FaultLineInstance) -> CreditPrediction

so scripts/faultline/score_credit_methods.py can run all of them over the same
instance set and compute the same three metrics (localization accuracy,
calibration, cost) for each.
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from reflectrl.pipelines.legibility_judge import LegibilityJudge, configure_judge_lm, judge_failed_trajectory  # noqa: E402
from reflectrl.faultline.verifier import FaultLineInstance  # noqa: E402


@dataclass
class CreditPrediction:
    predicted_module: str
    confidence: float  # [0,1]
    cost_usd: float
    latency_s: float
    method: str


# ---------------------------------------------------------------------------
# Method 1: LM reflection, flat (the default) -- this IS the Step 1.6
# legibility judge. No new mechanism: FaultLine's credit-scoring use of the
# judge and the legibility meter's use of the judge are the same model doing
# the same task; what differs is what ground truth their output gets
# compared against (FaultLine's counterfactual truth here, vs L's still-open
# dependency on that same ground truth -- see legibility_judge.py's
# docstring).
# ---------------------------------------------------------------------------

def score_flat_reflection(instance: FaultLineInstance, judge: LegibilityJudge, judge_lm, all_modules: list[str]) -> CreditPrediction:
    t0 = time.monotonic()
    result = judge_failed_trajectory(
        judge=judge,
        judge_lm=judge_lm,
        modules=all_modules,
        trajectory=instance.faulty_trajectory,
        question=instance.question,
        gold_answer=instance.gold_answer,
        predicted_answer=instance.faulty_answer,
        condition="baseline",  # scoring against the trajectory as-is; legibility DEGRADATION conditions are a separate sweep axis (Step 1.6), not mixed in here
    )
    latency = time.monotonic() - t0
    return CreditPrediction(
        predicted_module=result.failing_module,
        confidence=float(result.confidence),
        cost_usd=0.0,  # filled in by the caller from the cost meter delta -- see score_credit_methods.py
        latency_s=latency,
        method="flat_reflection",
    )


# ---------------------------------------------------------------------------
# Method 2: uniform baseline -- the floor. Picks uniformly at random among
# the pipeline's modules, deterministic given instance.example_id as the
# seed so results are reproducible without needing a true RNG stream.
# ---------------------------------------------------------------------------

def score_uniform(instance: FaultLineInstance, all_modules: list[str]) -> CreditPrediction:
    rng = random.Random(hash((instance.task, instance.example_id, instance.target_module, instance.operator)) & 0xFFFFFFFF)
    predicted = rng.choice(all_modules)
    return CreditPrediction(
        predicted_module=predicted,
        confidence=1.0 / len(all_modules),
        cost_usd=0.0,
        latency_s=0.0,
        method="uniform",
    )


# ---------------------------------------------------------------------------
# Method 3: Shapley-style ablate-and-rescore. For each candidate module,
# "fix" it back to correct instructions while leaving the rest of the
# pipeline as the faulty configuration, rerun, and measure the resulting
# score. The module whose fix produces the LARGEST score improvement is
# predicted as the responsible one -- a real leave-one-out mechanism (it
# reruns the pipeline once per module, unlike the single-shot reflection
# methods), which is exactly the accuracy/cost tradeoff FaultLine's
# 3-metric scoring (accuracy, calibration, cost) is built to compare.
# ---------------------------------------------------------------------------

def score_shapley(instance: FaultLineInstance, example, baseline_pipeline, run_fn, score_fn, all_modules: list[str], target_module: str, operator: str, swap_terms_map: dict) -> CreditPrediction:
    """Needs the original dspy.Example (for HotpotQA, its `context` field
    isn't stored on FaultLineInstance -- only question/gold_answer are) plus
    the (target_module, operator) that built the faulty pipeline in the
    first place, so this can reconstruct the exact faulty pipeline and then
    try fixing each module in turn, one at a time. score_credit_methods.py
    passes all of this through from the same generation-time data
    FaultLineInstance was built from."""
    from reflectrl.faultline.injectors import inject_fault
    import copy as _copy

    t0 = time.monotonic()
    faulty_pipeline = inject_fault(baseline_pipeline, target_module, operator, swap_terms_map.get(target_module))
    faulty_score = score_fn(example, run_fn(faulty_pipeline, example))

    best_module = all_modules[0]
    best_improvement = -1.0
    for candidate_module in all_modules:
        # "Fix" candidate_module by swapping in baseline_pipeline's own
        # (correct) predictor for just that one attribute, keeping every
        # other module exactly as faulty_pipeline has it -- a real
        # leave-one-out rerun, not a simulation.
        patched = _copy.copy(faulty_pipeline)
        patched.__dict__[candidate_module] = baseline_pipeline.__dict__[candidate_module]
        patched_score = score_fn(example, run_fn(patched, example))
        improvement = patched_score - faulty_score
        if improvement > best_improvement:
            best_improvement = improvement
            best_module = candidate_module

    latency = time.monotonic() - t0
    # Confidence proxy: how much better the best fix was vs a flat prior --
    # normalized improvement, clipped to [0,1]. Not a calibrated probability;
    # documented as a proxy so the calibration metric (Step 2.6) is scored
    # honestly against what this actually is.
    confidence = max(0.0, min(1.0, best_improvement))
    return CreditPrediction(
        predicted_module=best_module,
        confidence=confidence,
        cost_usd=0.0,
        latency_s=latency,
        method="shapley_ablate_rescore",
    )
