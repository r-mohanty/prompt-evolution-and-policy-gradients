"""
Step 1.4 -- MIPROv2 baseline at matched budget.

Same task, same held-out test split, same models, and a comparably-sized
budget as gepa_repro.py -- MIPROv2 is the reference baseline (not the paper's
main contrast, which is GEPA vs. Dr. GRPO), so this exists to sanity-check
GEPA's numbers against a second established DSPy optimizer, not as a
headline result. load_devset() uses a fixed shuffle seed, so calling it here
with the same n_train/n_dev reproduces gepa_repro.py's exact train/val/test
split for a fair side-by-side.

Run manually (after gepa_repro.py, to avoid contending for the same API
rate limits / cost-meter budget at the same time):
    python scripts/prompt_opt/miprov2_repro.py
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

from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset
from reflectrl.pipelines.metering import METER, cost_tag, flush_lm_history

RESULTS_DIR = WORKSHOP_ROOT / "results"

# Same model choices as gepa_repro.py: task LM is what's being optimized,
# prompt_model plays MIPROv2's analog of GEPA's reflection_lm (proposes
# candidate instructions/demos rather than reading failure feedback).
TASK_MODEL = "anthropic/claude-haiku-4-5"
PROMPT_MODEL = "anthropic/claude-sonnet-5"

# Same split sizes as gepa_repro.py -- load_devset's fixed seed(0) shuffle
# makes this the identical train/val/test partition.
N_TRAIN = 15
N_VAL = 20
N_TEST = 15

# auto="light" is MIPROv2's smallest built-in preset -- matched in spirit
# (not exact metric-call count, MIPROv2 doesn't expose that directly) to
# gepa_repro.py's MAX_METRIC_CALLS=150 budget cap for a first comparison.
AUTO_PRESET = "light"


def make_task_lm():
    return dspy.LM(TASK_MODEL, max_tokens=1024, api_key=os.environ["ANTHROPIC_API_KEY"])


def make_prompt_lm():
    return dspy.LM(PROMPT_MODEL, max_tokens=4096, api_key=os.environ["ANTHROPIC_API_KEY"])


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    task_lm = make_task_lm()
    dspy.configure(lm=task_lm)

    trainset, rest = load_devset(n_train=N_TRAIN, n_dev=N_VAL + N_TEST)
    valset, testset = rest[:N_VAL], rest[N_VAL:N_VAL + N_TEST]
    print(f"train={len(trainset)} val={len(valset)} test={len(testset)}")

    baseline_program = HotpotQAPipeline(k=5)
    evaluator = dspy.Evaluate(devset=testset, metric=em_f1_metric, num_threads=4, display_progress=True)

    print("\n=== Baseline (zero-shot) eval on held-out test ===")
    with cost_tag("miprov2-repro-baseline-eval"):
        baseline_result = evaluator(baseline_program)
    # dspy.Evaluate returns an EvaluationResult (.score is 0-100, a percent),
    # not a plain float -- normalize to [0,1] (see gepa_repro.py, where this
    # was caught after a crash on EvaluationResult subtraction).
    baseline_score = baseline_result.score / 100.0
    baseline_cost_so_far = METER.total_usd
    print(f"Baseline EM+F1 (mean): {baseline_score:.4f}")
    print(f"Running total spend: ${baseline_cost_so_far:.4f}")

    print("\n=== MIPROv2 optimization ===")
    prompt_lm = make_prompt_lm()
    mipro = dspy.MIPROv2(
        metric=em_f1_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto=AUTO_PRESET,
        num_threads=4,
    )
    t0 = time.monotonic()
    with cost_tag("miprov2-repro-optimize"):
        optimized_program = mipro.compile(
            baseline_program,
            trainset=trainset,
            valset=valset,
            requires_permission_to_run=False,
        )
        # prompt_lm's calls (instruction/demo proposals) bypass forward(),
        # same reasoning as gepa_repro.py's reflection_lm flush.
        flush_lm_history(prompt_lm, tag="miprov2-prompt-proposal")
    optimize_wall_s = time.monotonic() - t0
    optimize_cost = METER.total_usd - baseline_cost_so_far
    print(f"MIPROv2 optimize wall time: {optimize_wall_s:.0f}s, cost: ${optimize_cost:.4f}")

    print("\n=== Optimized program eval on held-out test ===")
    with cost_tag("miprov2-repro-optimized-eval"):
        optimized_result = evaluator(optimized_program)
    optimized_score = optimized_result.score / 100.0
    total_cost = METER.total_usd
    print(f"Optimized EM+F1 (mean): {optimized_score:.4f}")
    print(f"Total running spend: ${total_cost:.4f}")

    results = {
        "task": "hotpotqa",
        "optimizer": "MIPROv2",
        "task_model": TASK_MODEL,
        "prompt_model": PROMPT_MODEL,
        "auto_preset": AUTO_PRESET,
        "n_train": len(trainset),
        "n_val": len(valset),
        "n_test": len(testset),
        "baseline_em_f1": baseline_score,
        "optimized_em_f1": optimized_score,
        "gap": optimized_score - baseline_score,
        "optimize_wall_seconds": optimize_wall_s,
        "optimize_cost_usd": optimize_cost,
        "total_cost_usd_at_completion": total_cost,
    }
    out_path = RESULTS_DIR / "miprov2_hotpotqa_repro.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results


if __name__ == "__main__":
    run()
