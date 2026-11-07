"""
Step 1.6 -- The legibility judge and legibility meter.

The judge: a frozen LM that reads a (possibly degraded, see legibility.py)
failed trajectory and names the module most responsible for the failure.
L(task, system, condition) = the judge's top-1 localization accuracy against
ground truth over K failed trajectories.

IMPORTANT DEPENDENCY NOTE (honest, not hidden): true ground truth for "which
module actually caused this failure" only exists once FaultLine's
counterfactual verifier is built (Week 2, Track B) -- patch module i's
output, replay downstream, confirm success is restored. Until then, this
judge can be *run* (its predictions + reasoning + confidence are useful data
now, e.g. for the model bake-off in judge_bakeoff.py) but its accuracy
against FaultLine ground truth cannot be *scored* yet. Don't compute a fake L
number against anything else in the meantime -- report judge predictions as
predictions, not as a legibility score, until the ground truth exists.

Rubric grounding: Yu Su's "An Illusion of Progress? Assessing the Current
State of Web Agents" (arXiv:2504.01382) documents exactly the failure mode
this judge is vulnerable to -- high false-positive rates and low human
agreement for LLM-based evaluators. That is the explicit reason judge output
is validated against FaultLine's counterfactual ground truth rather than
trusted as self-report, and why the Week-1 bake-off (judge_bakeoff.py)
against human-labeled trajectories exists at all. See
paper/legibility_judge_rubric.md for the full rubric text.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import dspy

from reflectrl.pipelines.legibility import LegibilityCondition, apply_legibility_condition, format_trajectory_for_judge  # noqa: E402
from reflectrl.pipelines.metering import cost_tag, flush_lm_history  # noqa: E402


class LocalizeFailure(dspy.Signature):
    """You are auditing a failed run of a multi-module AI pipeline. You are
    given the pipeline's module execution trace (in order), the question/
    problem it was given, the correct (gold) answer, and the pipeline's
    (wrong) final answer. Identify which single module in the trace is most
    responsible for the failure -- i.e., which module's output, if it had
    been correct instead, would most likely have led to the correct final
    answer. Do not guess based on which module 'looks suspicious' in
    isolation; reason about the causal chain from that module's output to
    the final wrong answer."""

    modules: str = dspy.InputField(desc="comma-separated list of module names in pipeline order")
    trajectory: str = dspy.InputField(desc="the failed trajectory, rendered as text -- may be a degraded/partial view of the true trajectory")
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    failing_module: str = dspy.OutputField(desc="the single module name (must be one of `modules`) most responsible for the failure")
    confidence: float = dspy.OutputField(desc="confidence in [0,1] that this module is indeed the cause")


class LegibilityJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.localize = dspy.ChainOfThought(LocalizeFailure)

    def forward(self, modules: list[str], trajectory: list[dict], question: str, gold_answer: str, predicted_answer: str):
        # No internal cost_tag/hardcoded flush tag here -- BUG FOUND
        # 2026-08-16 (Step 2.6 scoring dry run): an inner `with
        # cost_tag("legibility-judge")` + explicit flush tag silently
        # overrode whatever tag the CALLER set (nested cost_tag contexts,
        # innermost wins), so every judge call got attributed to the same
        # fixed tag regardless of caller -- e.g. score_credit_methods.py's
        # per-instance unique tags all resolved to $0.00 because the real
        # spend was logged under "legibility-judge" instead. Falling back to
        # the ambient tag (whatever `with cost_tag(...)` the caller is
        # already inside) fixes this and also fixes a previously-noted
        # cosmetic issue (decisions.md, Step 1.4/1.6 entries) where a
        # script's own outer tag never showed up in cost breakdowns.
        pred = self.localize(
            modules=", ".join(modules),
            trajectory=format_trajectory_for_judge(trajectory),
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )
        flush_lm_history(dspy.settings.lm)
        return dspy.Prediction(
            failing_module=pred.failing_module,
            confidence=pred.confidence,
            reasoning=pred.reasoning,
        )


def configure_judge_lm(model: str, max_tokens: int = 1024) -> dspy.LM:
    """Configures a standalone LM for judge calls, separate from the task
    LM's dspy.settings.lm -- the plan requires judge != reflection model,
    and callers here explicitly pass the judge LM per call rather than
    relying on global dspy.configure(), so a script can run the bake-off
    (Haiku 4.5 vs Sonnet 5) in the same process without clobbering the task
    pipeline's own configured LM."""
    return dspy.LM(model, max_tokens=max_tokens, api_key=os.environ["ANTHROPIC_API_KEY"])


def judge_failed_trajectory(
    judge: LegibilityJudge,
    judge_lm: dspy.LM,
    modules: list[str],
    trajectory: list[dict],
    question: str,
    gold_answer: str,
    predicted_answer: str,
    condition: LegibilityCondition = "baseline",
    **condition_kwargs,
) -> dspy.Prediction:
    """Applies the requested legibility condition, then runs the judge on
    the degraded trajectory, using judge_lm for this call only (does not
    touch dspy.settings.lm globally)."""
    degraded = apply_legibility_condition(trajectory, condition, **condition_kwargs)
    with dspy.context(lm=judge_lm):
        return judge(
            modules=modules,
            trajectory=degraded,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )
