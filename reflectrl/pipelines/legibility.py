"""
Step 1.6 -- Feedback-condition switches and legibility-degradation conditions.

Two distinct axes, per the proposal's §3/§6 -- don't conflate them:

- FEEDBACK CONDITIONS control what signal an *optimizer* (GEPA) sees during
  optimization: {scalar-only, +evaluator rationale} (a 3rd, +traces, is
  stretch-only per the plan and not implemented here yet).
- LEGIBILITY CONDITIONS control what a trajectory looks like when the
  *legibility judge* (legibility_judge.py) tries to localize a failure from
  it. Five conditions: the original 4 (baseline/scalar-only-equivalent,
  strip tool logs, summarize traces, add distractor module) plus P2b's
  latent-reasoning proxy (new). P2 and P2b degrade legibility through
  mechanistically distinct routes -- P2 obscures/removes an existing trace,
  P2b compresses one module's reasoning so tightly that faithful
  verbalization becomes impossible even though nothing was hidden. Keep
  their effects analyzed separately downstream (pre-registration §1, P2b).
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Literal

import dspy

FeedbackCondition = Literal["scalar_only", "plus_rationale"]
LegibilityCondition = Literal[
    "baseline", "strip_tool_logs", "summarize_traces", "distractor_module", "latent_reasoning_proxy"
]

ALL_LEGIBILITY_CONDITIONS: list[LegibilityCondition] = [
    "baseline",
    "strip_tool_logs",
    "summarize_traces",
    "distractor_module",
    "latent_reasoning_proxy",
]


# ---------------------------------------------------------------------------
# Feedback conditions (what GEPA's reflection LM sees)
# ---------------------------------------------------------------------------

def make_gepa_feedback_fn(base_score_fn: Callable, condition: FeedbackCondition, feedback_text_fn: Callable | None = None):
    """Wraps a plain (gold, pred) -> float scorer into a GEPAFeedbackMetric.

    condition="scalar_only": returns the bare float -- GEPA's fallback text
    feedback ("This trajectory got a score of {score}.") is all the
    reflection LM sees, no qualitative signal.

    condition="plus_rationale": returns dspy.Prediction(score=..., feedback=...)
    using feedback_text_fn(gold, pred) to build a qualitative explanation --
    matches gepa_repro.py's existing gepa_feedback_metric, factored out here
    so both HotpotQA and AIME-style optimizer runs can toggle the same axis.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        score = base_score_fn(gold, pred)
        if condition == "scalar_only":
            return score
        if condition == "plus_rationale":
            if feedback_text_fn is None:
                raise ValueError("plus_rationale condition requires feedback_text_fn")
            feedback = feedback_text_fn(gold, pred)
            return dspy.Prediction(score=score, feedback=feedback)
        raise ValueError(f"unknown feedback condition {condition!r}")

    return metric


# ---------------------------------------------------------------------------
# Legibility degradation conditions (what the judge sees)
# ---------------------------------------------------------------------------
# Each transform takes a trajectory (the list-of-module-dicts shape built by
# HotpotQAPipeline.forward / AIMEStylePipeline.forward -- see their
# docstrings) and returns a new, degraded trajectory. Never mutates the input.

def _clone(trajectory: list[dict]) -> list[dict]:
    return copy.deepcopy(trajectory)


def apply_baseline(trajectory: list[dict]) -> list[dict]:
    """No degradation -- the judge sees the full trajectory (reasoning,
    inputs, outputs) for every module. This is the legibility ceiling for a
    given (task, system) pair, corresponding to the plan's scalar-only
    feedback-condition baseline for the legibility axis, not the feedback
    axis above -- same name, different axis, don't confuse the two."""
    return _clone(trajectory)


def apply_strip_tool_logs(trajectory: list[dict]) -> list[dict]:
    """Removes bulky tool/retrieval outputs from module inputs (e.g. the
    retrieved passages a generate_answer step read) while keeping each
    module's own reasoning and final output. Simulates losing the
    intermediate evidence a trace would otherwise carry."""
    out = _clone(trajectory)
    for step in out:
        inputs = step.get("inputs", {})
        for key in list(inputs.keys()):
            if key in ("context", "passages"):
                inputs[key] = "[tool output stripped]"
    return out


def apply_summarize_traces(trajectory: list[dict], max_words: int = 12) -> list[dict]:
    """Replaces every module's reasoning with a short truncation --
    simulates a lossy/compressed trace rather than the full chain of
    thought. Uniform across all modules (contrast with
    apply_latent_reasoning_proxy, which targets one module much more
    severely -- see the P2 vs P2b distinction in the module docstring)."""
    out = _clone(trajectory)
    for step in out:
        reasoning = step.get("reasoning", "") or ""
        words = reasoning.split()
        step["reasoning"] = " ".join(words[:max_words]) + (" [...summarized]" if len(words) > max_words else "")
    return out


_DISTRACTOR_MODULES = [
    {
        "module": "irrelevant_style_check",
        "inputs": {"note": "unrelated to task correctness"},
        "reasoning": "Checked response tone against a style guide; tone was acceptable, no changes needed.",
        "outputs": {"style_ok": True},
    },
    {
        "module": "irrelevant_length_check",
        "inputs": {"note": "unrelated to task correctness"},
        "reasoning": "Verified response length was within the configured display limit.",
        "outputs": {"length_ok": True},
    },
]


def apply_distractor_module(trajectory: list[dict], seed: int = 0) -> list[dict]:
    """Inserts one plausible-but-irrelevant module step at a random position
    -- tests whether the judge's localization accuracy degrades under noise
    it must first filter out, rather than under missing information."""
    out = _clone(trajectory)
    rng = random.Random(seed)
    distractor = copy.deepcopy(rng.choice(_DISTRACTOR_MODULES))
    insert_at = rng.randint(0, len(out))
    out.insert(insert_at, distractor)
    return out


def apply_latent_reasoning_proxy(trajectory: list[dict], target_module: str | None = None, max_words: int = 3) -> list[dict]:
    """P2b -- constrains ONE module's reasoning to a token budget tight
    enough to force compression past faithful verbalization (a cheap,
    no-retraining proxy for 'the policy's reasoning was never in language to
    begin with', per Coconut/latent-CoT). Distinct from
    apply_summarize_traces: that shortens every module uniformly (an
    observer's-access problem -- P2); this targets one module severely (a
    does-a-faithful-trace-exist-at-all problem -- P2b). If target_module is
    None, the first module in the trajectory is targeted (deterministic, not
    random, so repeated calls on the same trajectory shape are comparable)."""
    out = _clone(trajectory)
    if not out:
        return out
    idx = 0
    if target_module is not None:
        idx = next((i for i, s in enumerate(out) if s["module"] == target_module), 0)
    reasoning = out[idx].get("reasoning", "") or ""
    words = reasoning.split()
    out[idx]["reasoning"] = " ".join(words[:max_words]) if words else ""
    return out


_TRANSFORMS: dict[LegibilityCondition, Callable[..., list[dict]]] = {
    "baseline": apply_baseline,
    "strip_tool_logs": apply_strip_tool_logs,
    "summarize_traces": apply_summarize_traces,
    "distractor_module": apply_distractor_module,
    "latent_reasoning_proxy": apply_latent_reasoning_proxy,
}


def apply_legibility_condition(trajectory: list[dict], condition: LegibilityCondition, **kwargs) -> list[dict]:
    """Single entrypoint used by the sweep runner (Week 2, Step 2.1) and the
    judge bake-off harness alike, so both stay in sync on what each
    condition name means."""
    if condition not in _TRANSFORMS:
        raise ValueError(f"unknown legibility condition {condition!r}, expected one of {ALL_LEGIBILITY_CONDITIONS}")
    return _TRANSFORMS[condition](trajectory, **kwargs)


def format_trajectory_for_judge(trajectory: list[dict]) -> str:
    """Renders a (possibly degraded) trajectory into the text block the
    legibility judge reads. Kept as one shared function so every condition
    is rendered identically -- differences the judge sees come only from
    what apply_legibility_condition() removed/changed, not from formatting
    drift between conditions."""
    lines = []
    for i, step in enumerate(trajectory):
        lines.append(f"--- Step {i + 1}: module = {step['module']} ---")
        for k, v in step.get("inputs", {}).items():
            lines.append(f"input.{k}: {v}")
        lines.append(f"reasoning: {step.get('reasoning', '')}")
        for k, v in step.get("outputs", {}).items():
            lines.append(f"output.{k}: {v}")
    return "\n".join(lines)
