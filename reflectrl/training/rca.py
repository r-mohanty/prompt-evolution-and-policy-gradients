"""
Step 2.7 -- RCA (Reflective Credit Assignment): reflection schema, parser,
counterfactual validation, and the advantage-mask computation.

Per the proposal's §4 formal setup: a reflection LM produces a credit
distribution c over Delta^k (the k pipeline modules) from a failed
trajectory. Counterfactual validation (reusing FaultLine's own
patch-and-rerun mechanism, Step 2.4-2.6) checks whether patching the
top-credited module actually helps; if not, c falls back to uniform. The
validated mask c~ redistributes a scalar GRPO advantage A(tau) across
tokens: a token belonging to module i gets A(tau) * k * c~_i. Uniform credit
(c~_i = 1/k for all i) reduces this to exactly A(tau) everywhere -- the
"uniform-mask ablation = vanilla/Dr. GRPO sanity check" property the plan
calls for, and the reason compute_advantage_mask's unit tests check it
explicitly below.

The mask is applied inside the training loop by
reflectrl/training/multi_module_grpo_train.py (variants rca and rca_no_validate).
The reflection call, the counterfactual validation step (scored against
FaultLine's ground truth by scripts/faultline/rca_faultline_eval.py) and
compute_advantage_mask's arithmetic (tests/test_rca_mask.py) also run without a GPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import numpy as np


REFLECTION_MODEL = "claude-sonnet-5"  # frozen reflection model, consistent with gepa_repro.py's choice

def _build_credit_schema(modules: list[str]) -> dict:
    """Structured outputs (output_config.format=json_schema) reject
    `additionalProperties` set to anything but `false` -- a dynamic
    dict-of-numbers schema (one key per module) 400s with "For 'object'
    type, 'additionalProperties: object' is not supported" (found live,
    2026-08-16, on the very first real call). Since the module list is
    known at call time (2 for HotpotQA, 3 for AIME-style), the fix is to
    build an explicit named-property schema per call instead of a dynamic
    map."""
    return {
        "type": "object",
        "properties": {
            "credit": {
                "type": "object",
                "description": "raw (not necessarily normalized) credit weight per module for how responsible it is for the failure",
                "properties": {m: {"type": "number"} for m in modules},
                "required": modules,
                "additionalProperties": False,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["credit", "reasoning"],
        "additionalProperties": False,
    }


@dataclass
class ReflectionResult:
    raw_credit: dict[str, float]  # as returned by the LM, not yet normalized
    credit: dict[str, float]  # normalized to sum to 1 over `modules`
    reasoning: str
    cost_usd: float
    latency_s: float


def reflect_credit(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    trajectory_text: str,
    modules: list[str],
    client: anthropic.Anthropic | None = None,
) -> ReflectionResult:
    """The reflection call. Uses the Anthropic SDK's structured-outputs
    feature (output_config.format=json_schema) directly rather than DSPy's
    ChainOfThought+JSONAdapter -- Step 2.6's scoring run hit real
    AdapterParseError failures (11-16 per method out of 69) traced to
    max_tokens truncation interacting badly with DSPy's JSON adapter
    fallback path. Structured outputs guarantee schema-valid JSON or a
    clear API-level error, which is more robust for exactly this failure
    mode (see decisions.md, Step 2.6 entry, for the traced root cause)."""
    from reflectrl.pipelines.metering import METER, _tag_ctx

    client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.monotonic()

    prompt = (
        f"A multi-module AI pipeline failed to answer a question correctly. Assign "
        f"CREDIT (relative responsibility) to each module for the failure, as a "
        f"distribution over the {len(modules)} modules -- higher credit means more "
        f"responsible for the wrong final answer. Weights don't need to sum to any "
        f"specific value; they'll be normalized afterward.\n\n"
        f"Modules in pipeline order: {', '.join(modules)}\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Predicted (wrong) answer: {predicted_answer}\n\n"
        f"Trajectory:\n{trajectory_text}"
    )
    # FIXED 2026-08-18 -- caused real, 100%-of-attempts training-job crashes on
    # both Phase 2 pods running rca/rca_no_validate. TWO related failure modes
    # found live, not hypothesized -- fixing only the first one still let real
    # jobs crash on the second, so both are now handled by the same
    # retry-then-fallback loop rather than two separate patches:
    #   (a) `next(b.text for b in response.content if b.type == "text")` raised
    #       StopIteration whenever response.content had no "text"-typed block
    #       (`aime_rca_b2048_s2_001.log`, first occurrence).
    #   (b) even once a text block IS present, `json.loads(text)` can raise
    #       JSONDecodeError ("Unterminated string...") -- the SAME job,
    #       re-run after fix (a) landed, failed a second time this way
    #       (confirmed: (a)'s retry loop correctly found a text block that
    #       attempt, so this is a genuinely different failure, not (a)
    #       recurring). Both point at the same underlying cause: the
    #       reflection response occasionally gets cut off by max_tokens before
    #       the JSON structure closes -- bumped 2048->4096 to make this less
    #       frequent, but the graceful fallback is the real fix regardless of
    #       root cause, since no fixed max_tokens fully eliminates the
    #       possibility.
    # A single malformed API response killing an entire multi-hour GPU job is
    # a much worse failure mode than falling back to uniform credit for that
    # one group -- same "corroborate before trusting, else uniform"
    # philosophy already used elsewhere in this file (e.g. validate_credit).
    # Diagnostic info (stop_reason, content block types, or the JSON error)
    # is logged on every failed attempt so a recurring pattern stays
    # diagnosable instead of silently swallowed.
    # FIXED 2026-08-18 -- a third, distinct failure mode, found live on
    # pod-shared-0: `client.messages.create` itself can raise
    # `anthropic.OverloadedError` (HTTP 529, real Anthropic-side capacity
    # pressure, not a bug in our request) -- unlike (a)/(b) above, this
    # happens BEFORE any response exists, so it needs its own try/except
    # around the call, not just around parsing. Same "corroborate before
    # trusting, else uniform" fallback applies once retries are exhausted;
    # a short sleep before retrying gives transient overload a chance to
    # clear instead of hammering an already-overloaded endpoint immediately.
    parsed = None
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=REFLECTION_MODEL,
                max_tokens=4096,
                output_config={"format": {"type": "json_schema", "schema": _build_credit_schema(modules)}},
                messages=[{"role": "user", "content": prompt}],
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            print(
                f"[reflect_credit] attempt {attempt + 1}/2: API call failed ({e!r}) -- "
                f"{'retrying after a short backoff' if attempt == 0 else 'falling back to uniform credit'}"
            )
            if attempt == 0:
                time.sleep(5.0)
            continue
        latency = time.monotonic() - t0
        METER.record(
            model=REFLECTION_MODEL,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            tag=_tag_ctx.get(),
        )

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            print(
                f"[reflect_credit] attempt {attempt + 1}/2: no text content block in response "
                f"(stop_reason={response.stop_reason!r}, content_types={[b.type for b in response.content]}) "
                f"-- {'retrying' if attempt == 0 else 'falling back to uniform credit'}"
            )
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            print(
                f"[reflect_credit] attempt {attempt + 1}/2: JSON parse failed ({e}, "
                f"stop_reason={response.stop_reason!r}) -- "
                f"{'retrying' if attempt == 0 else 'falling back to uniform credit'}"
            )
            continue
        break

    if parsed is None:
        # time.monotonic() - t0 recomputed fresh here rather than reusing the
        # loop-local `latency` var: if every attempt fails inside the
        # try/except around client.messages.create (e.g. sustained 529s on
        # both tries), that var is never assigned, and referencing it would
        # raise UnboundLocalError -- an even worse crash than the one this
        # fallback exists to prevent.
        uniform = {m: 1.0 / len(modules) for m in modules}
        return ReflectionResult(
            raw_credit=uniform, credit=uniform,
            reasoning="[reflect_credit fallback: no valid parsed response after 2 attempts]",
            cost_usd=0.0, latency_s=time.monotonic() - t0,
        )

    raw_credit = {m: float(parsed["credit"].get(m, 0.0)) for m in modules}
    total = sum(raw_credit.values())
    if total <= 0:
        normalized = {m: 1.0 / len(modules) for m in modules}
    else:
        normalized = {m: v / total for m, v in raw_credit.items()}

    return ReflectionResult(
        raw_credit=raw_credit,
        credit=normalized,
        reasoning=parsed.get("reasoning", ""),
        cost_usd=0.0,  # filled in by caller via cost-log tag lookup, same pattern as scripts/faultline/score_credit_methods.py
        latency_s=latency,
    )


def validate_credit(
    credit: dict[str, float],
    example,
    baseline_pipeline,
    faulty_pipeline,
    run_fn,
    score_fn,
    fail_threshold: float = 0.35,
    success_threshold: float = 0.5,
) -> tuple[dict[str, float], bool]:
    """Counterfactual validation: patch the TOP-credited module back to its
    baseline (correct) predictor, keeping every other module as the faulty
    pipeline has it, rerun, and check whether the outcome actually improves.
    Returns (validated_credit, was_validated). On failure, validated_credit
    is the uniform fallback -- this IS the "uniform-mask fallback when
    validation fails" the plan's own §4 describes, not a separate code path.

    Reuses the exact same patch-to-baseline mechanism as
    reflectrl/faultline/credit_methods.py's Shapley scorer (swap in
    baseline_pipeline's own predictor object for one module attribute),
    applied here to just the ONE module RCA's own reflection nominated,
    rather than exhaustively testing every module like Shapley does."""
    import copy

    modules = list(credit.keys())
    top_module = max(credit, key=credit.get)

    faulty_score = score_fn(example, run_fn(faulty_pipeline, example))
    patched = copy.copy(faulty_pipeline)
    patched.__dict__[top_module] = baseline_pipeline.__dict__[top_module]
    patched_score = score_fn(example, run_fn(patched, example))

    validated = patched_score > faulty_score and patched_score >= success_threshold
    if validated:
        return credit, True
    uniform = {m: 1.0 / len(modules) for m in modules}
    return uniform, False


def compute_advantage_mask(
    module_token_spans: dict[str, tuple[int, int]],
    credit: dict[str, float],
    total_length: int,
    scalar_advantage: float,
) -> np.ndarray:
    """Per §4: token in module i's span -> scalar_advantage * k * credit[i].
    Tokens not covered by any module's span (prompt/template tokens, or any
    gap) get 0 -- only assistant-generated completion tokens should carry
    gradient signal in a GRPO-style update; this mask is applied on top of,
    not instead of, the trainer's own completion-token mask.

    k = number of modules in `credit`, NOT the number of spans provided --
    a module can be given credit even if, for some reason, its span isn't
    present in module_token_spans (defensive; shouldn't happen in practice
    but the formula's k must match the credit distribution's support)."""
    k = len(credit)
    mask = np.zeros(total_length, dtype=float)
    for module, (start, end) in module_token_spans.items():
        if not (0 <= start <= end <= total_length):
            raise ValueError(f"span for {module!r} ({start}, {end}) out of bounds for total_length={total_length}")
        c_i = credit.get(module, 0.0)
        mask[start:end] = scalar_advantage * k * c_i
    return mask


# ---------------------------------------------------------------------------
# Ablation variants (plan's Step 2.7): each takes the same inputs and
# returns a mask-ready credit distribution, so the trainer integration
# doesn't need to know which ablation is active -- only which function built
# the credit dict.
# ---------------------------------------------------------------------------

def credit_rca(reflection: ReflectionResult, validation_args: dict) -> dict[str, float]:
    """Full RCA: reflect, then validate. `validation_args` are passed to
    validate_credit (example/pipelines/run_fn/score_fn)."""
    validated, _ = validate_credit(reflection.credit, **validation_args)
    return validated


def credit_rca_no_validate(reflection: ReflectionResult, validation_args: dict) -> dict[str, float]:
    """Ablation: trust the reflection LM's credit distribution blindly, skip
    counterfactual validation entirely. Tests whether validation is doing
    real work (per the pre-registration's falsification check, proposal §1
    Test B in pre-registration.md: RCA-noValidate should track faithfulness
    WORSE than validated RCA, or the validation step isn't earning its
    keep)."""
    return reflection.credit


def credit_uniform_mask(modules: list[str], **_ignored) -> dict[str, float]:
    """Ablation: c~_i = 1/k for every module, which compute_advantage_mask
    reduces to plain scalar_advantage everywhere -- i.e., this ablation IS
    vanilla/Dr. GRPO's own per-token treatment, used as the sanity-check
    baseline within the RCA framework rather than a separate code path."""
    return {m: 1.0 / len(modules) for m in modules}
