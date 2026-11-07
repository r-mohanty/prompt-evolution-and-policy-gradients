"""
Step 2.6 -- Recursive/programmatic reflector (RLM-motivated variant).

Prediction under test (proposal §4): the recursive reflector maintains
higher measured localization accuracy than the flat reflector at long task
horizons, because it can zoom into specific modules' spans on demand rather
than reading the entire trace in one shot -- avoiding the "context rot" RLM
(Recursive Language Models) identifies in flat long-context reading.

SCOPE NOTE, documented rather than silently assumed: the plan's own wording
says the reflector is "given the trajectory as a queryable external object
via a code-execution tool." This implementation uses a lighter-weight
**custom tool** (`inspect_module`) instead of Claude's actual sandboxed
code-execution tool -- the reflector still queries the trajectory
on-demand/recursively rather than reading it flat, which is the mechanism
the prediction is actually about, but without sandbox provisioning
complexity for what the plan's own cut-list already marks as the
lowest-priority Week-2 item (drop second under schedule pressure, after the
coding-agent stretch domain). If literal code-execution-tool fidelity turns
out to matter for the paper's methods section, swap the tool declaration for
`{"type": "code_execution_20260521", "name": "code_execution"}` and give the
model the trajectory as an uploaded file -- the rest of the calling
convention here (CreditPrediction output) doesn't need to change.

Uses the Anthropic SDK directly (not DSPy) -- DSPy's tool-calling loop
abstractions don't map cleanly onto "give the model a custom tool and let it
call it recursively," and this method's whole point is measuring how a
*different interaction shape* (recursive/on-demand vs flat/one-shot) affects
localization accuracy, so it deliberately isn't built on the same DSPy
ChainOfThought machinery the flat reflector (legibility_judge.py) uses.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anthropic

from reflectrl.pipelines.metering import METER, _tag_ctx  # noqa: E402

from reflectrl.faultline.credit_methods import CreditPrediction
from reflectrl.faultline.verifier import FaultLineInstance

INSPECT_MODULE_TOOL = {
    "name": "inspect_module",
    "description": (
        "Look up the full detail (inputs, reasoning, outputs) for ONE module in the "
        "pipeline trajectory by name. Use this to zoom into a specific module instead "
        "of assuming you can see everything at once -- you were only given each "
        "module's name and a short output preview up front."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"module_name": {"type": "string", "description": "the module name to inspect"}},
        "required": ["module_name"],
    },
}

SUBMIT_TOOL = {
    "name": "submit_localization",
    "description": "Submit your final answer: which module is responsible for the failure, and your confidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "failing_module": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["failing_module", "confidence", "reasoning"],
    },
}


def _trajectory_summary(trajectory: list[dict]) -> str:
    lines = []
    for step in trajectory:
        preview = str(step.get("outputs", {}))[:80]
        lines.append(f"- module '{step['module']}': output preview = {preview}...")
    return "\n".join(lines)


def _module_detail(trajectory: list[dict], module_name: str) -> str:
    for step in trajectory:
        if step["module"] == module_name:
            return json.dumps(step, indent=2)
    return f"No module named {module_name!r} in this trajectory."


def score_recursive_reflector(
    instance: FaultLineInstance,
    all_modules: list[str],
    model: str = "claude-sonnet-5",
    max_turns: int = 6,
) -> CreditPrediction:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.monotonic()

    system = (
        "You are auditing a failed run of a multi-module AI pipeline, same task as a "
        "standard failure-localization judge, but you do NOT get the full trajectory "
        "up front. You get a one-line preview per module and must call `inspect_module` "
        "to zoom into any module's full detail before deciding. Only inspect modules you "
        "actually need to reason about the causal chain -- don't inspect all of them "
        "reflexively. When ready, call `submit_localization`."
    )
    user_content = (
        f"Question: {instance.question}\n"
        f"Gold answer: {instance.gold_answer}\n"
        f"Predicted (wrong) answer: {instance.faulty_answer}\n"
        f"Modules in pipeline order: {', '.join(all_modules)}\n\n"
        f"Trajectory preview (one line per module):\n{_trajectory_summary(instance.faulty_trajectory)}"
    )
    messages = [{"role": "user", "content": user_content}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            tools=[INSPECT_MODULE_TOOL, SUBMIT_TOOL],
            messages=messages,
        )
        # BUG FOUND 2026-08-16 (Step 2.6 scoring dry run): this function
        # calls the Anthropic SDK directly, bypassing DSPy entirely -- so it
        # was never going through metering.py's flush_lm_history at all,
        # meaning the recursive reflector's real API spend was silently
        # never logged anywhere. Recorded directly here instead; unlike the
        # DSPy/litellm wrapper (which had empty usage -- see decisions.md,
        # Step 1.3), a raw Anthropic SDK response's usage IS populated, so
        # this is exact, not an estimate.
        METER.record(
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=0.0,
            tag=_tag_ctx.get(),
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        submit_call = next((b for b in tool_uses if b.name == "submit_localization"), None)
        if submit_call is not None:
            latency = time.monotonic() - t0
            return CreditPrediction(
                predicted_module=submit_call.input.get("failing_module", all_modules[0]),
                confidence=float(submit_call.input.get("confidence", 0.5)),
                cost_usd=0.0,
                latency_s=latency,
                method="recursive_reflector",
            )

        if not tool_uses:
            break  # model gave up without submitting -- fall through to the fallback below

        tool_results = []
        for call in tool_uses:
            if call.name == "inspect_module":
                result_text = _module_detail(instance.faulty_trajectory, call.input.get("module_name", ""))
            else:
                result_text = "unknown tool"
            tool_results.append({"type": "tool_result", "tool_use_id": call.id, "content": result_text})
        messages.append({"role": "user", "content": tool_results})

    # Exhausted max_turns (or the model stalled) without submitting -- report
    # this as a real, honest low-confidence fallback rather than crash or
    # silently default to a specific module. This itself is a data point
    # (recursive reflectors that fail to converge in max_turns) worth
    # tracking in the scored results, not hidden.
    latency = time.monotonic() - t0
    return CreditPrediction(
        predicted_module=all_modules[0],
        confidence=0.0,
        cost_usd=0.0,
        latency_s=latency,
        method="recursive_reflector_no_submit",
    )
