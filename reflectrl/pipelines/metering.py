"""
Cost meter every model call in this project routes through (Step 1.2).
No exceptions — the plan's cost-honest headline claim lives in this plumbing.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

# $ per million tokens (input, output). Batch API is a flat 50% discount on both.
# Sonnet 5 intro pricing ($2/$10) holds through 2026-08-31; list price ($3/$15) after.
PRICING_PER_MTOK = {
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}
BATCH_DISCOUNT = 0.5

_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "cost_log.jsonl"
_tag_ctx: ContextVar[str] = ContextVar("cost_meter_tag", default="untagged")


@dataclass
class CostMeter:
    hard_cap_usd: float = 2500.0
    alarm_fraction: float = 0.8
    _lock: Lock = field(default_factory=Lock, repr=False)
    _total_usd: float = 0.0
    _alarm_fired: bool = False

    def _price(self, model: str, input_tokens: int, output_tokens: int, batch: bool) -> float:
        if model not in PRICING_PER_MTOK:
            raise ValueError(f"no pricing entry for model {model!r} — add it to PRICING_PER_MTOK")
        rates = PRICING_PER_MTOK[model]
        cost = (input_tokens / 1e6) * rates["input"] + (output_tokens / 1e6) * rates["output"]
        return cost * BATCH_DISCOUNT if batch else cost

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        batch: bool = False,
        tag: str | None = None,
    ) -> float:
        cost = self._price(model, input_tokens, output_tokens, batch)
        resolved_tag = tag or _tag_ctx.get()
        record = {
            "ts": time.time(),
            "tag": resolved_tag,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batch": batch,
            "latency_s": latency_s,
            "cost_usd": cost,
        }
        with self._lock:
            self._total_usd += cost
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            total = self._total_usd
            if total >= self.hard_cap_usd:
                raise RuntimeError(
                    f"COST METER HARD CAP HIT: ${total:,.2f} >= ${self.hard_cap_usd:,.2f}. "
                    f"Stopping — raise hard_cap_usd deliberately if this is expected."
                )
            if not self._alarm_fired and total >= self.hard_cap_usd * self.alarm_fraction:
                self._alarm_fired = True
                print(
                    f"[cost-meter ALARM] ${total:,.2f} has reached "
                    f"{self.alarm_fraction:.0%} of the ${self.hard_cap_usd:,.2f} cap "
                    f"(tag={resolved_tag})"
                )
        return cost

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd


METER = CostMeter()


@contextmanager
def cost_tag(tag: str):
    """Tag every meter.record() call made inside this block, e.g.
    with cost_tag("gepa-reflection"): ...
    """
    token = _tag_ctx.set(tag)
    try:
        yield
    finally:
        _tag_ctx.reset(token)


def metered_anthropic_call(client, *, tag: str | None = None, batch: bool = False, **create_kwargs):
    """Thin wrapper around client.messages.create(...) that meters cost.
    Every direct Anthropic call in this project should go through this
    function rather than calling client.messages.create(...) directly.
    """
    start = time.monotonic()
    response = client.messages.create(**create_kwargs)
    latency_s = time.monotonic() - start
    METER.record(
        model=create_kwargs["model"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_s=latency_s,
        batch=batch,
        tag=tag,
    )
    return response


def dspy_usage_callback():
    """DSPy callback (dspy.utils.callback.BaseCallback subclass). Verified
    empirically (2026-08-16 smoke test) that on_lm_end's `outputs` arg is
    just the decoded text list, and litellm's ModelResponse.usage comes back
    *empty* for Anthropic calls in this environment/litellm version --
    there are no per-token counts to read at the callback site. This
    callback is therefore a no-op hook kept for future litellm versions that
    populate usage correctly; real metering happens via flush_lm_history()
    below, called after every dspy.Predict/ChainOfThought invocation site.
    Do not rely on this callback alone -- see pipelines using it for the
    required flush_lm_history() pairing.
    """
    import dspy
    from dspy.utils.callback import BaseCallback

    class MeteringCallback(BaseCallback):
        def on_lm_end(self, call_id, outputs, exception):
            pass  # see docstring -- no usable usage data at this hook

    return MeteringCallback()


def flush_lm_history(lm, tag: str | None = None) -> int:
    """Scans `lm.history` for entries not yet metered and records them.
    Call this after every pipeline forward() pass (once per DSPy program
    run is enough -- it drains everything accumulated since the last flush).

    Cost source: DSPy/litellm's own `cost` field on each history entry.
    KNOWN LIMITATION (logged honestly, not silently): litellm's Anthropic
    usage payload was empty in the 2026-08-16 smoke test, so token counts
    are unavailable at this hook and we cannot cross-check litellm's cost
    estimate against PRICING_PER_MTOK. The logged records are tagged
    pricing_source="litellm_estimate" so this is auditable rather than
    presented as exact. Revisit before Week 2 sweeps: either pin a litellm
    version that populates usage for Anthropic, or read cost from the
    Anthropic API response directly (bypass DSPy's LM wrapper) for the
    judge/reflection calls where dollar-accounting precision matters most.
    """
    seen_attr = "_cost_meter_flushed_upto"
    resolved_tag = tag or _tag_ctx.get()
    n_recorded = 0
    # Whole read-slice-log-advance sequence must be atomic per lm instance:
    # dspy.Evaluate runs forward() concurrently across threads (num_threads),
    # and each forward() calls this on the same shared lm object. Reading
    # `start` outside the lock let two threads both slice from the same
    # offset and double-count (or, depending on interleaving, drop) entries.
    with METER._lock:
        start = getattr(lm, seen_attr, 0)
        history = lm.history
        for entry in history[start:]:
            cost = entry.get("cost") or 0.0
            usage = entry.get("usage") or {}
            record = {
                "ts": entry.get("timestamp", time.time()),
                "tag": resolved_tag,
                "model": entry.get("model", "unknown"),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "batch": False,
                "latency_s": 0.0,
                "cost_usd": cost,
                "pricing_source": "litellm_estimate",
            }
            METER._total_usd += cost
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            n_recorded += 1
        total = METER._total_usd
        if total >= METER.hard_cap_usd:
            raise RuntimeError(
                f"COST METER HARD CAP HIT: ${total:,.2f} >= ${METER.hard_cap_usd:,.2f}."
            )
        if not METER._alarm_fired and total >= METER.hard_cap_usd * METER.alarm_fraction:
            METER._alarm_fired = True
            print(f"[cost-meter ALARM] ${total:,.2f} has reached the alarm threshold (tag={resolved_tag})")
        setattr(lm, seen_attr, len(history))
    return n_recorded
