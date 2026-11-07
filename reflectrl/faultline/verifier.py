"""
Step 2.4 -- FaultLine's counterfactual verifier.

Ground-truth methodology (see paper/faultline_taxonomy.md "Validity
requirement" for the reasoning): an instance is only accepted if, on the
same example, (1) baseline succeeds, (2) the target-module-faulty pipeline
fails, and (3) corrupting every OTHER module (same operator, target module
left correct) still succeeds. Condition 3 is what rules out examples that
are simply fragile to any corruption rather than genuinely localizing the
fault to the named module -- this is the actual test the plan's "patching
module_i restores success; patching j!=i does not" language calls for.

No external substrate (Shepherd or otherwise) is used -- see decisions.md
2026-08-16, Step 1.7's no-go. This file *is* FaultLine's counterfactual
verifier; there is nothing else to build on top of.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from reflectrl.faultline.injectors import inject_fault  # noqa: E402


@dataclass
class FaultLineInstance:
    task: str
    example_id: int
    question: str
    gold_answer: str
    target_module: str
    operator: str
    baseline_answer: str
    baseline_score: float
    faulty_answer: str
    faulty_score: float
    faulty_trajectory: list[dict]
    other_module_checks: dict[str, float] = field(default_factory=dict)  # module -> score when THAT module corrupted instead (target left correct)


def try_build_instance(
    *,
    task: str,
    example_id: int,
    example,
    baseline_pipeline,
    all_modules: list[str],
    target_module: str,
    operator: str,
    run_fn: Callable,
    score_fn: Callable,
    success_threshold: float = 0.5,
    fail_threshold: float = 0.35,
    swap_terms: tuple[str, str] | None = None,
    other_module_swap_terms: dict[str, tuple[str, str]] | None = None,
) -> FaultLineInstance | None:
    """Attempts to build ONE validated FaultLine instance for one example
    under one (target_module, operator) fault. Returns None if any of the
    three validity conditions fails -- rejection is the common case, that's
    the point of the filter, not a bug."""
    baseline_pred = run_fn(baseline_pipeline, example)
    baseline_score = score_fn(example, baseline_pred)
    if baseline_score < success_threshold:
        return None  # condition 1 failed: baseline itself doesn't succeed here

    faulty_pipeline = inject_fault(baseline_pipeline, target_module, operator, swap_terms)
    faulty_pred = run_fn(faulty_pipeline, example)
    faulty_score = score_fn(example, faulty_pred)
    if faulty_score > fail_threshold:
        return None  # condition 2 failed: the fault didn't actually break it

    other_checks: dict[str, float] = {}
    for other_module in all_modules:
        if other_module == target_module:
            continue
        other_swap = (other_module_swap_terms or {}).get(other_module)
        other_faulty_pipeline = inject_fault(baseline_pipeline, other_module, operator, other_swap)
        other_pred = run_fn(other_faulty_pipeline, example)
        other_score = score_fn(example, other_pred)
        other_checks[other_module] = other_score
        if other_score < success_threshold:
            return None  # condition 3 failed: example is fragile to ANY corruption, not discriminative

    return FaultLineInstance(
        task=task,
        example_id=example_id,
        question=getattr(example, "question", ""),
        gold_answer=getattr(example, "answer", ""),
        target_module=target_module,
        operator=operator,
        baseline_answer=getattr(baseline_pred, "answer", ""),
        baseline_score=baseline_score,
        faulty_answer=getattr(faulty_pred, "answer", ""),
        faulty_score=faulty_score,
        faulty_trajectory=getattr(faulty_pred, "trajectory", []),
        other_module_checks=other_checks,
    )
