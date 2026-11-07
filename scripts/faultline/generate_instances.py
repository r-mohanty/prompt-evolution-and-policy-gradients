"""
Step 2.5 -- Generate + validate FaultLine instances.

Scans real devset examples for both tasks, tries every applicable
(target_module, operator) combination per example, keeps only instances that
pass all three validity conditions in verifier.py, and writes the validated
set to results/faultline_instances.json.

v2 (2026-08-16, scale-up pass): parallelized across examples with a thread
pool (the same pattern dspy.Evaluate itself uses -- concurrent calls on the
same dspy.Module instance are safe; metering.flush_lm_history's
read-slice-log-advance sequence is already lock-protected, see Step 1.4's
race-condition fix) and switched from a per-example combo subsample to
trying the FULL combo schedule per example. Rationale: baseline is cached
per example regardless, so once an example is baselined, the marginal cost
of trying more combos on it is small relative to paying a fresh baseline on
a new example -- exhausting each example is more information-efficient than
sampling a few combos across many examples. See decisions.md (2026-08-16,
"FaultLine v0 scale-up") for the acceptance-rate math this sizing is based
on.

Cost note: baseline is run once per example and cached across every
(module, operator) attempt on that example. "Other module" faulty checks are
not cached across attempts (bounded redundancy).

Run manually:
    python scripts/faultline/generate_instances.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

import dspy

from reflectrl.faultline.injectors import AIME_OPERATORS, AIME_SWAP_TERMS, HOTPOTQA_OPERATORS, HOTPOTQA_SWAP_TERMS
from reflectrl.faultline.verifier import try_build_instance
from reflectrl.pipelines.metering import METER, cost_tag

RESULTS_DIR = WORKSHOP_ROOT / "results"
# FaultLine generation needs a baseline pipeline that SUCCEEDS often enough
# to have candidates (it's building a diagnostic corpus, not sweeping
# optimizer cheapness) -- verified empirically (2026-08-16) on 8 sampled
# examples each: HotpotQA baseline succeeds (score>=0.5) only 1/8 on Haiku
# 4.5 vs 5/8 on Sonnet 5 -- needs the stronger model. AIME-style baseline
# already succeeds 7/8 on Haiku 4.5 -- no upgrade needed there.
HOTPOTQA_TASK_MODEL = "anthropic/claude-sonnet-5"
AIME_TASK_MODEL = "anthropic/claude-haiku-4-5"

N_EXAMPLES_PER_TASK = 60
MAX_COMBOS_PER_EXAMPLE = 100  # >= either schedule's length -> try every applicable combo per example
HOTPOTQA_K = 8
NUM_WORKERS = 8
CHECKPOINT_EVERY = 10  # examples, per task -- write partial results so a long run isn't all-or-nothing


def build_combo_schedule(operators_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Interleaves (module, operator) pairs round-robin across modules --
    BUG FOUND 2026-08-16: see git history / decisions.md for the full
    account of the break-order bug this fixed (a naive nested loop only
    ever reached the first module). Interleaving means even a small
    per-example budget samples every module; with MAX_COMBOS_PER_EXAMPLE
    now >= the schedule length, every example tries everything anyway, but
    the interleaved order is kept so a truncated/interrupted run still has
    balanced partial coverage."""
    from itertools import zip_longest

    per_module = [[(module, op) for op in ops] for module, ops in operators_map.items()]
    interleaved: list[tuple[str, str]] = []
    for group in zip_longest(*per_module):
        interleaved.extend(item for item in group if item is not None)
    return interleaved


def combos_for_example(schedule: list[tuple[str, str]], example_idx: int, max_combos: int) -> list[tuple[str, str]]:
    n = len(schedule)
    if max_combos >= n:
        return list(schedule)  # every combo, no repeats
    offset = (example_idx * max_combos) % n
    return [schedule[(offset + k) % n] for k in range(max_combos)]


def _run_task_parallel(*, task_name, devset, schedule, baseline_pipeline, all_modules, run_fn, score_fn, swap_terms_map, cost_tag_name, checkpoint_path):
    instances: list = []
    lock = threading.Lock()
    completed = 0

    def process_example(i, example):
        local = []
        for target_module, operator in combos_for_example(schedule, i, MAX_COMBOS_PER_EXAMPLE):
            with cost_tag(cost_tag_name):
                inst = try_build_instance(
                    task=task_name,
                    example_id=i,
                    example=example,
                    baseline_pipeline=baseline_pipeline,
                    all_modules=all_modules,
                    target_module=target_module,
                    operator=operator,
                    run_fn=run_fn,
                    score_fn=score_fn,
                    swap_terms=swap_terms_map.get(target_module),
                    other_module_swap_terms=swap_terms_map,
                )
            if inst is not None:
                local.append(inst)
        return i, local

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_example, i, ex): i for i, ex in enumerate(devset)}
        for future in as_completed(futures):
            i, local = future.result()
            with lock:
                instances.extend(local)
                completed += 1
                for inst in local:
                    print(f"  [{task_name}#{i}] ACCEPTED target={inst.target_module} op={inst.operator}")
                if completed % 5 == 0 or completed == len(devset):
                    print(f"{task_name}: {completed}/{len(devset)} examples done, {len(instances)} accepted so far, spend=${METER.total_usd:.3f}")
                if completed % CHECKPOINT_EVERY == 0:
                    checkpoint_path.write_text(json.dumps([asdict(x) for x in instances], indent=2))

    return instances


def run_hotpotqa_task():
    from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset

    dspy.configure(lm=dspy.LM(HOTPOTQA_TASK_MODEL, max_tokens=1024, api_key=os.environ["ANTHROPIC_API_KEY"]))
    _, devset = load_devset(n_train=0, n_dev=N_EXAMPLES_PER_TASK)
    baseline_pipeline = HotpotQAPipeline(k=HOTPOTQA_K)
    all_modules = list(HOTPOTQA_OPERATORS.keys())
    schedule = build_combo_schedule(HOTPOTQA_OPERATORS)

    def run_fn(pipeline, example):
        return pipeline(question=example.question, context=example.context)

    return _run_task_parallel(
        task_name="hotpotqa",
        devset=devset,
        schedule=schedule,
        baseline_pipeline=baseline_pipeline,
        all_modules=all_modules,
        run_fn=run_fn,
        score_fn=em_f1_metric,
        swap_terms_map=HOTPOTQA_SWAP_TERMS,
        cost_tag_name="faultline-generate-hotpotqa",
        checkpoint_path=RESULTS_DIR / "faultline_instances_checkpoint_hotpotqa.json",
    )


def run_aime_task():
    from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric, load_devset

    dspy.configure(lm=dspy.LM(AIME_TASK_MODEL, max_tokens=2048, api_key=os.environ["ANTHROPIC_API_KEY"]))
    _, devset = load_devset(n_train=0, n_dev=N_EXAMPLES_PER_TASK)
    baseline_pipeline = AIMEStylePipeline()
    all_modules = list(AIME_OPERATORS.keys())
    schedule = build_combo_schedule(AIME_OPERATORS)

    def run_fn(pipeline, example):
        return pipeline(question=example.question)

    return _run_task_parallel(
        task_name="aime",
        devset=devset,
        schedule=schedule,
        baseline_pipeline=baseline_pipeline,
        all_modules=all_modules,
        run_fn=run_fn,
        score_fn=answer_check_metric,
        swap_terms_map=AIME_SWAP_TERMS,
        cost_tag_name="faultline-generate-aime",
        checkpoint_path=RESULTS_DIR / "faultline_instances_checkpoint_aime.json",
    )


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== HotpotQA FaultLine generation (N={N_EXAMPLES_PER_TASK}, {NUM_WORKERS} workers) ===")
    hotpotqa_instances = run_hotpotqa_task()

    print(f"\n=== AIME-style FaultLine generation (N={N_EXAMPLES_PER_TASK}, {NUM_WORKERS} workers) ===")
    aime_instances = run_aime_task()

    all_instances = hotpotqa_instances + aime_instances
    out = {
        "n_instances": len(all_instances),
        "n_hotpotqa": len(hotpotqa_instances),
        "n_aime": len(aime_instances),
        "target_from_plan": 100,
        "instances": [asdict(inst) for inst in all_instances],
        "total_cost_usd_at_completion": METER.total_usd,
    }
    out_path = RESULTS_DIR / "faultline_instances.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n{len(all_instances)} validated instances ({len(hotpotqa_instances)} hotpotqa, {len(aime_instances)} aime)")
    print(f"Written to {out_path}")
    print(f"Total running spend: ${METER.total_usd:.4f}")


if __name__ == "__main__":
    run()
