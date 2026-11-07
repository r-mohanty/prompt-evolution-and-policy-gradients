"""
Step 2.7 -- Validates RCA's reflection + counterfactual-validation mechanism
against FaultLine's real ground truth (the 69 instances from Step 2.6).
Adds RCA as a comparable 5th "credit method" alongside Step 2.6's four
(flat reflection, recursive reflector, Shapley, uniform) -- same accuracy
metric (does top-credit module match instance.target_module), same cost
tracking. Also reports validation behavior specifically: how often
counterfactual validation ACCEPTS the reflection's own top pick vs falls
back to uniform, which is the direct test of whether validation earns its
keep (pre-registration.md, P3's falsification check).

Run manually:
    python scripts/faultline/rca_faultline_eval.py [--n N]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

import anthropic
import dspy

from reflectrl.training.rca import reflect_credit, validate_credit
from reflectrl.pipelines.metering import METER, cost_tag
from reflectrl.faultline.verifier import FaultLineInstance
from reflectrl.pipelines.legibility import format_trajectory_for_judge

RESULTS_DIR = WORKSHOP_ROOT / "results"
HOTPOTQA_MODULES = ["generate_query", "generate_answer"]
AIME_MODULES = ["plan", "solve", "verify"]


def load_instances() -> list[FaultLineInstance]:
    data = json.loads((RESULTS_DIR / "faultline_instances.json").read_text())
    return [FaultLineInstance(**row) for row in data["instances"]]


def _configure_lm_for_task(task: str) -> None:
    """Matches scripts/faultline/generate_instances.py's own model choice per task
    (Sonnet 5 for HotpotQA, Haiku 4.5 for AIME-style) -- BUG FOUND
    2026-08-16: without this, dspy.settings.lm was never configured at all
    in this script (ValueError: No LM is loaded), and even after adding
    *some* configuration, using one global model for both tasks would
    silently rerun AIME instances on the wrong model relative to how they
    were originally generated. Called once per instance right before
    rebuilding its pipeline, since instances alternate between tasks."""
    if task == "hotpotqa":
        dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-5", max_tokens=1024, api_key=os.environ["ANTHROPIC_API_KEY"]))
    else:
        dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5", max_tokens=2048, api_key=os.environ["ANTHROPIC_API_KEY"]))


def rebuild_example_and_pipeline(instance: FaultLineInstance):
    _configure_lm_for_task(instance.task)
    if instance.task == "hotpotqa":
        from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset

        _, devset = load_devset(n_train=0, n_dev=max(60, instance.example_id + 1))
        example = devset[instance.example_id]
        pipeline = HotpotQAPipeline(k=8)

        def run_fn(p, ex):
            return p(question=ex.question, context=ex.context)

        return example, pipeline, run_fn, em_f1_metric, HOTPOTQA_MODULES
    else:
        from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric, load_devset

        _, devset = load_devset(n_train=0, n_dev=max(60, instance.example_id + 1))
        example = devset[instance.example_id]
        pipeline = AIMEStylePipeline()

        def run_fn(p, ex):
            return p(question=ex.question)

        return example, pipeline, run_fn, answer_check_metric, AIME_MODULES


def eval_one(i: int, instance: FaultLineInstance, client: anthropic.Anthropic) -> dict:
    all_modules = HOTPOTQA_MODULES if instance.task == "hotpotqa" else AIME_MODULES
    gold_module = instance.target_module
    tag = f"rca-eval-{i}"

    try:
        with cost_tag(tag):
            reflection = reflect_credit(
                question=instance.question,
                gold_answer=instance.gold_answer,
                predicted_answer=instance.faulty_answer,
                trajectory_text=format_trajectory_for_judge(instance.faulty_trajectory),
                modules=all_modules,
                client=client,
            )

            example, pipeline, run_fn, score_fn, _ = rebuild_example_and_pipeline(instance)
            import sys as _sys
            from reflectrl.faultline.injectors import inject_fault

            faulty_pipeline = inject_fault(pipeline, instance.target_module, instance.operator, None)
            validated_credit, was_validated = validate_credit(
                reflection.credit, example, pipeline, faulty_pipeline, run_fn, score_fn,
            )
        top_module_raw = max(reflection.credit, key=reflection.credit.get)
        top_module_validated = max(validated_credit, key=validated_credit.get)
        return {
            "instance_idx": i, "task": instance.task, "gold": gold_module,
            "raw_top_module": top_module_raw, "raw_correct": top_module_raw == gold_module,
            "validated_top_module": top_module_validated, "validated_correct": top_module_validated == gold_module,
            "was_validated": was_validated,
            "raw_credit": reflection.credit,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 -- a failure is data, not a crash (Step 2.6 lesson)
        return {
            "instance_idx": i, "task": instance.task, "gold": gold_module,
            "raw_top_module": None, "raw_correct": False,
            "validated_top_module": None, "validated_correct": False,
            "was_validated": False, "raw_credit": {},
            "error": f"{type(e).__name__}: {e}",
        }


def run(n: int | None = None):
    instances = load_instances()
    if n is not None:
        instances = instances[:n]
    print(f"Evaluating RCA reflection+validation on {len(instances)} FaultLine instances")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rows = []
    for i, instance in enumerate(instances):
        row = eval_one(i, instance, client)
        rows.append(row)
        status = "OK" if row["error"] is None else f"ERROR: {row['error'][:60]}"
        print(f"[{i + 1}/{len(instances)}] {instance.task} gold={row['gold']} "
              f"raw={row['raw_top_module']}({row['raw_correct']}) "
              f"validated={row['validated_top_module']}({row['validated_correct']}) "
              f"was_validated={row['was_validated']} {status}")

    ok_rows = [r for r in rows if r["error"] is None]
    n_ok = len(ok_rows)
    raw_accuracy = sum(1 for r in ok_rows if r["raw_correct"]) / n_ok if n_ok else float("nan")
    validated_accuracy = sum(1 for r in ok_rows if r["validated_correct"]) / n_ok if n_ok else float("nan")
    validation_accept_rate = sum(1 for r in ok_rows if r["was_validated"]) / n_ok if n_ok else float("nan")

    out = {
        "n_instances": len(rows),
        "n_errors": len(rows) - n_ok,
        "raw_reflection_accuracy": raw_accuracy,
        "validated_accuracy": validated_accuracy,
        "validation_accept_rate": validation_accept_rate,
        "rows": rows,
        "total_project_cost_usd_at_completion": METER.total_usd,
    }
    out_path = RESULTS_DIR / "rca_faultline_eval.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nRaw reflection accuracy (before validation): {raw_accuracy:.3f}")
    print(f"Validated accuracy: {validated_accuracy:.3f}")
    print(f"Validation accept rate (validation agreed with reflection's own top pick): {validation_accept_rate:.3f}")
    print(f"Written to {out_path}")
    print(f"Total running spend: ${METER.total_usd:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    run(args.n)
