"""
Step 2.6 -- Score all four credit-assignment methods against the FaultLine
instance set. Localization accuracy, calibration (ECE), and cost per method,
per the plan's §5/§6.

Ground truth for each instance is instance.target_module -- the module the
generation harness (Step 2.5) deliberately corrupted and verified (baseline
succeeds, fault breaks it, no other module's corruption does) is the
responsible one. This is exactly the counterfactual ground truth the
legibility judge's rubric doc (Step 1.6) says L must eventually be validated
against -- this script is that validation becoming possible for the first
time, now that FaultLine instances exist.

Run manually (needs results/faultline_instances.json to exist --
run scripts/faultline/generate_instances.py first):
    python scripts/faultline/score_credit_methods.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

import dspy

from reflectrl.faultline.credit_methods import CreditPrediction, score_flat_reflection, score_shapley, score_uniform
from reflectrl.faultline.credit_recursive_reflector import score_recursive_reflector
from reflectrl.pipelines.legibility_judge import LegibilityJudge, configure_judge_lm
from reflectrl.pipelines.metering import METER, cost_tag
from reflectrl.faultline.verifier import FaultLineInstance

RESULTS_DIR = WORKSHOP_ROOT / "results"
INSTANCES_PATH = RESULTS_DIR / "faultline_instances.json"

HOTPOTQA_MODULES = ["generate_query", "generate_answer"]
AIME_MODULES = ["plan", "solve", "verify"]

# Judge model for the flat reflector -- frozen per Step 1.6's bake-off signal
# (Haiku 4.5 vs Sonnet 5 showed 93.3% inter-judge agreement; using Sonnet 5
# here since the human-label criterion that would justify the cheaper model
# is still an open item -- see paper/legibility_judge_rubric.md).
JUDGE_MODEL = "anthropic/claude-sonnet-5"


def load_instances() -> list[FaultLineInstance]:
    data = json.loads(INSTANCES_PATH.read_text())
    return [FaultLineInstance(**row) for row in data["instances"]]


def rebuild_example_and_pipeline(instance: FaultLineInstance):
    """Reconstructs enough of the original example + baseline pipeline for
    Shapley's rerun-based method. Needs the ORIGINAL example object (not
    just question/gold_answer), which FaultLineInstance doesn't store in
    full (e.g. HotpotQA's `context`) -- re-fetched from load_devset() using
    the same fixed-seed shuffle, keyed by example_id, so it's the identical
    object generate_instances.py used."""
    if instance.task == "hotpotqa":
        from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset

        _, devset = load_devset(n_train=0, n_dev=max(60, instance.example_id + 1))
        example = devset[instance.example_id]
        pipeline = HotpotQAPipeline(k=8)

        def run_fn(p, ex):
            return p(question=ex.question, context=ex.context)

        from reflectrl.faultline.injectors import HOTPOTQA_SWAP_TERMS

        return example, pipeline, run_fn, em_f1_metric, HOTPOTQA_MODULES, HOTPOTQA_SWAP_TERMS
    else:
        from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric, load_devset

        _, devset = load_devset(n_train=0, n_dev=max(60, instance.example_id + 1))
        example = devset[instance.example_id]
        pipeline = AIMEStylePipeline()

        def run_fn(p, ex):
            return p(question=ex.question)

        from reflectrl.faultline.injectors import AIME_SWAP_TERMS

        return example, pipeline, run_fn, answer_check_metric, AIME_MODULES, AIME_SWAP_TERMS


def expected_calibration_error(predictions: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """Standard ECE: bin by confidence, compare mean confidence to accuracy
    within each bin, weight by bin size."""
    if not predictions:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    total = len(predictions)
    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        bin_conf = sum(c for c, _ in bin_items) / len(bin_items)
        bin_acc = sum(1 for _, correct in bin_items if correct) / len(bin_items)
        ece += (len(bin_items) / total) * abs(bin_conf - bin_acc)
    return ece


def _cost_for_tag(tag: str) -> float:
    """Sums logs/cost_log.jsonl entries with an exact tag match. Used
    instead of a live METER.total_usd before/after delta -- under the
    thread-pool parallelism below, a global-total delta would be
    cross-contaminated by OTHER threads' concurrent spending happening
    between one thread's before/after reads (the shared counter itself is
    already lock-safe per the Step 1.4 fix, but "safe to read concurrently"
    is not the same as "accurate as a per-call delta under concurrency").
    Per-instance-per-method unique tags sidestep the problem entirely: each
    tag can only have been written by the one call that used it."""
    from reflectrl.pipelines.metering import _LOG_PATH

    total = 0.0
    if not _LOG_PATH.exists():
        return 0.0
    with open(_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("tag") == tag:
                total += rec["cost_usd"]
    return total


def _error_row(i: int, instance: FaultLineInstance, gold_module: str, method: str, cost_usd: float, error: Exception) -> dict:
    """A failed prediction is real data (an API hiccup, an empty LM
    response, a rate limit) -- recorded as an explicit failure, not
    silently dropped and not allowed to crash the whole batch. Excluded
    from accuracy/ECE aggregation (see run()'s summary loop) but kept in
    per_instance output and counted separately so failure RATE per method
    is itself visible, which is honest-analysis-relevant (a method that
    fails to produce an answer 10% of the time is a real cost/reliability
    finding, not noise to hide)."""
    return {
        "instance_idx": i, "task": instance.task, "gold": gold_module,
        "predicted": None, "correct": False, "confidence": 0.0,
        "cost_usd": cost_usd, "latency_s": 0.0,
        "error": f"{type(error).__name__}: {error}",
    }


def _score_one_instance(i: int, instance: FaultLineInstance, judge: LegibilityJudge, judge_lm) -> dict[str, dict]:
    all_modules = HOTPOTQA_MODULES if instance.task == "hotpotqa" else AIME_MODULES
    gold_module = instance.target_module
    row = {}

    flat_tag = f"faultline-score-flat-{i}"
    try:
        with cost_tag(flat_tag):
            pred = score_flat_reflection(instance, judge, judge_lm, all_modules)
        pred.cost_usd = _cost_for_tag(flat_tag)
        row["flat_reflection"] = {
            "instance_idx": i, "task": instance.task, "gold": gold_module,
            "predicted": pred.predicted_module, "correct": pred.predicted_module == gold_module,
            "confidence": pred.confidence, "cost_usd": pred.cost_usd, "latency_s": pred.latency_s,
        }
    except Exception as e:  # noqa: BLE001 -- see _error_row docstring: a failure is data, not a crash
        row["flat_reflection"] = _error_row(i, instance, gold_module, "flat_reflection", _cost_for_tag(flat_tag), e)

    recursive_tag = f"faultline-score-recursive-{i}"
    try:
        with cost_tag(recursive_tag):
            pred = score_recursive_reflector(instance, all_modules)
        pred.cost_usd = _cost_for_tag(recursive_tag)
        row["recursive_reflector"] = {
            "instance_idx": i, "task": instance.task, "gold": gold_module,
            "predicted": pred.predicted_module, "correct": pred.predicted_module == gold_module,
            "confidence": pred.confidence, "cost_usd": pred.cost_usd, "latency_s": pred.latency_s,
        }
    except Exception as e:  # noqa: BLE001
        row["recursive_reflector"] = _error_row(i, instance, gold_module, "recursive_reflector", _cost_for_tag(recursive_tag), e)

    shapley_tag = f"faultline-score-shapley-{i}"
    try:
        example, pipeline, run_fn, score_fn, _, swap_terms_map = rebuild_example_and_pipeline(instance)
        with cost_tag(shapley_tag):
            pred = score_shapley(instance, example, pipeline, run_fn, score_fn, all_modules, instance.target_module, instance.operator, swap_terms_map)
        pred.cost_usd = _cost_for_tag(shapley_tag)
        row["shapley_ablate_rescore"] = {
            "instance_idx": i, "task": instance.task, "gold": gold_module,
            "predicted": pred.predicted_module, "correct": pred.predicted_module == gold_module,
            "confidence": pred.confidence, "cost_usd": pred.cost_usd, "latency_s": pred.latency_s,
        }
    except Exception as e:  # noqa: BLE001
        row["shapley_ablate_rescore"] = _error_row(i, instance, gold_module, "shapley_ablate_rescore", _cost_for_tag(shapley_tag), e)

    pred = score_uniform(instance, all_modules)  # no LM calls -- can't fail the way the others can
    row["uniform"] = {
        "instance_idx": i, "task": instance.task, "gold": gold_module,
        "predicted": pred.predicted_module, "correct": pred.predicted_module == gold_module,
        "confidence": pred.confidence, "cost_usd": 0.0, "latency_s": 0.0,
    }
    return row


def run(num_workers: int = 6):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    instances = load_instances()
    print(f"Loaded {len(instances)} FaultLine instances")

    dspy.configure(lm=dspy.LM(JUDGE_MODEL, max_tokens=1024, api_key=os.environ["ANTHROPIC_API_KEY"]))
    judge = LegibilityJudge()
    judge_lm = configure_judge_lm(JUDGE_MODEL)

    results_by_method: dict[str, list[dict]] = {
        "flat_reflection": [], "recursive_reflector": [], "shapley_ablate_rescore": [], "uniform": [],
    }
    lock = threading.Lock()
    completed = 0

    checkpoint_path = RESULTS_DIR / "faultline_credit_scoring_checkpoint.json"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_score_one_instance, i, inst, judge, judge_lm): i for i, inst in enumerate(instances)}
        for future in as_completed(futures):
            # future.result() should not raise -- _score_one_instance
            # already catches per-method exceptions internally (the bug
            # this replaced: a single instance's failure used to crash the
            # entire batch, losing ~45 already-scored instances' worth of
            # spend along with it). Still guarded here as a last resort so
            # one truly unexpected failure logs and continues rather than
            # aborting the whole run a second way.
            try:
                row = future.result()
            except Exception as e:  # noqa: BLE001
                i = futures[future]
                print(f"  [WARNING] instance {i} raised outside _score_one_instance's own guards: {type(e).__name__}: {e}")
                continue
            with lock:
                for method, entry in row.items():
                    results_by_method[method].append(entry)
                completed += 1
                if completed % 5 == 0 or completed == len(instances):
                    print(f"{completed}/{len(instances)} instances scored, running spend: ${METER.total_usd:.3f}")
                if completed % 10 == 0:
                    checkpoint_path.write_text(json.dumps(results_by_method, indent=2))

    summary = {}
    for method, rows in results_by_method.items():
        ok_rows = [r for r in rows if r.get("error") is None]
        n_errors = len(rows) - len(ok_rows)
        n = len(ok_rows)
        accuracy = sum(1 for r in ok_rows if r["correct"]) / n if n else float("nan")
        ece = expected_calibration_error([(r["confidence"], r["correct"]) for r in ok_rows])
        total_cost = sum(r["cost_usd"] for r in rows)  # include failed calls' cost -- money was still spent
        summary[method] = {
            "n_instances": n,
            "n_errors": n_errors,
            "localization_accuracy": accuracy,
            "calibration_ece": ece,
            "total_cost_usd": total_cost,
            "mean_cost_usd_per_instance": total_cost / len(rows) if rows else 0.0,
        }

    out = {"summary": summary, "per_instance": results_by_method, "total_project_cost_usd_at_completion": METER.total_usd}
    out_path = RESULTS_DIR / "faultline_credit_scoring.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== FaultLine credit-method scoring summary ===")
    for method, s in summary.items():
        print(f"{method}: accuracy={s['localization_accuracy']:.3f} ECE={s['calibration_ece']:.3f} "
              f"cost/instance=${s['mean_cost_usd_per_instance']:.4f} (n={s['n_instances']}, errors={s['n_errors']})")
    print(f"\nWritten to {out_path}")
    print(f"Total running spend: ${METER.total_usd:.4f}")


if __name__ == "__main__":
    run()
