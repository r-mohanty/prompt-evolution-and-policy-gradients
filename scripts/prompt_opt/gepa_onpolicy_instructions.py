"""
Evolve per-module instructions with GEPA using Qwen3-8B ITSELF as the task model
(2026-09-03). Closes the prompt-model confound in the paper's regime-boundary result:
the existing gepa_then_grpo arm evolved its instructions against Claude Haiku 4.5 (the
configuration of gepa_repro.py, kept for fidelity to GEPA's published setup) and then
transplanted them into the Qwen3-8B policy. Here the task model that GEPA optimises
against IS the policy that will subsequently be trained, so any gain or loss after
Dr. GRPO training can no longer be attributed to a mismatch between the model the
prompt was tuned for and the model that runs it.

Everything else is held identical to run_gepa_for_instructions() in
multi_module_grpo_train.py: the same DSPy pipelines, the same 15 train / 20 validation
examples from the same fixed-seed loader, the same reflection model (Claude Sonnet 5,
4096 tokens), the same max_metric_calls=100 and reflection_minibatch_size=3. The only
change is the task LM, which points at a local vLLM OpenAI-compatible server. Thinking
mode is disabled through the chat template, matching the trainer's enable_thinking=False
so GEPA evaluates the same behaviour the trainer will roll out.

Requires a running server on the pod, e.g. (one GPU, before any training job starts):
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.85 \
        --max-model-len 8192 --dtype bfloat16

Then, one process per seed (they share the server):
    python scripts/prompt_opt/gepa_onpolicy_instructions.py --task hotpotqa --seed 0

Writes data/gepa_onpolicy/instructions/{task}_seed{seed}.json, which
multi_module_grpo_train.py --variant gepa_onpolicy_then_grpo reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

OUT_DIR = WORKSHOP_ROOT / "data" / "gepa_onpolicy" / "instructions"
REFLECTION_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_TASK_MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["hotpotqa", "aime"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=DEFAULT_TASK_MODEL)
    parser.add_argument("--max-metric-calls", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    import dspy

    from reflectrl.pipelines.metering import METER, cost_tag, flush_lm_history

    # Same per-task generation budgets as run_gepa_for_instructions() gave Haiku.
    max_tokens = 1024 if args.task == "hotpotqa" else 2048
    task_lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.api_base,
        api_key="EMPTY",
        max_tokens=max_tokens,
        # Matches the trainer, which renders every rollout with enable_thinking=False.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    reflection_lm = dspy.LM(REFLECTION_MODEL, max_tokens=4096, api_key=os.environ["ANTHROPIC_API_KEY"])
    dspy.configure(lm=task_lm)

    if args.task == "hotpotqa":
        from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset

        trainset, rest = load_devset(n_train=15, n_dev=20)
        baseline = HotpotQAPipeline(k=5)
        score_fn = em_f1_metric
        modules = ["generate_query", "generate_answer"]
    else:
        from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric, load_devset

        trainset, rest = load_devset(n_train=15, n_dev=20)
        baseline = AIMEStylePipeline()
        score_fn = answer_check_metric
        modules = ["plan", "solve", "verify"]

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
        return score_fn(gold, pred)

    t0 = time.monotonic()
    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=3,
        num_threads=args.num_threads,
        track_stats=False,
        seed=args.seed,
    )
    tag = f"gepa-onpolicy-{args.task}-seed{args.seed}"
    with cost_tag(tag):
        optimized = gepa.compile(baseline, trainset=trainset, valset=rest[:20])
        # Reflection calls bypass the pipelines' own flush; the local task model
        # costs nothing but is flushed too so its call count is on record.
        flush_lm_history(reflection_lm, tag=f"{tag}-reflection")
        flush_lm_history(task_lm, tag=f"{tag}-task-local")
    wall = time.monotonic() - t0

    instructions = {m: getattr(optimized, m).predict.signature.instructions for m in modules}
    payload = {
        "task": args.task,
        "seed": args.seed,
        "task_model": args.model,
        "task_model_served_at": args.api_base,
        "reflection_model": REFLECTION_MODEL,
        "max_metric_calls": args.max_metric_calls,
        "n_train": 15,
        "n_val": 20,
        "wall_seconds": round(wall, 1),
        "api_cost_usd_this_process": round(METER.total_usd, 4),
        "instruction_chars": {m: len(v) for m, v in instructions.items()},
        "instructions": instructions,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.task}_seed{args.seed}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  wall={wall:.0f}s  chars={payload['instruction_chars']}  api_cost=${METER.total_usd:.3f}")


if __name__ == "__main__":
    main()
