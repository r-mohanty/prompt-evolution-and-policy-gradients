"""
Multi-module GRPO training: trains the FULL compound pipeline (every LM
module, not a single-shot simplification) via group-relative policy
optimization, with Dr. GRPO's bias fixes and RCA's per-module credit-based
advantage redistribution unified into one script, parameterized by
--variant.

Dr. GRPO's two fixes (a constant normaliser in place of per-sequence length
normalisation, and no division of advantages by the group standard deviation)
are implemented by hand in compute_dr_grpo_loss, because this script does not
use TRL's GRPOTrainer (see "Why not TRL's GRPOTrainer" below).

--variant:
  dr_grpo        : credit_uniform_mask always. This IS Track A's plain
                   Dr. GRPO baseline (Step 1.5/2.1-2.3), now on the real
                   multi-module pipeline for a fair comparison against GEPA
                   (which optimizes that same pipeline's prompts).
  rca            : reflect_credit + validate_credit_training each group
                   (Step 2.7). Wired 2026-08-17 -- see that function's
                   docstring for how training-time validation differs from
                   FaultLine's evaluation-time validate_credit in rca.py.
  rca_no_validate: reflect_credit, skip validation, use raw credit directly.
  gepa_then_grpo : run GEPA to convergence first (run_gepa_for_instructions,
                   reuses scripts/prompt_opt/gepa_repro.py's set-up) for optimized
                   per-module instructions, THEN GRPO-trains weights
                   starting from those instructions with credit_uniform_mask
                   -- a different initialization, not a different loss
                   (operationalizes Soylu et al., per the plan's Step
                   1.4/2.7 notes). Wired 2026-08-17.

Every non-obvious design decision is documented where it is made; the
implementation notes at the bottom of this file record the failure modes
found and fixed on real hardware.

Why not TRL's GRPOTrainer: TRL's built-in rollout generation is prompt ->
one completion. It has no support for "generate module A, run a
deterministic tool call, generate module B conditioned on both, treat the
whole thing as one trajectory for a single group-relative loss." That
rollout shape is exactly what multi-module credit assignment needs, so
this script implements the GRPO loss by hand (raw transformers + peft,
Dr. GRPO's two fixes applied directly in compute_dr_grpo_loss) instead of
configuring TRL. We lose TRL's built-in robustness/optimizations in
exchange for the rollout shape we actually need.

Run on a GPU host (one 80 GB H100 per job), from the repository root:
    python -m reflectrl.training.multi_module_grpo_train --task hotpotqa --budget 128 --seed 0 --variant dr_grpo
    python -m reflectrl.training.multi_module_grpo_train --task aime --budget 128 --seed 0 --variant rca
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(WORKSHOP_ROOT / ".env")

POLICY_MODEL = "Qwen/Qwen3-8B"
LORA_CONFIG = dict(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
GROUP_SIZE = 8  # GRPO's "k" rollouts per prompt
MAX_NEW_TOKENS_PER_MODULE = 512
LEARNING_RATE = 1e-6
GRAD_ACCUM_STEPS = 2
KL_COEF = 0.04  # standard GRPO/DeepSeekMath default; one of step 1.6's tuning dimensions (added 2026-08-17, see compute_dr_grpo_loss)
VLLM_GPU_MEMORY_UTILIZATION = 0.4  # leaves ~48GB of one 80GB H100 for the training model + KL reference pass (measured ~17-20GB); see collect_trajectories_vllm's docstring

VARIANTS = ["dr_grpo", "rca", "rca_no_validate", "gepa_then_grpo", "gepa_onpolicy_then_grpo", "docstring_then_grpo"]
# docstring_then_grpo (added 2026-09-03): trains from the DSPy signature docstrings of
# reflectrl/pipelines/{hotpotqa,aime}_pipeline.py, UNCHANGED, i.e. GEPA's seed program with no
# evolution at all. Motivation: inspection of the archived gepa_instructions.json files
# showed that GEPA returned its seed program unchanged in all 12 math cells and in
# several HotpotQA modules, so the "GEPA arm" partly measures the wording difference
# between those docstrings and DEFAULT_INSTRUCTIONS below. This arm isolates that
# wording effect from GEPA's edits.
# gepa_onpolicy_then_grpo (added 2026-09-03): identical to gepa_then_grpo in every
# respect EXCEPT that the instructions were evolved by GEPA with Qwen3-8B ITSELF as
# the task model (served locally by vLLM; see scripts/prompt_opt/gepa_onpolicy_instructions.py),
# rather than with Claude Haiku 4.5 and then transplanted. It reads the pre-computed
# instructions from data/gepa_onpolicy/instructions/{task}_seed{seed}.json instead of
# running GEPA inside the training job. Purpose: close the prompt-model confound the
# paper's Section 6 names for the regime-boundary result.

# Single source of truth for default per-module instruction text -- used by
# both normal rollout collection AND gepa_then_grpo's instruction-override
# path (which replaces these with GEPA-optimized text), so there is exactly
# one place that defines "what does this module's prompt say" rather than
# two copies that could drift.
DEFAULT_INSTRUCTIONS = {
    "hotpotqa": {
        "generate_query": (
            "Answer the question using only the context that will be provided. "
            "First, write a short search query to find the needed evidence."
        ),
        "generate_answer": "Now answer the question with the shortest correct span of text, no explanation.",
    },
    "aime": {
        "plan": "Solve this competition math problem. First write a short (2-5 sentence) solution strategy, no final answer yet.",
        "solve": "Now work the full solution, ending with \\boxed{answer}.",
        "verify": "Check the arithmetic independently. Restate or correct the boxed answer.",
    },
}


@dataclass
class Trajectory:
    """One full rollout through every module of the pipeline."""
    full_text: str
    full_token_ids: list[int]  # the ENTIRE sequence: initial prompt + every module's generated tokens + any inserted tool-output text
    module_token_spans: dict[str, tuple[int, int]]  # module_name -> (start, end) marking GENERATED tokens only (never tool-output/prompt tokens)
    reward: float
    module_outputs: dict[str, str]  # module_name -> its raw generated text, for reflect_credit()
    question: str = ""
    gold_answer: str = ""


# ---------------------------------------------------------------------------
# Task-specific rollout collection. Each builds a Trajectory by generating
# every module in sequence as one growing multi-turn CHAT conversation (one
# user/assistant turn pair per module) -- this is what makes the later
# teacher-forced re-scoring pass mathematically equivalent to what
# autoregressive generation actually saw, while giving an instruction-tuned
# model (Qwen3-8B) the turn-boundary/stop tokens it needs to know when to
# stop generating. See module docstring.
#
# CHANGED 2026-08-17: originally built prompts via raw string concatenation
# (no chat template). Confirmed on the real Phase 1 smoke test to cause
# near-zero rewards (0.003-0.008 across 32 rollouts) -- inspecting one real
# trajectory showed WHY: the query module hallucinated a fake
# "Context:...Answer:" continuation instead of stopping after the query,
# and the answer module degenerated into repeating the same sentence ~60
# times until hitting max_new_tokens. Classic symptom of prompting an
# instruction-tuned model without its expected chat format -- no learned
# stop signal. Rebuilt on tokenizer.apply_chat_template with an
# incrementally-growing `conversation` list (one user/assistant pair per
# module) instead of raw text concatenation -- see _chat_generate_turn/
# _compute_chat_spans below, shared by both task functions to avoid
# duplicating the span-computation logic twice.
#
# resume_from / fixed_outputs (added 2026-08-17): used by
# regenerate_from_module() for RCA's training-time counterfactual
# validation. Modules BEFORE resume_from reuse fixed_outputs' text instead
# of sampling; resume_from and everything after is freshly sampled. Default
# behavior (resume_from = the first module) is unchanged: sample everything
# fresh, the normal rollout-collection case.
#
# instructions (added 2026-08-17): per-module instruction-text overrides,
# used by the gepa_then_grpo variant to swap in GEPA-optimized instructions
# instead of DEFAULT_INSTRUCTIONS.
# ---------------------------------------------------------------------------

def _chat_generate_turn(model, tokenizer, conversation: list[dict], user_content: str, device, max_new_tokens: int, generate_fresh: bool, fixed_text: str | None) -> str:
    """Appends one user turn + one assistant turn to `conversation` IN
    PLACE. If generate_fresh, samples the assistant turn from the model
    (conditioned on the full conversation so far, via apply_chat_template's
    add_generation_prompt=True); otherwise reuses fixed_text unchanged
    (RCA validation's resume_from mechanism). Returns the turn's text.

    enable_thinking=False (added 2026-08-17, caught on the real single-
    trajectory diagnostic after the chat-template fix): Qwen3 defaults to
    extended <think>...</think> reasoning before every response. With this
    trainer's small per-module budgets (64 tokens for a search query, for
    example) thinking alone can consume the ENTIRE budget, leaving zero
    tokens for the module's actual output -- confirmed directly (the query
    module generated 64 tokens of mid-sentence reasoning and no query at
    all). Each module here is meant to do one small, focused thing
    (mirroring the DSPy pipeline's own dspy.ChainOfThought design, where
    reasoning is a separate bounded field, not an open-ended budget-eating
    prefix), so disabling thinking mode is the correct fix for this task,
    not a token-budget increase (which would both blow up compute cost and
    still not guarantee the model finishes within any fixed budget)."""
    conversation.append({"role": "user", "content": user_content})
    if generate_fresh:
        prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        out = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
        text = fixed_text
    conversation.append({"role": "assistant", "content": text})
    return text


def _compute_chat_spans(tokenizer, conversation: list[dict], module_order: list[str]) -> tuple[str, list[int], dict[str, tuple[int, int]]]:
    """conversation is [user_0, assistant_0, user_1, assistant_1, ...], one
    user/assistant pair per module in module_order. Returns the full
    conversation's canonical single tokenization plus each module's
    (start, end) span within it.

    FIXED 2026-08-17 (v3 -- v2's "render the prefix with
    add_generation_prompt=True and take its length" approach was ALSO
    caught broken by the same GPU-free tokenizer test, before it ever
    reached a real GPU run): Qwen3's chat template inserts an empty
    "<think>\n\n</think>\n\n" scaffold before an assistant message's content
    ONLY when that message is the LAST one in the list being rendered --
    not for every assistant turn. That makes rendering "the same message,
    truncated at different points" context-dependent: conversation[:k] (this
    message is last -> gets the scaffold) renders differently than the same
    message's position within the full conversation (not last -> no
    scaffold, for every module except the final one). So NEITHER "render up
    to but excluding this message with add_generation_prompt=True" (v2) NOR
    "render up to and including this message" reliably reproduces this
    message's actual rendering within the full, untruncated conversation --
    there is no prefix-consistent way to predict boundaries via truncated
    re-renders when the template's output depends on global position, not
    just prefix content.
    The robust fix: don't try to predict boundaries via separate renders at
    all. Search for each module's known CONTENT STRING directly within the
    one authoritative full_text, in conversation order (searching only
    after the previous module's match, so turns can't cross-match each
    other) -- this only depends on the content actually being present
    verbatim in full_text, which apply_chat_template guarantees (it inserts
    message.content as-is), not on reproducing the template's own
    positional logic. Character positions are then mapped onto the single
    full tokenization's token offsets (return_offsets_mapping=True)."""
    full_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    spans = {}
    search_from = 0
    for i, mod in enumerate(module_order):
        assistant_idx = 2 * i + 1  # 0=user_0, 1=assistant_0, 2=user_1, 3=assistant_1, ...
        content = conversation[assistant_idx]["content"]
        char_start = full_text.index(content, search_from)
        char_end = char_start + len(content)
        search_from = char_end
        tok_start = next(idx for idx, (s, e) in enumerate(offsets) if s >= char_start)
        tok_end = next((idx for idx, (s, e) in enumerate(offsets) if s >= char_end), len(offsets))
        spans[mod] = (tok_start, tok_end)
    return full_text, full_token_ids, spans


# ---------------------------------------------------------------------------
# Answer-span extraction, applied UNIFORMLY to every variant (added 2026-09-01).
#
# WHY: GEPA's optimized instructions are produced inside DSPy, where a
# structured `answer: <span>` output contract is the norm. When those
# instructions are transplanted into this raw-text harness, some draws emit a
# reasoning preamble followed by `answer: X`, and scoring the WHOLE string
# against the gold span drives F1 to ~0. The 2026-08-19 diagnostic measured
# exactly this: reward 0.034 as-is vs 0.500 with the span extracted.
#
# This is NOT a variant-conditional hack. It is one rule -- "score the text
# after an `answer:` marker if present, else score the whole output" -- applied
# identically on every code path. Measured behaviour (2026-09-01 diagnostic,
# 8 examples each):
#   * DEFAULT instructions      : 0/8 markers -> exact no-op (0.0884 both ways)
#   * a clean GEPA draw         : 0/8 markers -> exact no-op (0.5500 both ways)
#   * a marker-emitting draw    : rescued 0.034 -> 0.500
# AIME needs no equivalent because collect_aime_trajectory already extracts via
# extract_boxed_answer(); this gives HotpotQA the same protection.
#
# CAVEAT for the writeup: the no-op property is verified on 8 DEFAULT samples,
# not proven for all outputs. A DEFAULT generation that happened to contain the
# literal token "answer:" would be scored on its tail instead of in full.
# ---------------------------------------------------------------------------

_ANSWER_SPAN_RE = re.compile(r"(?is)\banswer\b\s*[\"']?\s*[:=]\s*(.+)")


def _extract_answer_span(text: str) -> str:
    """Return the span after an `answer:` marker, else the text unchanged."""
    match = _ANSWER_SPAN_RE.search(text or "")
    return match.group(1).strip() if match else (text or "")


def collect_hotpotqa_trajectory(
    model, tokenizer, example, device,
    instructions: dict[str, str] | None = None,
    resume_from: str = "generate_query",
    fixed_outputs: dict[str, str] | None = None,
) -> Trajectory:
    from reflectrl.pipelines.hotpotqa_pipeline import BM25DistractorRetriever, exact_match, f1_score

    ins = {**DEFAULT_INSTRUCTIONS["hotpotqa"], **(instructions or {})}
    fixed_outputs = fixed_outputs or {}
    module_order = ["generate_query", "generate_answer"]
    resume_idx = module_order.index(resume_from)

    conversation: list[dict] = []
    query_text = _chat_generate_turn(
        model, tokenizer, conversation,
        user_content=f"{ins['generate_query']}\n\nQuestion: {example.question}",
        device=device, max_new_tokens=64,
        generate_fresh=(resume_idx <= 0), fixed_text=fixed_outputs.get("generate_query"),
    )

    retriever = BM25DistractorRetriever(example.context["title"], example.context["sentences"])
    passages = retriever.retrieve(query_text or example.question, k=5)
    context_block = "\n\n".join(passages)

    answer_text = _chat_generate_turn(
        model, tokenizer, conversation,
        user_content=f"Context:\n{context_block}\n\n{ins['generate_answer']}",
        device=device, max_new_tokens=MAX_NEW_TOKENS_PER_MODULE,
        generate_fresh=(resume_idx <= 1), fixed_text=fixed_outputs.get("generate_answer"),
    )

    scored_answer = _extract_answer_span(answer_text)
    reward = (exact_match(scored_answer, example.answer) + f1_score(scored_answer, example.answer)) / 2.0
    full_text, full_token_ids, spans = _compute_chat_spans(tokenizer, conversation, module_order)

    return Trajectory(
        full_text=full_text,
        full_token_ids=full_token_ids,
        module_token_spans=spans,
        reward=reward,
        module_outputs={"generate_query": query_text, "generate_answer": answer_text},
    )


def collect_aime_trajectory(
    model, tokenizer, example, device,
    instructions: dict[str, str] | None = None,
    resume_from: str = "plan",
    fixed_outputs: dict[str, str] | None = None,
) -> Trajectory:
    from reflectrl.pipelines.aime_pipeline import answer_check_metric, extract_boxed_answer

    ins = {**DEFAULT_INSTRUCTIONS["aime"], **(instructions or {})}
    fixed_outputs = fixed_outputs or {}
    module_order = ["plan", "solve", "verify"]
    resume_idx = module_order.index(resume_from)

    conversation: list[dict] = []
    plan_text = _chat_generate_turn(
        model, tokenizer, conversation,
        user_content=f"{ins['plan']}\n\nProblem: {example.question}",
        device=device, max_new_tokens=128,
        generate_fresh=(resume_idx <= 0), fixed_text=fixed_outputs.get("plan"),
    )
    solve_text = _chat_generate_turn(
        model, tokenizer, conversation,
        user_content=ins["solve"],
        device=device, max_new_tokens=MAX_NEW_TOKENS_PER_MODULE,
        generate_fresh=(resume_idx <= 1), fixed_text=fixed_outputs.get("solve"),
    )
    verify_text = _chat_generate_turn(
        model, tokenizer, conversation,
        user_content=ins["verify"],
        device=device, max_new_tokens=256,
        generate_fresh=(resume_idx <= 2), fixed_text=fixed_outputs.get("verify"),
    )

    predicted = extract_boxed_answer(verify_text) or extract_boxed_answer(solve_text) or ""
    reward = answer_check_metric(
        type("G", (), {"answer": example.answer})(), type("P", (), {"answer": predicted})(),
    )
    full_text, full_token_ids, spans = _compute_chat_spans(tokenizer, conversation, module_order)

    return Trajectory(
        full_text=full_text,
        full_token_ids=full_token_ids,
        module_token_spans=spans,
        reward=reward,
        module_outputs={"plan": plan_text, "solve": solve_text, "verify": verify_text},
    )


def collect_trajectory(task: str, model, tokenizer, example, device, **kwargs) -> Trajectory:
    fn = collect_hotpotqa_trajectory if task == "hotpotqa" else collect_aime_trajectory
    return fn(model, tokenizer, example, device, **kwargs)


# ---------------------------------------------------------------------------
# vLLM-accelerated GROUP-batched rollout collection (added 2026-08-17).
#
# Why: the per-rollout collect_*_trajectory functions above call
# model.generate() once per rollout, sequentially -- confirmed on real
# hardware to run at only ~6-8 tokens/sec with ~32-34% GPU utilization
# (batch_size=1 badly underuses an H100's parallel compute), and EVERY
# module call hit its max_new_tokens ceiling in a real diagnostic (the
# model rarely emits EOS within these budgets). At GROUP_SIZE=8, one
# training step needed 48+ minutes on AIME and still hadn't finished when
# investigated -- a real, measured problem, not a hypothetical one.
#
# Fix: vLLM for the GROUP's generation, kept colocated with the actively-
# training PyTorch/PEFT model on the same GPU (gpu_memory_utilization=0.4
# leaves ~48GB headroom for the training model + KL reference pass, which
# measured ~17-20GB in earlier runs -- verified to fit on one 80GB H100).
# Standalone-verified 2026-08-17: ~5x throughput (33.5 tok/s aggregate for
# a batched n=8 call vs. ~7 tok/s sequential), LoRA sync via
# model.save_pretrained() + a fresh LoRARequest works cleanly.
#
# REQUIRES a one-time manual patch on any fresh pod before this will work:
# flashinfer (a vLLM dependency) has a real upstream bug -- `array.array[int]`
# as a type annotation crashes on import under Python 3.11 (evaluated
# eagerly; needs `from __future__ import annotations`, which the file is
# missing). See the README's GPU environment section for the patch. Without it,
# vLLM's engine core process crashes with "TypeError: type 'array.array' is
# not subscriptable" during startup, even with enforce_eager=True (which
# does NOT avoid this particular import).
#
# Scope of this fix: only the MAIN rollout-collection path in run()'s
# training loop (the actual bottleneck, GROUP_SIZE calls every step). RCA's
# regenerate_from_module() (validate_credit_training's counterfactual
# check) deliberately still uses the OLD single-rollout HF-based
# collect_trajectory() above, unchanged -- it's infrequent (once per group
# for the rca variant only, not GROUP_SIZE times), so the speed benefit
# there is much smaller, and reusing already-verified code reduces risk
# rather than doubling the surface area of this rewrite.
#
# Batching strategy: module 1 shares an IDENTICAL prompt across the whole
# group (same example, no sampling has diverged the rollouts yet) --
# generated via ONE vLLM call with SamplingParams(n=group_size). Every
# module after that has DIVERGED per rollout (different retrieved context /
# different prior module output per rollout) -- generated via ONE vLLM call
# per module with a LIST of group_size different prompts
# (SamplingParams(n=1) each). Either way, one vLLM call replaces
# group_size sequential HF .generate() calls.
# ---------------------------------------------------------------------------

def _vllm_group_generate(llm, lora_request, prompt: str, group_size: int, max_tokens: int) -> list[str]:
    """ONE shared prompt (module 1, before any rollout has diverged from
    the others) -- group_size independent samples in a single vLLM call."""
    from vllm import SamplingParams

    sp = SamplingParams(n=group_size, temperature=1.0, max_tokens=max_tokens)
    outputs = llm.generate([prompt], sp, lora_request=lora_request, use_tqdm=False)
    return [o.text.strip() for o in outputs[0].outputs]


def _vllm_batch_generate(llm, lora_request, prompts: list[str], max_tokens: int) -> list[str]:
    """A DIFFERENT prompt per rollout (any module after the first, once
    rollouts have diverged) -- one completion per prompt, batched in a
    single vLLM call."""
    from vllm import SamplingParams

    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sp, lora_request=lora_request, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]


def collect_hotpotqa_trajectories_vllm(llm, lora_request, tokenizer, example, group_size: int, instructions: dict[str, str] | None = None) -> list[Trajectory]:
    from reflectrl.pipelines.hotpotqa_pipeline import BM25DistractorRetriever, exact_match, f1_score

    ins = {**DEFAULT_INSTRUCTIONS["hotpotqa"], **(instructions or {})}
    module_order = ["generate_query", "generate_answer"]

    query_user_content = f"{ins['generate_query']}\n\nQuestion: {example.question}"
    query_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": query_user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    query_texts = _vllm_group_generate(llm, lora_request, query_prompt, group_size, max_tokens=64)

    conversations = []
    answer_prompts = []
    for query_text in query_texts:
        conversation = [
            {"role": "user", "content": query_user_content},
            {"role": "assistant", "content": query_text},
        ]
        retriever = BM25DistractorRetriever(example.context["title"], example.context["sentences"])
        passages = retriever.retrieve(query_text or example.question, k=5)
        context_block = "\n\n".join(passages)
        conversation.append({"role": "user", "content": f"Context:\n{context_block}\n\n{ins['generate_answer']}"})
        answer_prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        conversations.append(conversation)

    answer_texts = _vllm_batch_generate(llm, lora_request, answer_prompts, max_tokens=MAX_NEW_TOKENS_PER_MODULE)

    trajectories = []
    for conversation, answer_text in zip(conversations, answer_texts):
        conversation.append({"role": "assistant", "content": answer_text})
        query_text = conversation[1]["content"]
        scored_answer = _extract_answer_span(answer_text)
        reward = (exact_match(scored_answer, example.answer) + f1_score(scored_answer, example.answer)) / 2.0
        full_text, full_token_ids, spans = _compute_chat_spans(tokenizer, conversation, module_order)
        trajectories.append(Trajectory(
            full_text=full_text, full_token_ids=full_token_ids, module_token_spans=spans, reward=reward,
            module_outputs={"generate_query": query_text, "generate_answer": answer_text},
        ))
    return trajectories


def collect_aime_trajectories_vllm(llm, lora_request, tokenizer, example, group_size: int, instructions: dict[str, str] | None = None) -> list[Trajectory]:
    from reflectrl.pipelines.aime_pipeline import answer_check_metric, extract_boxed_answer

    ins = {**DEFAULT_INSTRUCTIONS["aime"], **(instructions or {})}
    module_order = ["plan", "solve", "verify"]

    plan_user_content = f"{ins['plan']}\n\nProblem: {example.question}"
    plan_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": plan_user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    plan_texts = _vllm_group_generate(llm, lora_request, plan_prompt, group_size, max_tokens=128)

    conversations = []
    solve_prompts = []
    for plan_text in plan_texts:
        conversation = [
            {"role": "user", "content": plan_user_content},
            {"role": "assistant", "content": plan_text},
            {"role": "user", "content": ins["solve"]},
        ]
        solve_prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        conversations.append(conversation)
    solve_texts = _vllm_batch_generate(llm, lora_request, solve_prompts, max_tokens=MAX_NEW_TOKENS_PER_MODULE)

    verify_prompts = []
    for conversation, solve_text in zip(conversations, solve_texts):
        conversation.append({"role": "assistant", "content": solve_text})
        conversation.append({"role": "user", "content": ins["verify"]})
        verify_prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    verify_texts = _vllm_batch_generate(llm, lora_request, verify_prompts, max_tokens=256)

    trajectories = []
    for conversation, verify_text in zip(conversations, verify_texts):
        conversation.append({"role": "assistant", "content": verify_text})
        plan_text = conversation[1]["content"]
        solve_text = conversation[3]["content"]
        predicted = extract_boxed_answer(verify_text) or extract_boxed_answer(solve_text) or ""
        reward = answer_check_metric(
            type("G", (), {"answer": example.answer})(), type("P", (), {"answer": predicted})(),
        )
        full_text, full_token_ids, spans = _compute_chat_spans(tokenizer, conversation, module_order)
        trajectories.append(Trajectory(
            full_text=full_text, full_token_ids=full_token_ids, module_token_spans=spans, reward=reward,
            module_outputs={"plan": plan_text, "solve": solve_text, "verify": verify_text},
        ))
    return trajectories


def collect_trajectories_vllm(task: str, llm, lora_request, tokenizer, example, group_size: int, instructions: dict[str, str] | None = None) -> list[Trajectory]:
    fn = collect_hotpotqa_trajectories_vllm if task == "hotpotqa" else collect_aime_trajectories_vllm
    return fn(llm, lora_request, tokenizer, example, group_size, instructions=instructions)


# ---------------------------------------------------------------------------
# RCA's training-time counterfactual validation (wired in 2026-08-17).
# ---------------------------------------------------------------------------

def regenerate_from_module(
    task: str, model, tokenizer, example, trajectory: Trajectory, resume_module: str, device,
    instructions: dict[str, str] | None = None,
) -> float:
    """Returns just the REWARD of a fresh rollout that reuses trajectory's
    module outputs BEFORE resume_module and resamples resume_module and
    everything after. No gradient (torch.no_grad()), no need for correct
    span bookkeeping -- used only by validate_credit_training(), which
    reads just the resulting reward."""
    import torch

    with torch.no_grad():
        regenerated = collect_trajectory(
            task, model, tokenizer, example, device,
            instructions=instructions, resume_from=resume_module, fixed_outputs=trajectory.module_outputs,
        )
    return regenerated.reward


def validate_credit_training(
    variant_credit: dict[str, float], trajectory: Trajectory, task: str, example,
    model, tokenizer, device, instructions: dict[str, str] | None = None,
) -> tuple[dict[str, float], bool]:
    """Training-time counterfactual validation -- Step 2.7's 'uniform-mask
    fallback when validation fails', operationalized differently from
    FaultLine's evaluation-time validate_credit() in rca.py. FaultLine's
    version patches a module to a KNOWN-CORRECT baseline pipeline's output
    and checks if that restores success -- there's no such baseline during
    training (the policy IS what's being trained). Here, validation means:
    regenerate the TOP-credited module's output as a FRESH SAMPLE (not a
    known-correct reference) and everything downstream of it; if reward
    improves over what was actually rolled out, the credit assignment is
    corroborated (this module really was where the trajectory's headroom
    was, since a different sample from the CURRENT policy did better). If
    not, fall back to uniform. This is a different operationalization of
    the same principle, not the same mechanism -- flag this distinction
    explicitly if/when this shows up in the paper's methods section."""
    modules = list(variant_credit.keys())
    top_module = max(variant_credit, key=variant_credit.get)
    regenerated_reward = regenerate_from_module(task, model, tokenizer, example, trajectory, top_module, device, instructions)
    validated = regenerated_reward > trajectory.reward
    if validated:
        return variant_credit, True
    uniform = {m: 1.0 / len(modules) for m in modules}
    return uniform, False


# ---------------------------------------------------------------------------
# gepa_then_grpo's instruction-feeding (wired in 2026-08-17).
# ---------------------------------------------------------------------------

def run_gepa_for_instructions(task: str, max_metric_calls: int = 100) -> dict[str, str]:
    """Runs GEPA to convergence on the task's pipeline (reuses the same
    DSPy/GEPA machinery as gepa_repro.py, at a modest scale) and extracts
    the resulting per-module instructions. ONE-TIME cost per (task, seed)
    for the gepa_then_grpo variant, not per-rollout. Pure API cost (Haiku
    task model + Sonnet-5 reflection model, same choice as gepa_repro.py),
    no GPU needed -- independently testable right now. Uses
    metering.cost_tag/flush_lm_history exactly like gepa_repro.py does, so
    this call's spend lands in logs/cost_log.jsonl under a dedicated
    tag instead of silently going untracked."""
    import dspy

    from reflectrl.pipelines.metering import cost_tag, flush_lm_history

    reflection_lm = dspy.LM("anthropic/claude-sonnet-5", max_tokens=4096, api_key=os.environ["ANTHROPIC_API_KEY"])

    if task == "hotpotqa":
        from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric, load_devset

        dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5", max_tokens=1024, api_key=os.environ["ANTHROPIC_API_KEY"]))
        trainset, rest = load_devset(n_train=15, n_dev=20)
        baseline = HotpotQAPipeline(k=5)
        metric = lambda gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None: em_f1_metric(gold, pred)
        gepa = dspy.GEPA(
            metric=metric, reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls, reflection_minibatch_size=3, track_stats=False,
        )
        with cost_tag(f"mmgrpo-gepa-instructions-{task}"):
            optimized = gepa.compile(baseline, trainset=trainset, valset=rest[:20])
            # reflection_lm calls bypass HotpotQAPipeline.forward() (GEPA
            # calls it directly), so they need an explicit flush -- see
            # gepa_repro.py's identical note.
            flush_lm_history(reflection_lm, tag="mmgrpo-gepa-reflection")
        return {
            "generate_query": optimized.generate_query.predict.signature.instructions,
            "generate_answer": optimized.generate_answer.predict.signature.instructions,
        }
    else:
        from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric, load_devset

        dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5", max_tokens=2048, api_key=os.environ["ANTHROPIC_API_KEY"]))
        trainset, rest = load_devset(n_train=15, n_dev=20)
        baseline = AIMEStylePipeline()
        metric = lambda gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None: answer_check_metric(gold, pred)
        gepa = dspy.GEPA(
            metric=metric, reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls, reflection_minibatch_size=3, track_stats=False,
        )
        with cost_tag(f"mmgrpo-gepa-instructions-{task}"):
            optimized = gepa.compile(baseline, trainset=trainset, valset=rest[:20])
            flush_lm_history(reflection_lm, tag="mmgrpo-gepa-reflection")
        return {
            "plan": optimized.plan.predict.signature.instructions,
            "solve": optimized.solve.predict.signature.instructions,
            "verify": optimized.verify.predict.signature.instructions,
        }


# ---------------------------------------------------------------------------
# Credit assignment per variant. All four share the same call signature so
# the training loop doesn't need to know which is active.
# ---------------------------------------------------------------------------

def get_credit(
    variant: str, trajectory: Trajectory, modules: list[str], reflection_client,
    task: str, example, model, tokenizer, device, instructions: dict[str, str] | None = None,
) -> dict[str, float]:
    from reflectrl.training.rca import credit_uniform_mask, reflect_credit

    if variant in ("dr_grpo", "gepa_then_grpo", "gepa_onpolicy_then_grpo", "docstring_then_grpo"):
        return credit_uniform_mask(modules)

    # rca / rca_no_validate: reflect once on this trajectory. Cost note --
    # DESIGN DECISION (not found in the plan, made here): reflecting on
    # EVERY individual rollout would mean one extra Sonnet-5 API call per
    # rollout -- at budget=8192 that's 8192 extra API calls for one training
    # run, which is expensive and slow independent of the GPU cost. Instead,
    # reflection is called ONCE PER GROUP (not per rollout), on the
    # lowest-reward rollout in that group of GROUP_SIZE, and the resulting
    # credit distribution is applied to every rollout in the group. This
    # assumes rollouts from the same prompt share a similar
    # module-fragility profile, which is a reasonable but unverified
    # assumption -- flag this explicitly in the paper's methods section if
    # this design is kept past the first real run.
    trajectory_text = "\n".join(f"[{m}]: {t}" for m, t in trajectory.module_outputs.items())
    reflection = reflect_credit(
        question=trajectory.question,
        gold_answer=trajectory.gold_answer,
        predicted_answer=list(trajectory.module_outputs.values())[-1],
        trajectory_text=trajectory_text,
        modules=modules,
        client=reflection_client,
    )
    if variant == "rca_no_validate":
        return reflection.credit
    # variant == "rca": wired 2026-08-17 to real training-time
    # counterfactual validation -- see validate_credit_training's docstring
    # for how this differs from FaultLine's evaluation-time version.
    validated_credit, _was_validated = validate_credit_training(
        reflection.credit, trajectory, task, example, model, tokenizer, device, instructions,
    )
    return validated_credit


# ---------------------------------------------------------------------------
# The GRPO loss, Dr. GRPO's two fixes applied by hand (constant normaliser,
# no group-std scaling; see the module docstring), combined with RCA's
# per-token advantage mask.
# ---------------------------------------------------------------------------

def compute_dr_grpo_loss(model, tokenizer, trajectories: list[Trajectory], credits: list[dict[str, float]], device, grad_scale: float = 1.0, kl_coef: float = KL_COEF):
    """Computes the Dr. GRPO loss (policy-gradient term + a KL penalty
    against the frozen reference policy) AND backprops it immediately, one
    trajectory at a time, instead of summing all GROUP_SIZE trajectories'
    losses into one graph and calling .backward() once at the end. FIXED
    2026-08-17 after a real CUDA OOM on the Phase 1 smoke test (single
    80GB H100, GROUP_SIZE=8): the original held-graph version needed all 8
    trajectories' forward-pass activations resident simultaneously (~8x
    the necessary peak memory); "79.14 GiB memory in use" of 79.18 GiB
    capacity at the moment of failure. Per-trajectory backward is
    mathematically identical (gradients are linear/additive -- calling
    .backward() 8 times and letting .grad accumulate equals summing the 8
    losses and calling .backward() once) but only needs ONE trajectory's
    graph alive at a time, freed immediately after each backward() call.
    `grad_scale` folds in GRAD_ACCUM_STEPS' scaling (the caller no longer
    calls .backward() itself -- see run()'s call site). Returns the total
    loss as a plain float for logging, not a tensor -- callers must NOT
    call .backward() on it again.

    KL term added 2026-08-17: the set-up plan always called step 1.6 an
    "LR/KL/group-size" tuning sweep, but no KL penalty against a
    reference policy actually existed until this fix -- caught before
    running that sweep, not after. Standard GRPO/Dr. GRPO formulation
    includes this as regularization against the frozen initial policy;
    dropping it silently would have both violated what the setup doc
    already promised and left one of step 1.6's three named tuning
    dimensions nonexistent. Reference logprobs come from the SAME model
    with its LoRA adapter disabled (PEFT's model.disable_adapter() context
    manager) -- since the base weights are frozen, disabling the adapter
    gives exactly the reference (pre-training) policy's logits with no
    second model copy in memory, at the cost of one extra no_grad forward
    pass per trajectory (roughly doubling per-step GPU time -- a real,
    accepted cost, not an oversight; see decisions.md for the tradeoff
    discussion). Uses the k3 estimator (Schulman, http://joschu.net/blog/kl-approx.html
    -- the same low-variance, always-non-negative estimator used in
    DeepSeekMath/TRL's own GRPO implementations): for log_ratio =
    policy_logprob - ref_logprob on tokens actually sampled from the
    policy, KL ≈ exp(-log_ratio) - 1 + log_ratio. Applied only over
    GENERATED (module) tokens, never prompt/context tokens -- built from
    the same shifted spans as the advantage mask, but as a plain 0/1
    indicator (NOT reused from advantage_mask's nonzero entries, since a
    trajectory whose reward exactly equals the group mean has advantage=0
    on every one of its own generated tokens, which would make
    (advantage_mask != 0) wrongly zero out its entire KL contribution)."""
    import torch

    from reflectrl.training.rca import compute_advantage_mask

    rewards = [t.reward for t in trajectories]
    group_mean = sum(rewards) / len(rewards)
    # Dr. GRPO fix #2: advantage = reward - group_mean, NO division by
    # group std (vanilla GRPO's std-normalization inflates advantages for
    # low-variance/already-solved groups -- the bias this paper's own
    # cost-honest accounting can't afford to have contaminating it).
    advantages = [r - group_mean for r in rewards]

    # Dr. GRPO fix #1: normalize by a CONSTANT (max tokens across the
    # group's generated spans), not each trajectory's own length -- computed
    # up front (no model calls needed) since it must be known before any
    # trajectory's per-token loss can be normalized.
    total_generated_tokens = max(
        sum(e - s for s, e in traj.module_token_spans.values()) for traj in trajectories
    )

    total_loss_value = 0.0
    for traj, credit, advantage in zip(trajectories, credits, advantages):
        ids = torch.tensor([traj.full_token_ids], device=device)
        outputs = model(ids, labels=None, use_cache=False)  # use_cache=False: required alongside gradient checkpointing for a gradient-mode forward pass
        logits = outputs.logits[0, :-1, :]  # predict token t+1 from position t
        target_ids = ids[0, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (seq_len-1,)

        with torch.no_grad(), model.disable_adapter():
            ref_outputs = model(ids, labels=None, use_cache=False)
            ref_logits = ref_outputs.logits[0, :-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # module_token_spans are indices into the ORIGINAL (not shifted-by-1)
        # sequence; token_log_probs[i] is the log-prob of predicting
        # original-sequence position i+1. Shift spans by -1 accordingly.
        shifted_spans = {m: (s - 1, e - 1) for m, (s, e) in traj.module_token_spans.items()}
        mask_np = compute_advantage_mask(
            module_token_spans=shifted_spans,
            credit=credit,
            total_length=len(traj.full_token_ids) - 1,
            scalar_advantage=advantage,
        )
        advantage_mask = torch.tensor(mask_np, device=device, dtype=token_log_probs.dtype)

        generated_mask = torch.zeros(len(traj.full_token_ids) - 1, device=device, dtype=token_log_probs.dtype)
        for s, e in shifted_spans.values():
            generated_mask[s:e] = 1.0

        log_ratio = token_log_probs - ref_token_log_probs
        kl_per_token = torch.exp(-log_ratio) - 1 + log_ratio  # k3 estimator, always >= 0
        kl_penalty = (kl_per_token * generated_mask).sum() / max(1, total_generated_tokens) / len(trajectories)

        pg_loss = -(token_log_probs * advantage_mask).sum() / max(1, total_generated_tokens) / len(trajectories)
        traj_loss = pg_loss + kl_coef * kl_penalty
        (traj_loss * grad_scale).backward()
        total_loss_value += traj_loss.item()

        del outputs, logits, log_probs, token_log_probs, ref_outputs, ref_logits, ref_log_probs, ref_token_log_probs
        del advantage_mask, generated_mask, log_ratio, kl_per_token, kl_penalty, pg_loss, traj_loss
        torch.cuda.empty_cache()

    return total_loss_value


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def load_task_dataset(task: str, n_examples: int, seed: int):
    if task == "hotpotqa":
        from reflectrl.pipelines.hotpotqa_pipeline import load_devset
    else:
        from reflectrl.pipelines.aime_pipeline import load_devset
    _, examples = load_devset(n_train=0, n_dev=n_examples)
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def run(
    task: str, budget_rollouts: int, seed: int, variant: str, output_dir: str | None = None,
    learning_rate: float = LEARNING_RATE, group_size: int = GROUP_SIZE, kl_coef: float = KL_COEF,
):
    import anthropic
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    torch.manual_seed(seed)
    random.seed(seed)

    # Hyperparams promoted to CLI flags 2026-08-17 for step 1.6's LR/KL/
    # group-size tuning sweep (the set-up plan's suggested escape hatch
    # once the search "gets large" -- now a real 3-dimensional grid instead
    # of 2). Encoded into the default output_dir so sweep runs with
    # different hyperparams at the same (task, variant, budget, seed) don't
    # silently overwrite each other's checkpoint.
    output_dir = output_dir or str(
        WORKSHOP_ROOT / "checkpoints"
        / f"mmgrpo_{task}_{variant}_{budget_rollouts}_seed{seed}_lr{learning_rate}_gs{group_size}_kl{kl_coef}"
    )
    os.makedirs(output_dir, exist_ok=True)
    modules = ["generate_query", "generate_answer"] if task == "hotpotqa" else ["plan", "solve", "verify"]

    print(f"Loading {POLICY_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    # Needed on a single 80GB H100 at GROUP_SIZE=8 -- confirmed by real OOM on
    # the Phase 1 smoke test (2026-08-17): holding activations for the full
    # forward pass of every trajectory is too much even before compute_dr_grpo_loss's
    # per-trajectory-backward fix (see that function's docstring) is factored
    # in. enable_input_require_grads() is required alongside gradient
    # checkpointing specifically because PEFT freezes the base model's
    # embeddings (requires_grad=False) -- without it there is no tensor in the
    # recomputation path that requires grad, and backward() silently produces
    # no gradient for the LoRA params.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    device = next(model.parameters()).device

    # vLLM-accelerated rollout generation (added 2026-08-17) -- see
    # collect_trajectories_vllm's docstring for why (measured ~7 tok/s and
    # ~5x slower than this on real hardware) and REQUIRES the flashinfer
    # patch documented in the README (Python 3.11 only), or engine init
    # crashes. enforce_eager=True is required in THIS environment
    # specifically -- confirmed the crash persists with it alone (without
    # the patch) and the patch alone was only tested WITH enforce_eager=True,
    # not without it; keeping both is the one proven-working combination,
    # and enforce_eager also sidesteps torch.compile/CUDAGraph JIT risk
    # entirely, which barely matters at these short generation lengths
    # anyway.
    # FIXED 2026-08-18 -- real, reproducible CUDA OOM found live on
    # pod-shared-0, gepa_then_grpo only, crashing deep into normal training
    # steps (not during GEPA itself). First hypothesis (leftover cached
    # CUDA allocations from `import dspy` during run_gepa_for_instructions)
    # was WRONG -- a torch.cuda.empty_cache() call right after GEPA made
    # zero difference: the retried job crashed at the identical step (12)
    # with the identical training-process memory reading (44.77 GiB) as
    # the first failure, proving the cause isn't a one-time leftover but a
    # sustained effect through the whole run. Real cause, confirmed by
    # reading the job's own saved gepa_instructions.json: GEPA-optimized
    # instructions are 2262-2819 chars vs. DEFAULT_INSTRUCTIONS' ~130-190
    # -- ~15-20x longer, and gepa_then_grpo threads these into every
    # rollout's prompt for the ENTIRE run (not just the GEPA phase), so its
    # sequences are systematically longer than dr_grpo/rca's throughout
    # training. Longer sequences need proportionally more backward-pass
    # activation memory -- this is genuine, sustained extra memory
    # pressure on the TRAINING side, not vLLM's (vLLM's own reservation is
    # a fixed fraction of GPU memory regardless of prompt length, and its
    # engine's own footprint was identical, ~33.6GB, in every crash).
    # The actual shortfall each time was tiny (<1GB free at OOM time), so
    # a modest reduction in vLLM's fixed reservation -- freeing headroom
    # for the training side specifically when this variant's longer
    # instructions are in play -- is a safe, proportionate fix instead of
    # a blanket global cut that would waste vLLM throughput on every other
    # variant that never needed the extra margin.
    vllm_gpu_mem_util = 0.35 if variant in ("gepa_then_grpo", "gepa_onpolicy_then_grpo", "docstring_then_grpo") else VLLM_GPU_MEMORY_UTILIZATION
    print("Loading vLLM engine for group-batched rollout generation ...")
    vllm_engine = LLM(
        model=POLICY_MODEL, enable_lora=True, max_lora_rank=LORA_CONFIG["r"],
        gpu_memory_utilization=vllm_gpu_mem_util, dtype="bfloat16", enforce_eager=True,
    )

    lora_sync_dir = Path(output_dir) / "_lora_sync"
    lora_sync_dir.mkdir(exist_ok=True)
    lora_version = 0

    def sync_lora_to_vllm() -> LoRARequest:
        """Saves the training model's CURRENT LoRA adapter to disk and
        registers it with vLLM under a fresh numeric id -- vLLM's
        LoRARequest identifies adapters by id, so reusing the same id with
        different weights risks a stale cache hit instead of a real
        reload; a fresh id per sync avoids that ambiguity entirely. Only
        the PREVIOUS sync dir is kept around (not the whole history) to
        avoid accumulating hundreds of small adapter dirs on a long run
        (up to ~512 optimizer steps at the largest planned budget)."""
        nonlocal lora_version
        lora_version += 1
        path = lora_sync_dir / f"v{lora_version}"
        model.save_pretrained(str(path))
        request = LoRARequest(f"policy_v{lora_version}", lora_version, str(path))
        prev_path = lora_sync_dir / f"v{lora_version - 1}"
        if prev_path.exists():
            import shutil
            shutil.rmtree(prev_path, ignore_errors=True)
        return request

    lora_request = sync_lora_to_vllm()

    instructions: dict[str, str] | None = None
    if variant == "gepa_then_grpo":
        print("gepa_then_grpo: running GEPA to convergence first for initial instructions ...")
        instructions = run_gepa_for_instructions(task)
        print(f"  Optimized instructions obtained for modules: {list(instructions.keys())}")
        (Path(output_dir) / "gepa_instructions.json").write_text(json.dumps(instructions, indent=2))
    elif variant == "gepa_onpolicy_then_grpo":
        # Pre-computed by scripts/prompt_opt/gepa_onpolicy_instructions.py against a locally
        # served Qwen3-8B (the policy itself), one file per (task, seed). Fail loudly
        # if it is missing rather than silently training from defaults.
        src = WORKSHOP_ROOT / "data" / "gepa_onpolicy" / "instructions" / f"{task}_seed{seed}.json"
        if not src.is_file():
            raise FileNotFoundError(f"gepa_onpolicy_then_grpo needs {src}; run gepa_onpolicy_instructions.py first")
        payload = json.loads(src.read_text())
        instructions = payload["instructions"]
        expected = set(modules)
        if set(instructions) != expected:
            raise ValueError(f"{src} has modules {sorted(instructions)}, expected {sorted(expected)}")
        print(f"gepa_onpolicy_then_grpo: loaded on-policy GEPA instructions from {src.name} "
              f"(task model {payload.get('task_model')}, GEPA seed {payload.get('seed')})")
        (Path(output_dir) / "gepa_instructions.json").write_text(json.dumps(instructions, indent=2))
        (Path(output_dir) / "gepa_instructions_provenance.json").write_text(json.dumps(
            {k: v for k, v in payload.items() if k != "instructions"}, indent=2))
    elif variant == "docstring_then_grpo":
        # The unchanged DSPy signature docstrings, exactly what GEPA's seed program
        # carries. Read from the signature classes so they cannot drift from the file.
        if task == "hotpotqa":
            from reflectrl.pipelines.hotpotqa_pipeline import GenerateAnswer, GenerateQuery
            instructions = {"generate_query": GenerateQuery.instructions,
                            "generate_answer": GenerateAnswer.instructions}
        else:
            from reflectrl.pipelines.aime_pipeline import PlanSolution, SolveProblem, VerifySolution
            instructions = {"plan": PlanSolution.instructions, "solve": SolveProblem.instructions,
                            "verify": VerifySolution.instructions}
        print(f"docstring_then_grpo: using unchanged DSPy signature docstrings "
              f"({ {k: len(v) for k, v in instructions.items()} } chars)")
        (Path(output_dir) / "gepa_instructions.json").write_text(json.dumps(instructions, indent=2))

    reflection_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")) if variant in ("rca", "rca_no_validate") else None

    examples = load_task_dataset(task, n_examples=max(200, budget_rollouts // group_size), seed=seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    rollouts_used = 0
    step = 0
    log = []
    example_idx = 0
    n_unrenderable_skipped = 0
    while rollouts_used < budget_rollouts:
        example = examples[example_idx % len(examples)]
        example_idx += 1

        try:
            trajectories = collect_trajectories_vllm(task, vllm_engine, lora_request, tokenizer, example, group_size, instructions=instructions)
        except ValueError as exc:
            # FOUND LIVE 2026-09-04 (hotpotqa gepa_onpolicy seed 0, step 97): a module's
            # generated text was not found verbatim in the re-rendered chat transcript
            # (_compute_chat_spans -> "substring not found"). Qwen3's chat template
            # rewrites earlier assistant turns (it strips <think>...</think> reasoning
            # from history), so an output that happens to contain those tags cannot be
            # located after re-rendering. It never occurred in the 105 archived runs;
            # it did under a long evolved instruction that invites explicit reasoning.
            # Skipping the example is the fidelity-preserving choice: no budget is
            # consumed, nothing is trained on text the policy did not generate, and
            # the count is reported so the skip rate is visible.
            n_unrenderable_skipped += 1
            print(f"WARNING: skipped example {example_idx - 1} (unrenderable module output: {exc}); "
                  f"skipped so far={n_unrenderable_skipped}")
            continue
        for t in trajectories:
            t.question = example.question
            t.gold_answer = example.answer

        credits = [
            get_credit(variant, t, modules, reflection_client, task, example, model, tokenizer, device, instructions)
            for t in trajectories
        ]

        # compute_dr_grpo_loss backprops internally, per-trajectory (see its
        # docstring) -- grad_scale folds in GRAD_ACCUM_STEPS since this call
        # site no longer does its own .backward().
        loss_value = compute_dr_grpo_loss(model, tokenizer, trajectories, credits, device, grad_scale=1.0 / GRAD_ACCUM_STEPS, kl_coef=kl_coef)
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            # Re-sync vLLM's rollout-generation weights to match what was
            # just trained -- without this, every subsequent group would be
            # sampled from a stale (pre-update) policy.
            lora_request = sync_lora_to_vllm()

        mean_reward = sum(t.reward for t in trajectories) / len(trajectories)
        rollouts_used += group_size
        step += 1
        log.append({"step": step, "rollouts_used": rollouts_used, "mean_reward": mean_reward, "loss": loss_value})
        print(f"step={step} rollouts={rollouts_used}/{budget_rollouts} mean_reward={mean_reward:.3f} loss={loss_value:.4f}")

    model.save_pretrained(output_dir)
    (Path(output_dir) / "training_log.json").write_text(json.dumps(log, indent=2))
    (Path(output_dir) / "training_summary.json").write_text(json.dumps({
        "task": task, "variant": variant, "budget_rollouts": budget_rollouts, "seed": seed,
        "policy_model": POLICY_MODEL, "group_size": group_size, "learning_rate": learning_rate, "kl_coef": kl_coef,
        "final_mean_reward": log[-1]["mean_reward"] if log else None,
        "n_unrenderable_skipped": n_unrenderable_skipped,
    }, indent=2))
    import shutil
    shutil.rmtree(lora_sync_dir, ignore_errors=True)
    print(f"Done. Model + logs written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["hotpotqa", "aime"], required=True)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="step 1.6 tuning sweep dimension")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE, help="step 1.6 tuning sweep dimension")
    parser.add_argument("--kl-coef", type=float, default=KL_COEF, help="step 1.6 tuning sweep dimension")
    args = parser.parse_args()
    run(args.task, args.budget, args.seed, args.variant, args.output_dir, args.learning_rate, args.group_size, args.kl_coef)

# -----------------------------------------------------------------------------
# IMPLEMENTATION NOTES (failure modes found on real hardware, all fixed above)
#
# 1. Token spans. Each module's tokens are located by searching for its known
#    output string inside the single chat-templated full text and mapping the
#    character positions onto that tokenisation's offsets
#    (return_offsets_mapping=True). Counting tokens of separately tokenised
#    prefixes fails because BPE merges differ at concatenation boundaries, and
#    predicting boundaries from add_generation_prompt renders fails because
#    Qwen3's template inserts an empty <think></think> scaffold only before the
#    last assistant message.
#
# 2. Chat template and thinking mode. Raw-text prompts gave the instruction-tuned
#    policy no stop signal (rewards 0.003-0.008 on the first smoke test). Prompts
#    are built with tokenizer.apply_chat_template, and enable_thinking=False is
#    required: with thinking on, the 64-token query module emits only reasoning.
#
# 3. Memory. Backpropagating all GROUP_SIZE trajectories as one summed loss ran
#    out of memory on an 80 GB H100. compute_dr_grpo_loss backpropagates per
#    trajectory (mathematically identical), and gradient checkpointing is on
#    (with enable_input_require_grads, needed because PEFT freezes the base).
#
# 4. Rollout speed. HF generate() ran at about 7 tok/s; vLLM group-batched
#    generation with LoRA hot-swapping (sync_lora_to_vllm) is about 5x faster
#    and shares the GPU with the training model at gpu_memory_utilization=0.4.
#    On Python 3.11 vLLM's flashinfer dependency needs the one-line patch in the
#    README before the engine will start.
#
# 5. Interfaces. The answer module is scored on the span after an `answer:`
#    marker if one is present, else on the whole output (_extract_answer_span),
#    for every arm; the rule is a no-op on the default instructions. No
#    equivalent rule exists for the query module
#    (64-token cap, raw output passed to BM25), which is why evolved query
#    instructions that ask for reasoning first fail at this interface.
# -----------------------------------------------------------------------------
