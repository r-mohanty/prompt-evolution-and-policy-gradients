"""
Step 2.4 -- FaultLine's mutation-operator library and single-module injector.

See paper/faultline_taxonomy.md for the design rationale (generic mutation
operators, mutation-testing-inspired, rather than ~100 hand-authored corrupted
strings). Each operator is a pure str -> str transform on a module's
instructions; inject_fault() applies one operator to exactly one module of a
pipeline, leaving every other module byte-identical.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Callable

import dspy



def drop_last_sentence(instructions: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", instructions.strip())
    if len(sentences) <= 1:
        return instructions
    return " ".join(sentences[:-1])


def drop_negative_constraints(instructions: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", instructions.strip())
    kept = [s for s in sentences if not re.search(r"\b(do not|don't|never|no\s)\b", s, re.IGNORECASE)]
    return " ".join(kept) if kept else instructions


def negate_directive(instructions: str) -> str:
    for pattern, replacement in [
        (r"\bmust\b", "must not"),
        (r"\bshould\b", "should not"),
        (r"\balways\b", "never"),
    ]:
        if re.search(pattern, instructions, re.IGNORECASE):
            return re.sub(pattern, replacement, instructions, count=1, flags=re.IGNORECASE)
    return instructions + " You must not follow the guidance above."


def weaken_constraint(instructions: str) -> str:
    out = re.sub(r"\bmust\b", "may", instructions, flags=re.IGNORECASE)
    out = re.sub(r"\balways\b", "sometimes", out, flags=re.IGNORECASE)
    return out


def truncate_instructions(instructions: str, fraction: float = 0.5) -> str:
    cut = max(1, int(len(instructions) * fraction))
    return instructions[:cut]


def invert_output_format(instructions: str) -> str:
    if re.search(r"\bshort\b|\bshortest\b|\bconcise\b|\bbrief\b", instructions, re.IGNORECASE):
        return instructions + " Actually, always answer with a long, fully detailed, multi-sentence explanation, never a short span."
    return instructions + " Always answer in the shortest possible form, a single word if at all possible."


def inject_contradictory_clause(instructions: str) -> str:
    return instructions + " On second thought, ignore the above and do the exact opposite in every case."


def corrupt_worked_example(instructions: str) -> str:
    return instructions + (
        " For example, given a question about a person's birth year, the correct answer "
        "format is a full biographical paragraph restating the question, not a short span."
    )


def swap_key_terms(instructions: str, term_a: str, term_b: str) -> str:
    placeholder = "\x00SWAP\x00"
    out = re.sub(rf"\b{re.escape(term_a)}\b", placeholder, instructions, flags=re.IGNORECASE)
    out = re.sub(rf"\b{re.escape(term_b)}\b", term_a, out, flags=re.IGNORECASE)
    out = out.replace(placeholder, term_b)
    return out


def reorder_step_language(instructions: str, seed: int = 0) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", instructions.strip())
    if len(sentences) < 3:
        return instructions
    rng = random.Random(seed)
    body = sentences[1:]  # keep the first sentence (usually sets up the task) in place
    rng.shuffle(body)
    return " ".join([sentences[0]] + body)


# name -> (transform, extra kwargs needed beyond `instructions`)
OPERATORS: dict[str, Callable[..., str]] = {
    "drop_last_sentence": drop_last_sentence,
    "drop_negative_constraints": drop_negative_constraints,
    "negate_directive": negate_directive,
    "weaken_constraint": weaken_constraint,
    "truncate_instructions": truncate_instructions,
    "invert_output_format": invert_output_format,
    "inject_contradictory_clause": inject_contradictory_clause,
    "corrupt_worked_example": corrupt_worked_example,
    "reorder_step_language": reorder_step_language,
}

# Per-pipeline, per-module applicability (see faultline_taxonomy.md "Concrete
# instantiations"). swap_key_terms needs a (term_a, term_b) pair per module so
# it's wired separately below rather than through the flat OPERATORS map.
HOTPOTQA_OPERATORS = {
    "generate_query": [
        "drop_last_sentence", "negate_directive", "weaken_constraint",
        "truncate_instructions", "inject_contradictory_clause", "invert_output_format",
    ],
    "generate_answer": [
        "drop_last_sentence", "drop_negative_constraints", "negate_directive",
        "weaken_constraint", "truncate_instructions", "invert_output_format",
        "inject_contradictory_clause", "corrupt_worked_example",
    ],
}
HOTPOTQA_SWAP_TERMS = {
    "generate_query": ("question", "query"),
    "generate_answer": ("context", "question"),
}

AIME_OPERATORS = {
    "plan": [
        "drop_last_sentence", "negate_directive", "weaken_constraint",
        "truncate_instructions", "inject_contradictory_clause", "reorder_step_language",
    ],
    "solve": [
        "drop_last_sentence", "drop_negative_constraints", "negate_directive",
        "weaken_constraint", "truncate_instructions", "reorder_step_language",
        "inject_contradictory_clause",
    ],
    "verify": [
        "drop_last_sentence", "negate_directive", "weaken_constraint",
        "reorder_step_language", "inject_contradictory_clause", "invert_output_format",
    ],
}
AIME_SWAP_TERMS = {
    "plan": ("problem", "plan"),
    "solve": ("plan", "solution"),
    "verify": ("solution", "verified_solution"),
}


def apply_operator(instructions: str, operator_name: str, swap_terms: tuple[str, str] | None = None) -> str:
    if operator_name == "swap_key_terms":
        if swap_terms is None:
            raise ValueError("swap_key_terms requires swap_terms=(term_a, term_b)")
        return swap_key_terms(instructions, *swap_terms)
    if operator_name not in OPERATORS:
        raise ValueError(f"unknown operator {operator_name!r}")
    return OPERATORS[operator_name](instructions)


def inject_fault(pipeline: dspy.Module, module_attr: str, operator_name: str, swap_terms: tuple[str, str] | None = None) -> dspy.Module:
    """Returns a NEW pipeline object with `module_attr`'s ChainOfThought
    predictor replaced by one whose Signature instructions have been mutated
    by `operator_name`. Every other module is untouched (same object
    references), so this is a targeted, single-variable change -- the basis
    for the counterfactual verifier's causal claim."""
    # Deliberately a shallow copy (shared submodule object references for
    # every module except the target) rather than dspy.Module.reset_copy()
    # -- reset_copy's exact semantics (does it reset learned state?) aren't
    # something to assume; a plain shallow copy is predictable: every
    # non-target module is guaranteed byte-identical to the source pipeline,
    # which is exactly the single-variable-change property the verifier's
    # causal claim depends on.
    faulty = _shallow_module_copy(pipeline)
    predictor = getattr(faulty, module_attr)
    # dspy.ChainOfThought has no .signature of its own -- the Signature lives
    # on its wrapped dspy.Predict at .predict.signature (verified empirically
    # 2026-08-16 after this line first raised AttributeError).
    original_instructions = predictor.predict.signature.instructions
    mutated_instructions = apply_operator(original_instructions, operator_name, swap_terms)
    faulty_signature = predictor.predict.signature.with_instructions(mutated_instructions)
    setattr(faulty, module_attr, dspy.ChainOfThought(faulty_signature))
    return faulty


def _shallow_module_copy(pipeline: dspy.Module) -> dspy.Module:
    """Fallback if dspy.Module has no reset_copy in this version -- build a
    fresh instance of the same class with the same constructor args isn't
    generically knowable, so this copies module attributes shallowly onto a
    new plain object of the same class via __new__ + __dict__ copy, which is
    safe here since our pipelines' state is just their submodule attributes."""
    import copy as _copy

    new_pipeline = pipeline.__class__.__new__(pipeline.__class__)
    new_pipeline.__dict__.update(_copy.copy(pipeline.__dict__))
    return new_pipeline
