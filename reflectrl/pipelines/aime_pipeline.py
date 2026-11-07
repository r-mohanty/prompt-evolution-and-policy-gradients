"""
Step 1.3 — AIME-style math compound system: plan -> solve -> verify.

3-module DSPy program over competition-level math. Data source is DSPy's
built-in MATH (Hendrycks MATH-lighteval) dataset filtered to Level 5 (the
hardest tier, AIME-comparable difficulty), used as the documented proxy for
AIME per the execution plan's Step 1.3 allowance -- swap in real AIME
problem sets later without touching the pipeline interface if a suitable
public set is vetted. Grading avoids the external `math_equivalence` package
(a fragile git-install dependency) in favor of a boxed-answer + numeric/
normalized-string exact-match check, closer to how AIME (integer 0-999
answers) is actually graded.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import dspy

from reflectrl.pipelines.metering import cost_tag, dspy_usage_callback, flush_lm_history  # noqa: E402


class PlanSolution(dspy.Signature):
    """Given a competition math problem, write a short plan: which technique
    or theorem applies, and the key steps, without doing the arithmetic yet."""

    problem: str = dspy.InputField()
    plan: str = dspy.OutputField(desc="2-5 sentence solution strategy, no final answer")


class SolveProblem(dspy.Signature):
    """Given a competition math problem and a solution plan, work the
    problem to completion. End with the final answer inside \\boxed{...}."""

    problem: str = dspy.InputField()
    plan: str = dspy.InputField()
    solution: str = dspy.OutputField(desc="full worked solution ending in \\boxed{answer}")


class VerifySolution(dspy.Signature):
    """Given a competition math problem and a proposed solution, check the
    arithmetic and logic independently. If correct, restate the boxed answer
    unchanged. If incorrect, redo the calculation and give the corrected
    boxed answer."""

    problem: str = dspy.InputField()
    solution: str = dspy.InputField()
    verified_solution: str = dspy.OutputField(desc="corrected/confirmed solution ending in \\boxed{answer}")


class AIMEStylePipeline(dspy.Module):
    """plan -> solve -> verify, per Step 1.3 of the execution plan."""

    def __init__(self):
        super().__init__()
        self.plan = dspy.ChainOfThought(PlanSolution)
        self.solve = dspy.ChainOfThought(SolveProblem)
        self.verify = dspy.ChainOfThought(VerifySolution)

    def forward(self, question: str):
        # No internal cost_tag/hardcoded flush tag -- see the matching note
        # in hotpotqa_pipeline.py / legibility_judge.py (2026-08-16): a
        # hardcoded inner tag here silently overrode every caller's own
        # outer cost_tag. Falling back to the ambient tag is strictly more
        # accurate for every current caller.
        plan_pred = self.plan(problem=question)
        plan = plan_pred.plan
        solve_pred = self.solve(problem=question, plan=plan)
        solution = solve_pred.solution
        verify_pred = self.verify(problem=question, solution=solution)
        verified = verify_pred.verified_solution
        answer = extract_boxed_answer(verified) or extract_boxed_answer(solution) or ""
        flush_lm_history(dspy.settings.lm)
        # See hotpotqa_pipeline.forward -- same structured trajectory for
        # the Step 1.6 legibility judge.
        trajectory = [
            {"module": "plan", "inputs": {"problem": question}, "reasoning": plan_pred.reasoning, "outputs": {"plan": plan}},
            {"module": "solve", "inputs": {"problem": question, "plan": plan}, "reasoning": solve_pred.reasoning, "outputs": {"solution": solution}},
            {"module": "verify", "inputs": {"problem": question, "solution": solution}, "reasoning": verify_pred.reasoning, "outputs": {"verified_solution": verified}},
        ]
        return dspy.Prediction(plan=plan, solution=solution, verified_solution=verified, answer=answer, trajectory=trajectory)


def configure_lm(model: str = "anthropic/claude-haiku-4-5", max_tokens: int = 2048) -> None:
    """See hotpotqa_pipeline.configure_lm — same wiring, different default
    max_tokens (math solutions run longer than QA answers)."""
    lm = dspy.LM(model, max_tokens=max_tokens, api_key=os.environ["ANTHROPIC_API_KEY"])
    dspy.configure(lm=lm, callbacks=[dspy_usage_callback()])


def extract_boxed_answer(s: str) -> str | None:
    """Same \\boxed{...} extraction dspy.datasets.math uses, so gold and
    predicted answers are comparable."""
    start = s.find("\\boxed{")
    if start == -1:
        return None
    idx = start + len("\\boxed{")
    brace_level = 1
    answer = ""
    while idx < len(s) and brace_level > 0:
        c = s[idx]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        answer += c
        idx += 1
    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\!", "", answer)
    return answer.strip()


def normalize_math_answer(s: str) -> str:
    """Loose normalization for comparing boxed answers: strip whitespace,
    dollar signs, trailing punctuation, and normalize simple fraction/
    integer formatting differences."""
    s = s.strip().strip("$").strip()
    s = s.replace(" ", "")
    s = s.rstrip(".")
    s = re.sub(r"^\\!", "", s)
    return s


def answer_check_metric(example, pred, trace=None) -> float:
    """AIME-style exact-match on the boxed final answer (Step 1.3's
    'answer-check eval'). Falls back to raw numeric comparison when both
    sides parse as plain integers/floats (the common AIME case)."""
    gold = normalize_math_answer(example.answer or "")
    got = normalize_math_answer(pred.answer or "")
    if not gold or not got:
        return 0.0
    if gold == got:
        return 1.0
    try:
        return float(float(gold) == float(got))
    except ValueError:
        return 0.0


def load_devset(n_train: int = 20, n_dev: int = 50, level: str = "Level 5"):
    """Loads Hendrycks MATH-lighteval via DSPy's built-in loader, filtered to
    the requested difficulty level (default: Level 5, AIME-comparable).
    Network + first-run HF dataset download required."""
    from datasets import load_dataset

    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
    examples = []
    for split_name in ("test",):
        for ex in ds[split_name]:
            if level is not None and ex.get("level") != level:
                continue
            answer = extract_boxed_answer(ex["solution"])
            if not answer:
                continue
            examples.append(dspy.Example(question=ex["problem"], answer=answer).with_inputs("question"))

    import random

    random.Random(0).shuffle(examples)
    trainset = examples[:n_train]
    devset = examples[n_train : n_train + n_dev]
    return trainset, devset
