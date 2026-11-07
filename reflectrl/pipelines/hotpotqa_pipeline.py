"""
Step 1.3 -- HotpotQA compound system: query-gen -> retrieve -> answer.

3-module DSPy program, DSPy-standard (recognizable to organizers). Each module
exposes a GEPA-optimizable prompt via its Signature docstring.

Retrieval note (decisions.md, 2026-08-16): the public ColBERTv2 wiki17-abstracts
demo server DSPy's own tutorials point at (20.102.90.50:2017) is unreachable
from this environment -- and depending on someone else's demo box for a
paper's reproducibility is fragile regardless. Retrieval instead runs locally
with BM25 (rank_bm25) over each HotpotQA example's own 10-paragraph
distractor pool -- this is the standard HotpotQA "distractor setting" from
the original paper, not an improvised workaround, and it makes the pipeline
fully self-contained (no external service, deterministic, works in a fresh
clone per Step 3.6's reproducibility bar).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import dspy

from reflectrl.pipelines.metering import cost_tag, dspy_usage_callback, flush_lm_history  # noqa: E402

HF_HOTPOTQA_REPO = "hotpotqa/hotpot_qa"


class GenerateQuery(dspy.Signature):
    """Given a multi-hop question, write a short search query that would
    retrieve a Wikipedia passage containing the next piece of evidence needed
    to answer it. Do not try to answer the question directly."""

    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a concise search query, not a restatement of the question")


class GenerateAnswer(dspy.Signature):
    """Given a multi-hop question and retrieved context passages, answer the
    question. Answer with the shortest span of text that correctly answers
    the question — no full sentences, no explanation."""

    context: str = dspy.InputField(desc="retrieved passages, most relevant first")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a short factual answer, a few words at most")


class BM25DistractorRetriever:
    """Retrieves from one HotpotQA example's own 10-paragraph distractor pool
    (title + sentences) via BM25 -- the standard distractor-setting corpus,
    built fresh per example since each question ships its own pool."""

    def __init__(self, titles: list[str], sentences: list[list[str]]):
        from rank_bm25 import BM25Okapi

        self.titles = titles
        self.passages = ["".join(sents) for sents in sentences]
        tokenized = [p.lower().split() for p in self.passages]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(self.passages)), key=lambda i: scores[i], reverse=True)[:k]
        return [f"{self.titles[i]} | {self.passages[i]}" for i in ranked]


class HotpotQAPipeline(dspy.Module):
    """query-gen -> retrieve -> answer, per Step 1.3 of the execution plan.
    `context` (an example's {"title": [...], "sentences": [[...]]} dict, as
    shipped by the HotpotQA distractor setting) must be passed at call time
    -- retrieval is local to each question's own paragraph pool."""

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str, context: dict):
        # No internal cost_tag/hardcoded flush tag -- see the matching note
        # in legibility_judge.py (2026-08-16): a hardcoded inner tag here
        # silently overrode every caller's own outer cost_tag (innermost
        # context wins), which is why decisions.md's earlier cost
        # breakdowns only ever showed "hotpotqa-pipeline" instead of the
        # calling script's own more specific tags. Every current caller
        # (gepa_repro.py, judge_bakeoff.py, generate_instances.py,
        # score_credit_methods.py) already sets its own meaningful outer
        # cost_tag, so falling back to the ambient tag is strictly more
        # accurate, not a regression.
        retriever = BM25DistractorRetriever(context["title"], context["sentences"])
        query_pred = self.generate_query(question=question)
        query = query_pred.query
        passages = retriever.retrieve(query, k=self.k)
        joined_context = "\n\n".join(passages)
        answer_pred = self.generate_answer(context=joined_context, question=question)
        answer = answer_pred.answer
        flush_lm_history(dspy.settings.lm)
        # Structured trajectory for the legibility judge (Step 1.6) --
        # module order, each module's key inputs/outputs, and its ChainOfThought
        # reasoning trace. Kept separate from the plain fields above so
        # nothing that already reads pred.answer/pred.query breaks.
        trajectory = [
            {
                "module": "generate_query",
                "inputs": {"question": question},
                "reasoning": query_pred.reasoning,
                "outputs": {"query": query},
            },
            {
                "module": "generate_answer",
                "inputs": {"question": question, "context": passages},
                "reasoning": answer_pred.reasoning,
                "outputs": {"answer": answer},
            },
        ]
        return dspy.Prediction(context=passages, query=query, answer=answer, trajectory=trajectory)


def configure_lm(model: str = "anthropic/claude-haiku-4-5", max_tokens: int = 1024) -> None:
    """Wires the policy/reflection LM and the metering callback into DSPy's
    global settings. Call once at the top of any script that runs this
    pipeline. Default model is the cheap bake-off candidate for pipeline
    smoke-testing, NOT the frozen Step 1.6 judge/reflection model."""
    lm = dspy.LM(model, max_tokens=max_tokens, api_key=os.environ["ANTHROPIC_API_KEY"])
    dspy.configure(lm=lm, callbacks=[dspy_usage_callback()])


def load_devset(n_train: int = 20, n_dev: int = 50, hard_only_dev: bool = True):
    """Loads HotpotQA (fullwiki config, which ships the 10-paragraph
    distractor pool per example) from the actively-maintained
    `hotpotqa/hotpot_qa` HF mirror -- DSPy's own bundled loader points at the
    legacy `hotpot_qa` repo, which the current `datasets` library can no
    longer resolve (script-based dataset, no namespace). Dev is restricted to
    level="hard" to match the official HotpotQA dev/DSPy-standard convention;
    train is not, matching dspy.datasets.HotPotQA's own default behavior."""
    from datasets import load_dataset

    train_raw = load_dataset(HF_HOTPOTQA_REPO, "fullwiki", split="train")
    dev_raw = load_dataset(HF_HOTPOTQA_REPO, "fullwiki", split="validation")

    def to_example(raw) -> dspy.Example:
        return dspy.Example(
            question=raw["question"],
            answer=raw["answer"],
            context=raw["context"],
        ).with_inputs("question", "context")

    rng = random.Random(0)
    train_idx = list(range(len(train_raw)))
    rng.shuffle(train_idx)
    trainset = [to_example(train_raw[i]) for i in train_idx[:n_train]]

    dev_pool = [dev_raw[i] for i in range(len(dev_raw)) if (not hard_only_dev or dev_raw[i]["level"] == "hard")]
    rng.shuffle(dev_pool)
    devset = [to_example(r) for r in dev_pool[:n_dev]]

    return trainset, devset


def normalize_answer(s: str) -> str:
    """Standard SQuAD/HotpotQA answer normalization (lowercase, strip
    articles/punctuation, collapse whitespace) — needed before EM/F1."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def em_f1_metric(example, pred, trace=None) -> float:
    """Combined metric for dspy.Evaluate / GEPA: mean of EM and F1 against
    the gold answer. Matches the plan's 'EM+F1' eval spec (Step 1.3)."""
    gold = example.answer
    return (exact_match(pred.answer, gold) + f1_score(pred.answer, gold)) / 2.0
