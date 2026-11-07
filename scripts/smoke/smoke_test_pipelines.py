"""One-off smoke test for both Step 1.3 pipelines. Run manually:
    python scripts/smoke/smoke_test_pipelines.py
Runs one hand-picked example through each pipeline on Haiku 4.5 (cheap) to
confirm module wiring, retrieval connectivity, and metering all work
end-to-end -- this is the CI smoke test Step 1.1 calls for, run by hand
until a CI runner exists.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from reflectrl.pipelines.aime_pipeline import AIMEStylePipeline, answer_check_metric
from reflectrl.pipelines.aime_pipeline import configure_lm as configure_aime_lm
from reflectrl.pipelines.hotpotqa_pipeline import HotpotQAPipeline, em_f1_metric
from reflectrl.pipelines.hotpotqa_pipeline import configure_lm as configure_hotpot_lm
from reflectrl.pipelines.metering import METER

import dspy

print("=== HotpotQA pipeline smoke test ===")
configure_hotpot_lm()
hotpot = HotpotQAPipeline(k=3)
from reflectrl.pipelines.hotpotqa_pipeline import load_devset as load_hotpot_devset

_, hotpot_dev = load_hotpot_devset(n_train=0, n_dev=1)
example = hotpot_dev[0]
pred = hotpot(question=example.question, context=example.context)
print("question:", example.question)
print("query:", pred.query)
print("answer:", pred.answer, "| gold:", example.answer)
print("em_f1:", em_f1_metric(example, pred))

print("\n=== AIME-style pipeline smoke test ===")
configure_aime_lm()
aime = AIMEStylePipeline()
math_example = dspy.Example(
    question="What is the sum of the first 10 positive integers?",
    answer="55",
).with_inputs("question")
math_pred = aime(question=math_example.question)
print("plan:", math_pred.plan[:200])
print("answer:", math_pred.answer)
print("score:", answer_check_metric(math_example, math_pred))

print(f"\nTotal metered spend this run: ${METER.total_usd:.6f}")
