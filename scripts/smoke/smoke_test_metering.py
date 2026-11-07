"""One-off smoke test for the cost meter (Step 1.2). Run manually:
    python scripts/smoke/smoke_test_metering.py
Makes one cheap Haiku 4.5 call and confirms it's metered and logged.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import anthropic

from reflectrl.pipelines.metering import METER, cost_tag, metered_anthropic_call

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

with cost_tag("smoke-test"):
    response = metered_anthropic_call(
        client,
        model="claude-haiku-4-5",
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly: metering-ok"}],
    )

text = next(b.text for b in response.content if b.type == "text")
print("Model replied:", text)
print(f"Running total after this call: ${METER.total_usd:.6f}")
