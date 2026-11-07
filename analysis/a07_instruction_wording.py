#!/usr/bin/env python
"""A07 -- what did GEPA actually change? (2026-09-03)

Discovered while preparing the on-policy GEPA arm: the archived gepa_instructions.json
of every math cell in the transplanted `gepa_then_grpo` arm is byte-identical to the
DSPy signature docstrings of reflectrl/pipelines/aime_pipeline.py, i.e. GEPA returned its seed
program unchanged in all 12 cells. On HotpotQA it left `generate_query` unchanged in
4 of 12 cells and `generate_answer` unchanged in 2. The "GEPA arm" therefore measures,
in large part, the WORDING difference between the DSPy docstrings and the trainer's
DEFAULT_INSTRUCTIONS, not the effect of GEPA's edits.

This script makes that inspection reproducible: for every gepa_then_grpo run it flags
each module as `docstring` (identical to the signature's instructions) or `evolved`,
joins the run means of that cell's gepa and dr_grpo arms, and summarises the gain by
flag. Standard library plus the pipelines' signature classes (dspy import only).

    python analysis/a07_instruction_wording.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from lib_runs import ROOT, RUNS_DIR, load_runs

BASE = {"lr": 1e-6, "gs": 8, "kl": 0.04}


def docstrings() -> dict[str, dict[str, str]]:
    """The seed instructions are the DSPy signature docstrings. Read them from the
    pipeline SOURCE with `ast` (inspect.cleandoc semantics, which is what DSPy's
    Signature.instructions applies) so this script stays standard-library only."""
    import ast

    def doc(path: Path, cls: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                return ast.get_docstring(node, clean=True)
        raise KeyError(cls)

    hp = ROOT / "reflectrl" / "pipelines" / "hotpotqa_pipeline.py"
    am = ROOT / "reflectrl" / "pipelines" / "aime_pipeline.py"
    return {
        "hotpotqa": {"generate_query": doc(hp, "GenerateQuery"), "generate_answer": doc(hp, "GenerateAnswer")},
        "aime": {"plan": doc(am, "PlanSolution"), "solve": doc(am, "SolveProblem"), "verify": doc(am, "VerifySolution")},
    }


def main() -> None:
    docs = docstrings()
    runs = {(r.task, r.variant, r.budget, r.seed): r for r in load_runs()
            if r.rewards and r.lr == BASE["lr"] and r.gs == BASE["gs"] and r.kl == BASE["kl"]}
    out: dict = {"cells": [], "summary": {}}
    print("=" * 96)
    print("A07 -- DID GEPA CHANGE THE INSTRUCTIONS? per cell of the transplanted gepa_then_grpo arm")
    print("=" * 96)
    for task in ("hotpotqa", "aime"):
        mods = list(docs[task])
        print(f"\n{task}:  budget seed  " + "  ".join(f"{m[:14]:>14s}" for m in mods) + "     gepa  dr_grpo    diff")
        for d in sorted(RUNS_DIR.glob(f"{task}_gepa_then_grpo_*")):
            f = d / "gepa_instructions.json"
            if not f.is_file():
                continue
            ins = json.loads(f.read_text(encoding="utf-8"))
            parts = d.name.replace("_lr1e-06_gs8_kl0.04", "").split("_")
            budget, seed = int(parts[-2]), int(parts[-1].replace("seed", ""))
            g = runs.get((task, "gepa_then_grpo", budget, seed))
            dr = runs.get((task, "dr_grpo", budget, seed))
            if not (g and dr):
                continue
            flags = {m: ("docstring" if ins[m] == docs[task][m] else "evolved") for m in mods}
            row = {"task": task, "budget": budget, "seed": seed, "flags": flags,
                   "chars": {m: len(ins[m]) for m in mods},
                   "gepa": g.mean_reward, "dr_grpo": dr.mean_reward, "diff": g.mean_reward - dr.mean_reward,
                   "gepa_zero_frac": g.zero_reward_frac, "dr_grpo_zero_frac": dr.zero_reward_frac}
            out["cells"].append(row)
            print(f"  {budget:12d} {seed:4d}  " + "  ".join(f"{flags[m][:9]:>9s}({row['chars'][m]:4d})" for m in mods)
                  + f"   {g.mean_reward:.4f}  {dr.mean_reward:.4f}  {row['diff']:+.4f}")
        cells = [c for c in out["cells"] if c["task"] == task]
        summ = {}
        for m in mods:
            for flag in ("docstring", "evolved"):
                sel = [c["diff"] for c in cells if c["flags"][m] == flag]
                if sel:
                    summ[f"{m}={flag}"] = {"n": len(sel), "mean_diff": mean(sel), "min": min(sel), "max": max(sel)}
                    print(f"  {m}={flag:9s}  n={len(sel):2d}  mean gain over dr_grpo={mean(sel):+.4f}  "
                          f"[{min(sel):+.4f}, {max(sel):+.4f}]")
        n_unchanged = sum(1 for c in cells if all(v == "docstring" for v in c["flags"].values()))
        print(f"  cells with EVERY module unchanged: {n_unchanged}/{len(cells)}")
        summ["cells_all_unchanged"] = n_unchanged
        summ["n_cells"] = len(cells)
        out["summary"][task] = summ

    dest = ROOT / "results" / "a07_instruction_wording.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
