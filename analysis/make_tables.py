#!/usr/bin/env python
"""Generate the LaTeX appendix tables from the archived runs (standard library only).

    python analysis/make_tables.py

Writes results/tables/percell.tex (every run of the main grid) and
results/tables/promptcells.tex (the prompt arms at budget 2048 with per-module
wording flags from a07). Both read only data/runs and results, so the
tables cannot drift from the reported numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

from lib_runs import ROOT, load_runs

TABLES = ROOT / "results" / "tables"
BS = chr(92)  # backslash, built at runtime so no shell layer can halve it


def anchor(runs):
    idx: dict = {}
    for r in runs:
        if r.unpatched or abs(r.lr - 1e-6) > 1e-12 or r.gs != 8 or abs(r.kl - 0.04) > 1e-9:
            continue
        idx[(r.task, r.variant, r.budget, r.seed)] = r
    return idx


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def diff(a, b):
    return "n/a" if a is None or b is None else f"{a - b:+.3f}"


def percell(idx) -> str:
    L = []
    L.append(BS + "begin{table}[!htbp]")
    L.append("  " + BS + "caption{Mean group reward per run for every cell of the main grid (learning rate $10^{-6}$, group size 8, KL 0.04). Each entry is the mean over all steps of that run; the difference columns are what the cluster-robust tests in Sections 4.1 and 4.3 operate on. HP = HotpotQA, M = competition math.}")
    L.append("  " + BS + "label{tab:percell}")
    L.append("  " + BS + "centering")
    L.append("  " + BS + "scriptsize")
    L.append("  " + BS + "setlength{" + BS + "tabcolsep}{4pt}")
    L.append("  " + BS + "begin{tabular}{@{}llr rrrr rr@{}}")
    L.append("    " + BS + "toprule")
    L.append("    Task & Budget & Seed & " + BS + "drgrpo{} & " + BS + "rca{} & " + BS + "rca{} no-val. & GEPA$" + BS + "rightarrow$GRPO & " + BS + "rca{}$-$" + BS + "drgrpo{} & GEPA$-$" + BS + "drgrpo{} " + BS + BS)
    L.append("    " + BS + "midrule")
    for task, lab in (("hotpotqa", "HP"), ("aime", "M")):
        for b in (128, 256, 512, 2048, 8192):
            for s in (0, 1, 2):
                g = idx.get((task, "dr_grpo", b, s)); rc = idx.get((task, "rca", b, s))
                rn = idx.get((task, "rca_no_validate", b, s)); ge = idx.get((task, "gepa_then_grpo", b, s))
                m = lambda r: r.mean_reward if r else None
                L.append(f"    {lab} & {b} & {s} & {fmt(m(g))} & {fmt(m(rc))} & {fmt(m(rn))} & {fmt(m(ge))} & {diff(m(rc), m(g))} & {diff(m(ge), m(g))} " + BS + BS)
        if task == "hotpotqa":
            L.append("    " + BS + "midrule")
    L.append("    " + BS + "bottomrule")
    L.append("  " + BS + "end{tabular}")
    L.append(BS + "end{table}")
    return "\n".join(L) + "\n"


def promptcells(idx) -> str:
    a07 = json.loads((ROOT / "results" / "a07_instruction_wording.json").read_text(encoding="utf-8"))
    flags = {(c["task"], c["budget"], c["seed"]): c["flags"] for c in a07["cells"]}
    docs = {"hotpotqa": {"generate_query": "doc", "generate_answer": "doc"},
            "aime": {"plan": "doc", "solve": "doc", "verify": "doc"}}

    def flagstr(task, variant, b, s):
        if variant == "docstring_then_grpo":
            f = docs[task]
        elif variant == "gepa_then_grpo":
            f = {k: ("doc" if v == "docstring" else "evo") for k, v in flags.get((task, b, s), {}).items()}
        else:
            # on-policy: read the instruction file the run trained from
            r = idx.get((task, variant, b, s))
            d = (ROOT / "data" / "runs" / r.name / "gepa_instructions.json") if r else None
            if d and d.is_file():
                ins = json.loads(d.read_text(encoding="utf-8"))
                lens = {"generate_query": 197, "generate_answer": 190, "plan": 142, "solve": 131, "verify": 223}
                f = {k: ("doc" if len(v) == lens[k] else "evo") for k, v in ins.items()}
            else:
                f = {}
        return ", ".join(f"{k.split('_')[-1]}={v}" for k, v in f.items()) if f else "n/a"

    L = []
    L.append(BS + "begin{table}[!htbp]")
    L.append("  " + BS + "caption{The prompt arms at budget 2048: run mean, zero-reward step rate, and whether each module's instruction was GEPA's unchanged seed docstring (doc) or edited by GEPA (evo). The transplanted arm evolved prompts against Claude Haiku~4.5; the on-policy arm against Qwen3-8B itself; the docstring arm ran no GEPA. Math has no separate docstring arm because every transplanted math cell already trained from the docstrings.}")
    L.append("  " + BS + "label{tab:promptcells}")
    L.append("  " + BS + "centering")
    L.append("  " + BS + "scriptsize")
    L.append("  " + BS + "begin{tabular}{@{}llrrrl@{}}")
    L.append("    " + BS + "toprule")
    L.append("    Task & Arm & Seed & Mean & Zero-reward & Instructions " + BS + BS)
    L.append("    " + BS + "midrule")
    names = {"gepa_then_grpo": "GEPA transplanted", "gepa_onpolicy_then_grpo": "GEPA on policy", "docstring_then_grpo": "docstrings, no GEPA", "dr_grpo": BS + "drgrpo{} (defaults)"}
    for task, lab in (("hotpotqa", "HotpotQA"), ("aime", "math")):
        for variant in ("dr_grpo", "docstring_then_grpo", "gepa_onpolicy_then_grpo", "gepa_then_grpo"):
            for s in (0, 1, 2):
                r = idx.get((task, variant, 2048, s))
                if not r:
                    continue
                ins = "default text" if variant == "dr_grpo" else flagstr(task, variant, 2048, s)
                L.append(f"    {lab} & {names[variant]} & {s} & {r.mean_reward:.3f} & {r.zero_reward_frac:.2f} & {ins} " + BS + BS)
        if task == "hotpotqa":
            L.append("    " + BS + "midrule")
    L.append("    " + BS + "bottomrule")
    L.append("  " + BS + "end{tabular}")
    L.append(BS + "end{table}")
    return "\n".join(L) + "\n"


def main() -> None:
    idx = anchor(load_runs())
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "percell.tex").write_text(percell(idx), encoding="utf-8")
    (TABLES / "promptcells.tex").write_text(promptcells(idx), encoding="utf-8")
    print("wrote", TABLES / "percell.tex", "and", TABLES / "promptcells.tex")


if __name__ == "__main__":
    main()
