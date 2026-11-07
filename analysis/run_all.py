#!/usr/bin/env python
"""Reproduce every headline number in the paper from the archived run data.

    python analysis/extract_runs.py    # once, to build data/runs/
    python analysis/run_all.py

Standard library only -- no numpy/scipy required.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ["a01_variant_comparison.py", "a02_grpo_learning.py", "a03_faultline.py",
           "a05_legibility.py", "a06_onpolicy_gepa.py", "a07_instruction_wording.py"]


def main() -> None:
    if not (ROOT / "data" / "runs").is_dir():
        raise SystemExit("no run data -- run `python analysis/extract_runs.py` first")
    for s in SCRIPTS:
        print("\n" + "#" * 92)
        print(f"# {s}")
        print("#" * 92)
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"{s} failed with code {r.returncode}")

    print("\n" + "=" * 92)
    print("HEADLINE NUMBERS")
    print("=" * 92)
    res = ROOT / "results"

    a01 = json.loads((res / "a01_variant_comparison.json").read_text())
    k = a01["rca_vs_dr_grpo"]
    print(f"  RCA vs Dr. GRPO          {k['mean_diff']:+.4f}  "
          f"CI=[{k['ci'][0]:+.4f},{k['ci'][1]:+.4f}]  t={k['t']:+.2f}  "
          f"{k['n_better']}/{k['n_cells']} cells better  -> NULL")

    a02 = json.loads((res / "a02_grpo_learning.json").read_text())
    g = a02["grpo_learning"]
    print(f"  GRPO total learning      {g['mean_delta']:+.4f}  "
          f"CI=[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}]  t={g['t']:+.2f}  (8192 rollouts)")
    lc = a02.get("lr_control", {}).get("1e-05_vs_1e-06")
    if lc:
        b = lc["bootstrap"]
        print(f"  LR positive control      {b['mean']:+.4f}  "
              f"CI=[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]  (block bootstrap, single cell)")

    a04p = res / "a4_verbosity_penalty.json"
    if a04p.is_file():
        a04 = json.loads(a04p.read_text())
        print(f"  Reward verbosity penalty {a04['verbosity_penalty']:+.4f}  "
              f"({100*a04['verbosity_penalty']/a04['mean_oracle']:.0f}% of achievable; "
              f"raw={a04['mean_raw']:.4f} oracle={a04['mean_oracle']:.4f})")

    a03 = json.loads((res / "a03_faultline.json").read_text())
    v = a03["accuracy"]["rca_validated"]
    print(f"  FaultLine (raw acc)      rca_validated {v['overall']:.3f} vs majority-class "
          f"{a03['label_balance']['overall_majority']:.3f}  -> does NOT beat a constant predictor")
    if "balanced_acc" in v:
        gap = a03.get("balanced_gap", {}).get("rca_validated_vs_uniform")
        extra = ""
        if gap:
            extra = (f"; vs uniform gap={gap['gap']:+.3f} "
                     f"CI=[{gap['ci'][0]:+.3f},{gap['ci'][1]:+.3f}]"
                     f"{' EXCLUDES ZERO' if gap['excludes_zero'] else ''}")
        print(f"  FaultLine (balanced acc) rca_validated {v['balanced_acc']:.3f} vs 0.200 chance "
              f"(a constant predictor scores exactly chance here){extra}")
    m = a03["paired"]["rca_validated_vs_rca_raw"]
    print(f"  Validation vs raw        {m['a_wins']}-{m['b_wins']} paired, chi2={m['chi2']:.2f} "
          f"-> {'SIGNIFICANT' if m['significant'] else 'ns'}")

    a05p = res / "a05_legibility.json"
    if a05p.is_file():
        a05 = json.loads(a05p.read_text())
        ta = a05.get("trajectory_ablation")
        if ta:
            print(f"  Trace ablation           full={ta['full_raw']:.3f} vs none={ta['none_raw']:.3f} "
                  f"-> the judge gains NOTHING from the execution trace")
        base = a05["conditions"].get("baseline", {})
        print(f"  Legibility degradation   no condition degrades localisation "
              f"(baseline balanced={base.get('balanced_acc', float('nan')):.3f} vs 0.200 chance)")

    g = a01.get("gepa_then_grpo_vs_dr_grpo")
    if g:
        print(f"  GEPA vs Dr. GRPO         {g['mean_diff']:+.4f}  "
              f"CI=[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}]  t={g['t']:+.2f}  "
              f"{g['n_better']}/{g['n_cells']} cells better  -> "
              f"{'SIGNIFICANT' if g['significant'] else 'ns'}")
        gl = a02.get("grpo_learning", {}).get("mean_delta")
        if gl:
            print(f"       ^ that gap is {g['mean_diff']/gl:.1f}x GRPO's ENTIRE training gain "
                  f"({gl:+.4f} over 8,192 rollouts)")


if __name__ == "__main__":
    main()
