#!/usr/bin/env python
"""FaultLine: can any credit-assignment method localise the faulty module?
Run:  python analysis/a03_faultline.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_runs import ROOT, mcnemar  # noqa: E402

RES = ROOT / "results"


def main() -> None:
    fl = json.loads((RES / "faultline_credit_scoring.json").read_text())
    rca = json.loads((RES / "rca_faultline_eval.json").read_text())
    rows = [r for r in rca["rows"] if not r.get("error")]

    # `validated_correct` is CONTAMINATED and must not be used directly.
    # When counterfactual validation REJECTS the reflection's pick, credit falls
    # back to a uniform distribution and max() on an exactly-tied dict returns
    # the FIRST module in pipeline order (plan for AIME, generate_query for
    # HotpotQA). That is an abstention, not a prediction -- but it scores as
    # correct whenever the gold module happens to be first. Verified 2026-09-01:
    # 9 of 45 "correct" rows are this artifact, and they are EXACTLY the 9 rows
    # that produced the apparent "validation beats raw reflection 9-0" result.
    # `plan` recall was 7/7 with ZERO real hits.
    #
    # Honest accounting: a rejected row is an ABSTENTION (scored incorrect).
    methods: dict[str, dict[int, bool]] = {
        "rca_validated": {r["instance_idx"]: bool(r["validated_correct"]) and bool(r.get("was_validated"))
                          for r in rows},
        "rca_validated_RAWFIELD_CONTAMINATED": {r["instance_idx"]: bool(r["validated_correct"])
                                                for r in rows},
        "rca_raw": {r["instance_idx"]: bool(r["raw_correct"]) for r in rows},
    }
    for name, per in fl.get("per_instance", {}).items():
        methods[name] = {r["instance_idx"]: bool(r["correct"]) for r in per if r.get("predicted")}
    task_of = {r["instance_idx"]: r["task"] for r in rows}
    gold_of = {r["instance_idx"]: r["gold"] for r in rows}

    out: dict = {}

    print("=" * 92)
    print("A03.1 -- GOLD LABEL BALANCE (the reference that accuracy must beat)")
    print("=" * 92)
    per_task_major = {}
    for task in sorted(set(task_of.values())):
        c = Counter(g for i, g in gold_of.items() if task_of[i] == task)
        n = sum(c.values())
        maj = max(c.values()) / n
        per_task_major[task] = maj
        print(f"  {task:9s} n={n:3d}  {dict(c)}")
        print(f"            majority-class baseline = {maj:.3f}")
    allc = Counter(gold_of.values())
    overall_major = max(allc.values()) / sum(allc.values())
    print(f"  OVERALL   majority-class baseline = {overall_major:.3f}")
    out["label_balance"] = {"per_task_majority": per_task_major,
                            "overall_majority": overall_major,
                            "counts": {k: v for k, v in allc.items()}}

    print()
    print("=" * 92)
    print("A03.2 -- ACCURACY vs MAJORITY-CLASS BASELINE")
    print("=" * 92)
    out["accuracy"] = {}
    for name, m in sorted(methods.items()):
        overall = sum(m.values()) / len(m) if m else float("nan")
        line = f"  {name:22s} overall={overall:.3f} (n={len(m):2d})"
        rec = {"overall": overall, "n": len(m), "per_task": {}}
        for task, maj in per_task_major.items():
            sub = [c for i, c in m.items() if task_of.get(i) == task]
            if not sub:
                continue
            acc = sum(sub) / len(sub)
            rec["per_task"][task] = {"acc": acc, "n": len(sub), "majority": maj,
                                     "beats_majority": acc > maj}
            line += f"   {task}={acc:.3f}{'*' if acc > maj else '!'}"
        print(line)
        out["accuracy"][name] = rec
    print(f"\n  (* beats that task's majority-class baseline; ! loses to it)")

    print()
    print("=" * 92)
    print("A03.2b -- BALANCED ACCURACY (mean per-class recall)")
    print("=" * 92)
    print("  Raw accuracy is dominated by the majority gold class. Balanced accuracy weights")
    print("  every gold module equally, so a constant predictor scores 1/k regardless of skew.")
    print("  The skew is STRUCTURAL, not a sampling artifact: an instance is kept only if faults")
    print("  in OTHER modules leave the example working, and a corrupted DOWNSTREAM module")
    print("  (generate_answer) almost always breaks it, so upstream-fault instances are")
    print("  intrinsically rare. Forcing balance would mean relaxing that discriminative check,")
    print("  which would weaken the benchmark rather than improve it.")
    print()
    classes = sorted(set(gold_of.values()))
    for name, m in sorted(methods.items()):
        recalls, detail = [], []
        for cls in classes:
            idxs = [i for i in m if gold_of.get(i) == cls]
            if not idxs:
                continue
            rec = sum(m[i] for i in idxs) / len(idxs)
            recalls.append(rec)
            detail.append(f"{cls}={rec:.2f}(n={len(idxs)})")
        if recalls:
            bal = sum(recalls) / len(recalls)
            chance = 1.0 / len(recalls)
            print(f"  {name:22s} balanced_acc={bal:.3f}  chance={chance:.3f}  "
                  f"{'ABOVE chance' if bal > chance else 'AT/BELOW chance'}")
            print(f"       {'  '.join(detail)}")
            out["accuracy"][name]["balanced_acc"] = bal
            out["accuracy"][name]["n_classes_covered"] = len(recalls)

    print()
    print("=" * 92)
    print("A03.2c -- BALANCED-ACCURACY GAP, stratified bootstrap (resample within gold class)")
    print("=" * 92)
    print("  A majority-class predictor scores exactly 1/k balanced accuracy (chance), so this")
    print("  is the comparison that is NOT confounded by the structural label skew.")
    print()
    import random

    def balanced(m, idx_by_class):
        recs = []
        for cls, idxs in idx_by_class.items():
            idxs = [i for i in idxs if i in m]
            if idxs:
                recs.append(sum(m[i] for i in idxs) / len(idxs))
        return sum(recs) / len(recs) if recs else float("nan")

    idx_by_class = {}
    for i, g in gold_of.items():
        idx_by_class.setdefault(g, []).append(i)

    for a, b in [("rca_validated", "uniform"), ("rca_validated", "rca_raw"),
                 ("rca_validated", "flat_reflection")]:
        if a not in methods or b not in methods:
            continue
        rng = random.Random(2024)
        obs = balanced(methods[a], idx_by_class) - balanced(methods[b], idx_by_class)
        boots = []
        for _ in range(5000):
            res = {c: [rng.choice(v) for _ in v] for c, v in idx_by_class.items()}
            boots.append(balanced(methods[a], res) - balanced(methods[b], res))
        boots.sort()
        lo, hi = boots[125], boots[4874]
        excl = lo > 0 or hi < 0
        print(f"  {a} - {b:18s} gap={obs:+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  "
              f"{'EXCLUDES ZERO' if excl else 'includes zero'}")
        out.setdefault("balanced_gap", {})[f"{a}_vs_{b}"] = {
            "gap": obs, "ci": [lo, hi], "excludes_zero": excl}



    print()
    print("=" * 92)
    print("A03.3 -- PAIRED COMPARISONS (McNemar; same instances, so pairing is valid)")
    print("=" * 92)
    pairs = [("rca_validated", "rca_raw"), ("rca_validated", "uniform"),
             ("rca_validated", "flat_reflection"), ("rca_raw", "uniform")]
    out["paired"] = {}
    for a, b in pairs:
        if a not in methods or b not in methods:
            continue
        r = mcnemar(methods[a], methods[b])
        print(f"  {a:16s} vs {b:18s} n={r['n_pairs']:3d}  "
              f"{a[:12]}_wins={r['a_wins']:2d}  {b[:12]}_wins={r['b_wins']:2d}  "
              f"chi2={r['chi2']:5.2f}  {'SIGNIFICANT' if r['significant'] else 'not significant'}")
        out["paired"][f"{a}_vs_{b}"] = r

    print()
    print("=" * 92)
    print("A03.4 -- RCA AS PRECISION/COVERAGE, not accuracy (the honest characterisation)")
    print("=" * 92)
    acc_rows = [r for r in rows if r.get("was_validated")]
    rej_rows = [r for r in rows if not r.get("was_validated")]
    n = len(rows)
    prec = (sum(r["validated_correct"] for r in acc_rows) / len(acc_rows)) if acc_rows else float("nan")
    print(f"  validation ACCEPTS on {len(acc_rows)}/{n} rows ({100*len(acc_rows)/n:.1f}% coverage)")
    print(f"  precision when it accepts: {prec:.3f}")
    print(f"  abstains on {len(rej_rows)}/{n} rows ({100*len(rej_rows)/n:.1f}%)")
    print()
    print("  CAVEAT -- both numbers are FORCED by FaultLine's single-fault construction:")
    print("  patching module X restores success IFF X is the true fault, so validation accepts")
    print("  exactly when reflection was already right. Precision is 1.000 by construction, and")
    print("  accept-rate necessarily equals raw reflection accuracy. FaultLine as built therefore")
    print("  CANNOT distinguish 'validation works' from 'validation is tautological here'.")
    out["precision_coverage"] = {"coverage": len(acc_rows)/n, "precision": prec,
                                 "abstain_rate": len(rej_rows)/n,
                                 "caveat": "both forced by the single-fault construction"}

    dest = RES / "a03_faultline.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
