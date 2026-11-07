"""Shared loader + statistics for the reflection-vs-RL analysis.

Standard library only, deliberately: every number in the paper is
reproducible with a bare `python` and no scientific stack. numpy/scipy are NOT
required.

Data source: data/runs/<run_name>/{training_log,training_summary}.json,
produced by analysis/extract_runs.py from the pod backup tarballs.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"

# rca_no_validate MUST precede rca in the alternation or it parses as `rca`.
RUN_RE = re.compile(
    r"^(?P<prefix>UNPATCHED_)?"
    r"(?P<task>hotpotqa|aime)_"
    r"(?P<variant>dr_grpo|rca_no_validate|rca|gepa_onpolicy_then_grpo|gepa_then_grpo|docstring_then_grpo)_"
    r"(?P<budget>\d+)_seed(?P<seed>\d+)"
    r"(?:_lr(?P<lr>[\d.e+-]+))?(?:_gs(?P<gs>\d+))?(?:_kl(?P<kl>[\d.]+))?$"
)


@dataclass
class Run:
    name: str
    task: str
    variant: str
    budget: int
    seed: int
    lr: float
    gs: int
    kl: float
    rewards: list[float] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    unpatched: bool = False

    @property
    def cell(self) -> tuple[str, int, int]:
        """The (task, budget, seed) triple that pairs variants for comparison."""
        return (self.task, self.budget, self.seed)

    @property
    def mean_reward(self) -> float:
        return mean(self.rewards) if self.rewards else float("nan")

    @property
    def zero_reward_frac(self) -> float:
        if not self.rewards:
            return float("nan")
        return sum(1 for r in self.rewards if r == 0.0) / len(self.rewards)


def load_runs(runs_dir: Path | None = None, include_unpatched: bool = False) -> list[Run]:
    d = runs_dir or RUNS_DIR
    out: list[Run] = []
    if not d.is_dir():
        raise SystemExit(f"no run data at {d} -- run `python analysis/extract_runs.py` first")
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            continue
        if m.group("prefix") and not include_unpatched:
            continue
        log = sub / "training_log.json"
        rewards: list[float] = []
        if log.is_file():
            try:
                rewards = [float(e["mean_reward"]) for e in json.loads(log.read_text())
                           if "mean_reward" in e]
            except Exception:  # noqa: BLE001
                rewards = []
        summ = sub / "training_summary.json"
        summary = json.loads(summ.read_text()) if summ.is_file() else {}
        out.append(Run(
            name=sub.name, task=m.group("task"), variant=m.group("variant"),
            budget=int(m.group("budget")), seed=int(m.group("seed")),
            lr=float(m.group("lr") or 1e-6), gs=int(m.group("gs") or 8),
            kl=float(m.group("kl") or 0.04), rewards=rewards, summary=summary,
            unpatched=bool(m.group("prefix")),
        ))
    return out


# --------------------------------------------------------------------------
# statistics (stdlib only)
# --------------------------------------------------------------------------

def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_critical_95(df: int) -> float:
    """Two-sided 95% critical value. Table for small df, normal limit beyond."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
             14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
             20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
             26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021,
             60: 2.000, 120: 1.980}
    if df in table:
        return table[df]
    for k in sorted(table):
        if df < k:
            return table[k]
    return 1.960


@dataclass
class PairedResult:
    n: int
    mean_diff: float
    sd: float
    se: float
    t: float
    ci_lo: float
    ci_hi: float
    n_positive: int

    @property
    def significant(self) -> bool:
        return abs(self.t) > t_critical_95(max(self.n - 1, 1))

    def line(self, label: str = "") -> str:
        verdict = "SIGNIFICANT" if self.significant else "null"
        return (f"{label:<38s} n={self.n:3d}  mean={self.mean_diff:+.4f}  SE={self.se:.4f}  "
                f"t={self.t:+.2f}  95%CI=[{self.ci_lo:+.4f},{self.ci_hi:+.4f}]  "
                f"better={self.n_positive}/{self.n}  {verdict}")


def paired_test(diffs: list[float]) -> PairedResult:
    """One-sample t-test on paired differences (the cluster-robust design:
    one difference per (task,budget,seed) cell, t across cells)."""
    n = len(diffs)
    if n < 2:
        return PairedResult(n, float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), 0)
    m, sd = mean(diffs), stdev(diffs)
    se = sd / math.sqrt(n)
    t = m / se if se else float("nan")
    crit = t_critical_95(n - 1)
    return PairedResult(n, m, sd, se, t, m - crit * se, m + crit * se,
                        sum(1 for d in diffs if d > 0))


def mcnemar(a: dict, b: dict) -> dict:
    """Paired binary comparison over shared keys. Returns discordant counts and
    the continuity-corrected chi-square (df=1, critical 3.841)."""
    keys = [k for k in a if k in b]
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    n_disc = a_only + b_only
    chi2 = ((abs(a_only - b_only) - 1) ** 2 / n_disc) if n_disc else 0.0
    return {"n_pairs": len(keys), "a_wins": a_only, "b_wins": b_only,
            "n_discordant": n_disc, "chi2": chi2, "significant": chi2 > 3.841}


def block_bootstrap_mean(diffs: list[float], block: int = 20, n_boot: int = 10000,
                         seed: int = 12345) -> dict:
    """Moving-block bootstrap CI for the mean of an AUTOCORRELATED series.

    Sequential training steps are strongly autocorrelated, so an ordinary
    step-paired t-test massively understates the SE and inflates t. Resampling
    contiguous blocks preserves within-block dependence and gives an honest
    interval. Used for the learning-rate positive control, which exists for only
    ONE (task,budget,seed) cell and therefore cannot get the cluster-robust
    treatment applied to the main grid.
    """
    import random
    n = len(diffs)
    if n < block * 2:
        block = max(1, n // 4)
    rng = random.Random(seed)
    n_blocks = -(-n // block)
    starts = max(1, n - block + 1)
    means = []
    for _ in range(n_boot):
        acc, cnt = 0.0, 0
        for _ in range(n_blocks):
            st = rng.randrange(starts)
            for v in diffs[st:st + block]:
                acc += v
                cnt += 1
        means.append(acc / cnt)
    means.sort()
    point = mean(diffs)
    return {
        "mean": point, "block": block, "n_boot": n_boot,
        "se": stdev(means),
        "ci_lo": means[int(0.025 * n_boot)], "ci_hi": means[int(0.975 * n_boot)],
        "excludes_zero": means[int(0.025 * n_boot)] > 0 or means[int(0.975 * n_boot)] < 0,
    }
