# Analysis

Every number, table and figure in the paper is produced by these scripts from the run
data in `data/runs/` and the diagnostic outputs in `results/`. Standard library only,
except `make_figures.py` (matplotlib). Run from the repository root:

```bash
python analysis/run_all.py          # a01 to a07, then prints the headline numbers
python analysis/make_tables.py      # results/tables/percell.tex, promptcells.tex
python analysis/make_figures.py     # results/figures/ (PAPER_FIGS=1: results/figures/paper/)
```

## Scripts

| Script | What it computes |
|---|---|
| `lib_runs.py` | Run loader, run-name parser and stdlib statistics (paired t-test, McNemar, moving-block bootstrap). |
| `a01_variant_comparison.py` | Each variant against Dr. GRPO, paired by (task, budget, seed) cell: RCA, RCA without validation, transplanted and on-policy GEPA. |
| `a02_grpo_learning.py` | GRPO's total learning, the share of steps with zero group-relative advantage, and the learning-rate positive control. |
| `a03_faultline.py` | FaultLine localisation: label balance, majority-class baseline, balanced accuracy, paired McNemar. |
| `a05_legibility.py` | Judge accuracy under the trace-degradation conditions and with the trace removed. |
| `a06_onpolicy_gepa.py` | The budget-2048 prompt arms (default, docstring, transplanted GEPA, on-policy GEPA) with paired tests. |
| `a07_instruction_wording.py` | Flags each transplanted GEPA cell's modules as the unchanged seed docstring or GEPA-edited (docstrings read from the pipeline source with `ast`), and the gain over Dr. GRPO by flag. Drives Figure 1's hatching. |
| `a08_gpu_hours.py` | GPU-hours from the timestamps of the archived worker logs. Needs the raw archives under `archives/`, which are not distributed; its output is in `results/`. |
| `c1_make_label_sheet.py`, `c1_score_labels.py` | Build the blinded human label sheets in `data/human_labels/` and score the judge against the filled labels. |
| `extract_runs.py` | Unpacks the run JSON files from the raw archives into `data/runs/`, with one pinned source archive per run (see below). Kept for provenance; the extracted runs are already in the repository. |
| `make_tables.py`, `make_figures.py` | Appendix tables and figures. |

Two results have no script in this directory: `results/a4_verbosity_penalty.json` (raw
versus oracle-extraction score on 50 HotpotQA examples with default instructions, with
every row included) and `results/gepa_format_fix_diagnostic.json` (the controlled A/B
for the answer-extraction rule). Both were one-off measurements; the JSON files carry
the per-example data needed to check them.

## Two data-integrity rules

**Pinned provenance.** Archives from the cloud machines contained stale copies of runs
produced elsewhere, including contaminated originals of three RCA runs that were later
re-run under the same names. `extract_runs.py` names the single authoritative archive
for each run and fails if any pinned run resolves elsewhere. `data/provenance.json`
records the result.

**Cluster-robust statistics.** Steps within a run are autocorrelated, so treating them
as independent observations inflates *t*. Each run is reduced to one mean, one
difference is formed per (task, budget, seed) cell, and the test runs across cells.
Where only a single cell exists (the learning-rate control), a moving-block bootstrap
is used instead.
