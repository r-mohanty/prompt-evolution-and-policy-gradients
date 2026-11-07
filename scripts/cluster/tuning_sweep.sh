#!/bin/bash
# Step 1.6 tuning sweep -- 10 runs, 5 pairs, one run per GPU per pair.
# Design (see decisions.md 2026-08-17): one-dimension-at-a-time around the
# lr=1e-6/group_size=8/kl_coef=0.04 anchor, not a full factorial grid.
#
# Run from anywhere; the script changes to the repository root:
#   nohup bash scripts/cluster/tuning_sweep.sh > sweep_driver.log 2>&1 & disown
#
# The trainer saves only at the end of a full run, so to resume after an
# interruption re-run the script from the top.
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mkdir -p sweep_logs

run() {
  gpu=$1; task=$2; lr=$3; gs=$4; kl=$5; tag=$6
  CUDA_VISIBLE_DEVICES=$gpu python -m reflectrl.training.multi_module_grpo_train \
    --task "$task" --budget 128 --seed 0 --variant dr_grpo \
    --learning-rate "$lr" --group-size "$gs" --kl-coef "$kl" \
    > "sweep_logs/${tag}.log" 2>&1
  echo "SWEEP_RUN_DONE tag=$tag exit=$?"
}

echo "SWEEP_START"

echo "=== Pair 1: anchor, cross-task ==="
run 0 hotpotqa 1e-6 8 0.04 hotpotqa_anchor &
run 1 aime     1e-6 8 0.04 aime_anchor &
wait
echo "PAIR_1_DONE"

echo "=== Pair 2: LR sweep low/mid ==="
run 0 hotpotqa 3e-7 8 0.04 hotpotqa_lr3e-7 &
run 1 hotpotqa 3e-6 8 0.04 hotpotqa_lr3e-6 &
wait
echo "PAIR_2_DONE"

echo "=== Pair 3: LR sweep high + group_size low ==="
run 0 hotpotqa 1e-5 8 0.04 hotpotqa_lr1e-5 &
run 1 hotpotqa 1e-6 4 0.04 hotpotqa_gs4 &
wait
echo "PAIR_3_DONE"

echo "=== Pair 4: group_size high + KL zero ==="
run 0 hotpotqa 1e-6 16 0.04 hotpotqa_gs16 &
run 1 hotpotqa 1e-6 8  0.0  hotpotqa_kl0 &
wait
echo "PAIR_4_DONE"

echo "=== Pair 5: KL high + cross-task at an alternate LR ==="
run 0 hotpotqa 1e-6 8 0.1  hotpotqa_kl0.1 &
run 1 aime     3e-6 8 0.04 aime_lr3e-6 &
wait
echo "PAIR_5_DONE"

echo "SWEEP_COMPLETE"
