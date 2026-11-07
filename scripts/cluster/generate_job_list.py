"""
Generates the full Phase 2 job list (Track A + RCA ablations) as one job per
line, and splits it into pod-specific queue files.

Scheduling: per-step costs were measured on H100s during the tuning sweep
(math is about 5x more expensive per step than HotpotQA). Jobs within
ISOLATE_THRESHOLD_FRACTION of the longest job get a dedicated GPU each; the
rest are split across SHARED_POD_SIZES with longest-processing-time-first
balancing, checked with a greedy list-scheduling simulation rather than
total-work/N averaging (which understates the makespan).

This builds the main grid (Dr. GRPO at five budgets; RCA, RCA without
validation and transplanted GEPA at four). The budget-2048 prompt arms
(docstring_then_grpo, gepa_onpolicy_then_grpo) and the learning-rate control
were queued separately; see scripts/cluster/run_onpolicy_arm.sh.

Run locally (no GPU needed -- just generates text files):
    python scripts/cluster/generate_job_list.py
"""
import heapq
import json
from pathlib import Path

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = WORKSHOP_ROOT / "job_queues"

TASKS = ["hotpotqa", "aime"]
SEEDS = [0, 1, 2]
TRACK_A_BUDGETS = [128, 256, 512, 2048, 8192]  # 2^7, 2^8, 2^9, 2^11, 2^13
RCA_BUDGETS = [128, 256, 512, 2048]  # RCA drops 2^13 per the plan's cut-list
RCA_VARIANTS = ["rca", "rca_no_validate", "gepa_then_grpo"]  # uniform_mask/dr_grpo is reused from Track A, not re-run

GROUP_SIZE = 8  # frozen 2026-08-17 by the tuning sweep, see decisions.md
OVERHEAD_S = 300  # vLLM cold start + model load + LoRA JIT warmup, measured 2026-08-17
STEP_S = {"hotpotqa": 15, "aime": 75}  # measured steady-state per-step cost at group_size=8
# A job qualifies for its own dedicated GPU (no shared queue) if it's within
# this fraction of the single longest job -- i.e. it's part of the same
# "dramatically more expensive than everything else" tier, not just
# incrementally pricier than the pack. Isolating these avoids the straggler
# risk of a very long job landing late in a shared pull-based queue.
ISOLATE_THRESHOLD_FRACTION = 0.8
# Pod sizes for the "everything else" (non-isolated) tier, 1-8 GPUs each per
# the provider's limit. Edit this list to match the pod sizes available at
# deploy time; the balancing adapts to however many pods are listed.
SHARED_POD_SIZES = [7, 1]


def job_seconds(job: dict) -> float:
    steps = job["budget"] // GROUP_SIZE
    return OVERHEAD_S + steps * STEP_S[job["task"]]


def build_all_jobs() -> list[dict]:
    jobs = []
    # Track A: plain Dr. GRPO baseline (--variant dr_grpo), all 5 budgets
    for task in TASKS:
        for budget in TRACK_A_BUDGETS:
            for seed in SEEDS:
                jobs.append({"task": task, "budget": budget, "seed": seed, "variant": "dr_grpo", "track": "A"})
    # RCA ablations: 3 variants, 4 budgets (no 8192)
    for task in TASKS:
        for budget in RCA_BUDGETS:
            for seed in SEEDS:
                for variant in RCA_VARIANTS:
                    jobs.append({"task": task, "budget": budget, "seed": seed, "variant": variant, "track": "RCA"})
    return jobs


def job_to_command(job: dict) -> str:
    return (
        f"python -m reflectrl.training.multi_module_grpo_train "
        f"--task {job['task']} --budget {job['budget']} --seed {job['seed']} --variant {job['variant']}"
    )


def simulate_makespan(n_gpus: int, job_list: list[dict]) -> float:
    """Realistic greedy list-scheduling simulation (longest job first,
    always assigned to whichever GPU is currently least loaded) -- NOT
    naive total-work/N averaging, which can be misleadingly optimistic:
    tested directly during this revision, 6 GPUs for the 99 non-isolated
    jobs looked fine by averaging (21.9hr, just over the 21.42hr floor by
    eye) but simulation showed the real greedy makespan (21.97hr) actually
    exceeded it. Returns hours."""
    heap = [0.0] * n_gpus
    heapq.heapify(heap)
    for j in sorted(job_list, key=job_seconds, reverse=True):
        least_loaded = heapq.heappop(heap)
        heapq.heappush(heap, least_loaded + job_seconds(j))
    return max(heap) / 3600


def plan_pods(all_jobs: list[dict]) -> list[tuple[str, int, list[dict]]]:
    """Isolate the dramatically-more-expensive tier onto their own
    dedicated GPUs (one job each, no queue -- they set the wall-clock floor
    no topology can beat), then LPT-balance everything else across however
    many pods SHARED_POD_SIZES lists (was always exactly one pod sized at
    the max allowed until 8-GPU pods became unavailable 2026-08-18; now
    generalized to an arbitrary list so it adapts to whatever's actually
    available). Balances by (assigned_seconds / that pod's own GPU count),
    same principle as v1's whole-grid balancing, just scoped to this tier
    and GPU-count-aware so a 1-GPU pod doesn't get the same job COUNT as a
    7-GPU pod."""
    jobs_sorted = sorted(all_jobs, key=job_seconds, reverse=True)
    floor_hr = job_seconds(jobs_sorted[0]) / 3600
    max_pod_size = max(SHARED_POD_SIZES)

    isolated = [j for j in jobs_sorted if job_seconds(j) / 3600 >= floor_hr * ISOLATE_THRESHOLD_FRACTION]
    remaining = jobs_sorted[len(isolated):]
    assert len(isolated) <= max_pod_size, (
        f"{len(isolated)} jobs qualify for dedicated GPUs but the largest available pod only has "
        f"{max_pod_size} -- widen ISOLATE_THRESHOLD_FRACTION or split the isolated tier across 2 pods"
    )

    plan = [("pod_isolated_big", len(isolated), isolated)]

    shared_names = [f"pod_shared_{i}" for i in range(len(SHARED_POD_SIZES))]
    shared_assigned = {name: [] for name in shared_names}
    shared_load = {name: 0.0 for name in shared_names}
    shared_gpus = dict(zip(shared_names, SHARED_POD_SIZES))
    for j in remaining:  # already sorted longest-first
        target = min(shared_assigned, key=lambda name: shared_load[name] / shared_gpus[name])
        shared_assigned[target].append(j)
        shared_load[target] += job_seconds(j)
    for name in shared_names:
        plan.append((name, shared_gpus[name], shared_assigned[name]))

    return plan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_jobs = build_all_jobs()
    assert len(all_jobs) == 102

    plan = plan_pods(all_jobs)

    total_gpus = 0
    for name, gpu_count, jobs in plan:
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
        total_gpu_hr = sum(job_seconds(j) for j in jobs) / 3600
        makespan_hr = simulate_makespan(gpu_count, jobs)
        total_gpus += gpu_count
        print(f"{name} ({gpu_count} GPUs): {len(jobs)} jobs, {total_gpu_hr:.1f} GPU-hr, "
              f"simulated makespan ~{makespan_hr:.1f}hr -> {path}")

    print(f"\nTotal GPUs requested: {total_gpus}")

    # Also write the full list + human-readable commands, for reference.
    (OUT_DIR / "all_jobs.jsonl").write_text("\n".join(json.dumps(j) for j in all_jobs))
    (OUT_DIR / "all_jobs_commands.txt").write_text("\n".join(job_to_command(j) for j in all_jobs))
    total_gpu_hr = sum(job_seconds(j) for j in all_jobs) / 3600
    print(f"Total: {len(all_jobs)} jobs, {total_gpu_hr:.1f} GPU-hr")


if __name__ == "__main__":
    main()
