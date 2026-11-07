"""
Per-GPU job worker. Run once per GPU on a pod (each
invocation pinned to one CUDA device via --gpu), and each instance claims
jobs from the pod's shared queue directory until it's empty.

Coordination is a directory-based atomic claim (os.rename is atomic on a
POSIX filesystem, which every RunPod pod is) -- no file lock needed:

    job_queues/<pod>/pending/<job_id>.json   -- unclaimed
    job_queues/<pod>/claimed/<job_id>.json   -- a worker is running it
    job_queues/<pod>/done/<job_id>.json      -- finished successfully
    job_queues/<pod>/failed/<job_id>.json    -- crashed; NOT auto-retried
                                                 (requeue by moving the file
                                                 back to pending/)

Setup (once per pod, after explode_queue.py has run):
    python scripts/cluster/gpu_worker.py --pod pod_shared_0 --gpu 0 &
    python scripts/cluster/gpu_worker.py --pod pod_shared_0 --gpu 1 &
    ... (one per GPU on this pod)

If a worker process itself dies (not the training subprocess -- the
training subprocess failing is caught and moves the job to failed/), its
currently-claimed job stays in claimed/ forever and needs manual requeue:
    mv job_queues/<pod>/claimed/*.json job_queues/<pod>/pending/
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]


def claim_next_job(pending_dir: Path, claimed_dir: Path) -> Path | None:
    for name in sorted(os.listdir(pending_dir)):
        src = pending_dir / name
        dst = claimed_dir / name
        try:
            os.rename(src, dst)  # atomic on POSIX -- this IS the claim
            return dst
        except FileNotFoundError:
            continue  # another worker claimed it first, try the next name
    return None


def run_job(job_path: Path, gpu_index: int) -> bool:
    job = json.loads(job_path.read_text())
    cmd = [
        sys.executable, "-m", "reflectrl.training.multi_module_grpo_train",
        "--task", job["task"], "--budget", str(job["budget"]),
        "--seed", str(job["seed"]), "--variant", job["variant"],
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # Without this, Python fully-buffers stdout when it's not a TTY (i.e.
    # always, here -- redirected to log_path below), so short "step=..."
    # print() lines can sit unflushed for a long time (found live 2026-08-18:
    # a real 21.4hr AIME budget=8192 job showed zero step= output for 20+
    # minutes despite genuinely training the whole time, confirmed via GPU
    # power draw/utilization -- not a hang, just buffering. On a job this
    # long, that's poor observability for most of the run, not a cosmetic
    # nit.) PYTHONUNBUFFERED=1 forces line-by-line flushing.
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[gpu {gpu_index}] START {job_path.name}: {' '.join(cmd)}")
    log_path = WORKSHOP_ROOT / "checkpoints" / "worker_logs" / f"{job_path.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # start_new_session=True (own process group) is required for the cleanup
    # below, not just tidiness -- found live 2026-08-18: a training script
    # that dies via an uncaught Python exception (e.g. the reflect_credit bug
    # this same debugging session fixed) does NOT reliably take its own
    # multiprocessing-spawned vLLM EngineCore child down with it. subprocess.run
    # only waits for the DIRECT child, so an orphaned EngineCore can keep
    # holding ~30GB+ of GPU memory after this call returns -- then poisons
    # whichever job THIS SAME GPU index picks up next with a CUDA OOM,
    # regardless of that next job's own variant (confirmed: a gepa_then_grpo
    # job failed this way, even though gepa_then_grpo never touches the code
    # that actually crashed). Killing the whole process group after every job
    # -- success or failure -- guarantees no such orphan survives to poison
    # the next one.
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=str(WORKSHOP_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        returncode = proc.wait()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already fully exited, nothing left to clean up -- the common case
        else:
            time.sleep(1)  # let the OS actually release GPU memory before the next job claims this GPU
    ok = returncode == 0
    print(f"[gpu {gpu_index}] {'DONE' if ok else 'FAILED'} {job_path.name} (exit {returncode}, log: {log_path})")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True, help="job_queues subdirectory name, e.g. pod_a_big")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index this worker owns exclusively")
    args = parser.parse_args()

    base = WORKSHOP_ROOT / "job_queues" / args.pod
    pending_dir, claimed_dir, done_dir, failed_dir = (base / d for d in ("pending", "claimed", "done", "failed"))
    for d in (pending_dir, claimed_dir, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_done = 0
    while True:
        job_path = claim_next_job(pending_dir, claimed_dir)
        if job_path is None:
            print(f"[gpu {args.gpu}] queue empty, {n_done} jobs completed by this worker. Exiting.")
            break
        ok = run_job(job_path, args.gpu)
        job_path.rename((done_dir if ok else failed_dir) / job_path.name)
        n_done += 1
        time.sleep(1)  # brief pause so a crash-loop on a bad job doesn't spin the CPU


if __name__ == "__main__":
    main()
