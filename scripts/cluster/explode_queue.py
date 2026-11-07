"""
Explodes a pod's *.jsonl queue file (from generate_job_list.py) into
individual per-job JSON files under job_queues/<pod>/pending/, which is
the shape gpu_worker.py's atomic-claim mechanism expects. Run once per pod,
on that pod, after rsync-ing the repo and before launching any gpu_worker.py
processes.

Run (on the pod, after rsync):
    python scripts/cluster/explode_queue.py --pod pod_shared_0
"""
import argparse
import json
from pathlib import Path

WORKSHOP_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    args = parser.parse_args()

    base = WORKSHOP_ROOT / "job_queues" / args.pod
    src = base.parent / f"{args.pod}.jsonl"
    pending_dir = base / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for i, line in enumerate(src.read_text().splitlines()):
        if not line.strip():
            continue
        job = json.loads(line)
        job_id = f"{job['task']}_{job['variant']}_b{job['budget']}_s{job['seed']}_{i:03d}"
        (pending_dir / f"{job_id}.json").write_text(json.dumps(job))
        n += 1
    print(f"Exploded {n} jobs from {src} into {pending_dir}")


if __name__ == "__main__":
    main()
