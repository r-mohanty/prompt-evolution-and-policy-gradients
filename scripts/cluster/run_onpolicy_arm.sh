#!/usr/bin/env bash
# On-policy GEPA arm, one pod per task (2026-09-03). Three phases, each idempotent
# enough to re-run after a failure:
#
#   bash scripts/cluster/run_onpolicy_arm.sh gepa  <task> <pod>   # serve Qwen3-8B on GPU 0, evolve 3 seeds, stop server
#   bash scripts/cluster/run_onpolicy_arm.sh train <task> <pod>   # queue 3 budget-2048 jobs, launch 3 workers
#   bash scripts/cluster/run_onpolicy_arm.sh status <task> <pod>  # queue state + tail of each log
#
# Run on a GPU host with at least three GPUs, inside the venv.
set -uo pipefail
PHASE=${1:?phase}; TASK=${2:?task}; POD=${3:?pod}
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"   # flashinfer's ninja subprocess needs .venv/bin on PATH
export PYTHONUNBUFFERED=1
# Hosts with driver 570 / CUDA 12.8 and a cu130 pip torch need this:
# NVIDIA's forward-compatibility package (apt-get install cuda-compat-13-0) provides a
# CUDA 13 libcuda for datacenter GPUs; verified 2026-09-03 with a real kernel launch.
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
LOGS=checkpoints/worker_logs; mkdir -p "$LOGS" data/gepa_onpolicy/instructions

if [ "$PHASE" = gepa ]; then
  echo "[$(date -u +%H:%M:%S)] starting vLLM server for Qwen/Qwen3-8B on GPU 0"
  CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.85 \
      --max-model-len 8192 --dtype bfloat16 --enforce-eager \
      > "$LOGS/vllm_serve.log" 2>&1 &
  SERVE_PID=$!
  echo $SERVE_PID > "$LOGS/vllm_serve.pid"
  for i in $(seq 1 90); do
    if curl -sf localhost:8000/v1/models > /dev/null; then break; fi
    if ! kill -0 $SERVE_PID 2>/dev/null; then echo "server died, see $LOGS/vllm_serve.log"; exit 1; fi
    sleep 10
  done
  curl -sf localhost:8000/v1/models > /dev/null || { echo "server not up after 15 min"; exit 1; }
  echo "[$(date -u +%H:%M:%S)] server up; launching GEPA for seeds 0 1 2"
  PIDS=()
  for s in 0 1 2; do
    nohup python scripts/prompt_opt/gepa_onpolicy_instructions.py --task "$TASK" --seed $s \
        > "$LOGS/gepa_onpolicy_${TASK}_s${s}.log" 2>&1 &
    PIDS+=($!)
  done
  RC=0
  for p in "${PIDS[@]}"; do wait $p || RC=1; done
  echo "[$(date -u +%H:%M:%S)] GEPA processes finished (rc=$RC); stopping server"
  kill $SERVE_PID 2>/dev/null; sleep 5; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 10
  nvidia-smi --query-gpu=index,memory.used --format=csv
  ls -la data/gepa_onpolicy/instructions/
  exit $RC
fi

if [ "$PHASE" = train ]; then
  for s in 0 1 2; do
    f="data/gepa_onpolicy/instructions/${TASK}_seed${s}.json"
    [ -f "$f" ] || { echo "missing $f; run the gepa phase first"; exit 1; }
  done
  Q="job_queues/$POD"; mkdir -p "$Q/pending" "$Q/claimed" "$Q/done" "$Q/failed"
  for s in 0 1 2; do
    printf '{"task": "%s", "budget": 2048, "seed": %d, "variant": "gepa_onpolicy_then_grpo", "track": "A1b"}\n' \
        "$TASK" $s > "$Q/pending/${TASK}_gepa_onpolicy_then_grpo_b2048_s${s}.json"
  done
  ls "$Q/pending"
  for g in 0 1 2; do
    nohup python scripts/cluster/gpu_worker.py --pod "$POD" --gpu $g > "$LOGS/worker_gpu$g.log" 2>&1 &
    echo "worker gpu $g pid $!"
  done
  exit 0
fi

if [ "$PHASE" = status ]; then
  Q="job_queues/$POD"
  for d in pending claimed done failed; do echo "$d: $(ls $Q/$d 2>/dev/null | tr '\n' ' ')"; done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
  for f in "$LOGS"/*.log; do echo "--- $f"; grep -E "step=|wrote|Traceback|Error|OOM|Killed|Done\." "$f" | tail -3; done
  exit 0
fi
echo "unknown phase $PHASE"; exit 2
