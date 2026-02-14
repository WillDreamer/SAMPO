#!/usr/bin/env bash
# One-stop runner for world-agent API evaluation.
# Usage (override vars if needed):
#   MODEL="gpt-4o-mini" TRAIN_DATA=~/data/text/train.parquet VAL_DATA=~/data/text/test.parquet bash recipe/world_agent/eval_world_agent.sh

set -euo pipefail

# Raise the open file limit for vLLM-style I/O
ulimit -n 131072 || true
# Avoid MKL thread conflicts
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL


MODEL="${MODEL:-gpt-4o}"                    # e.g., gpt-4o-mini or gpt-4o
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen2.5-7B-Instruct}"  # Tokenizer for fallback decode 注意tokenizer其实是改过了，不过对于调api来讲，这个并不重要，所以姑且用这个规范一下写法
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/text/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/text/test.parquet}"
BATCH="${BATCH:-64}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESP_LEN="${MAX_RESP_LEN:-500}"
TEMP="${TEMP:-0.6}"
NUM_CPUS_PER_ENV="${NUM_CPUS_PER_ENV:-0.1}"
SEED="${SEED:-0}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${LOG_DIR}"

echo "[INFO] Using MODEL=${MODEL}"
echo "[INFO] Tokenizer=${TOKENIZER_PATH}"
echo "[INFO] Train data=${TRAIN_DATA}"
echo "[INFO] Val data=${VAL_DATA}"
echo "[INFO] Batch size=${BATCH}, Max prompt len=${MAX_PROMPT_LEN}, Max resp len=${MAX_RESP_LEN}"
echo "[INFO] Seed=${SEED}, CPUs per env=${NUM_CPUS_PER_ENV}"
echo "[INFO] Logs -> ${LOG_DIR}"

#这个确实需要覆盖一下: model_config

python3 -m recipe.world_agent.eval_world_agent_api \
  model_config.model_name="${MODEL}" \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.train_batch_size="${BATCH}" \
  data.val_batch_size="${BATCH}" \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESP_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${TOKENIZER_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.99 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature="${TEMP}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed="${SEED}" \
  env.max_steps=50 \
  env.rollout.n=1 \
  env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV}" \
  2>&1 | tee "${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
