#!/bin/bash

# Math Agent API Evaluation Runner
# Usage: bash eval_math_agent.sh [model_name] [data_file]

# Set environment variables (replace with your API keys)
# export OPENAI_API_KEY="your-openai-api-key"
# export DEEPSEEK_API_KEY="your-deepseek-api-key"
# export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Default values
MODEL_NAME=${1:-gpt-4o-mini}
DATA_FILE=${2:-${HOME}/ARLArena/datasets/simplelr_math_35/test.parquet}
VAL_BATCH_SIZE=${3:-8}
NUM_SAMPLES=${4:-4}

# Set resource limits
ulimit -n 65536
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Print configuration
echo "=========================================="
echo "Math Agent API Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Data: $DATA_FILE"
echo "Batch Size: $VAL_BATCH_SIZE"
echo "Num Samples (n): $NUM_SAMPLES"
echo "=========================================="
echo ""

# Run evaluation
python3 -m recipe.math_agent.eval.eval_math_agent_api \
    model_config.model_name=$MODEL_NAME \
    data.val_files=$DATA_FILE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_SAMPLES \
    agent.max_turns=5 \
    agent.sandbox_run_timeout=3.0 \
    reward_model.reward_manager=math \
    evaluation.output_dir=outputs_math_agent/eval_results

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: outputs_math_agent/eval_results"
echo "=========================================="
