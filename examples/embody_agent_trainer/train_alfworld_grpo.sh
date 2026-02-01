set -x
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL
ENGINE=${1:-vllm}

# ======================== GPU auto selection ========================
GPU_LIST=(4 5 6 7)  # <<<------  which GPUs to use, directly fill here
# Automatically concatenate CUDA_VISIBLE_DEVICES according to GPU_LIST
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Automatically detect the number of n_gpus_per_node
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

ROLLOUT_MODE="sync"
num_cpus_per_env_worker=0.2 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

MODEL=xxxx/Qwen3-4B-rft-alfworld-e1
MODEL_SHORT="${MODEL##*/}"
estimator="grpo"
project_name="alfworld"

# Check if any ray processes are running, exit if present, otherwise start ray
# if pgrep -u "$USER" "ray" > /dev/null; then
#     echo "==================== Detected existing Ray processes, exiting... ===================="
#     echo "==================== run "ray stop" to stop ray ===================="
#     exit 1
# fi
# PORT=$(( ( RANDOM % 10000 +1000) ))
PORT=1110
ray start --head --port $PORT --dashboard-port=1029

WANDB_API_KEY="xxxx" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    mkdir -p wandb/${project_name}/${experiment_name}
    SAVE_PATH=wandb/${project_name}/${experiment_name}
    export WANDB_DIR=${SAVE_PATH}
fi

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# for seed in 0 42 33
for seed in 0
do
    experiment_name="Seed${seed}_${MODEL_SHORT}_${estimator}"
    mkdir -p checkpoints/${project_name}/${experiment_name}

    python3 -m recipe.world_agent.main_world_agent\
        algorithm.adv_estimator=$estimator \
        data.train_files=$HOME/data/text/train.parquet \
        data.val_files=$HOME/data/text/test.parquet \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_update=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        use_invalid_action_penalty=True \
        invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=0.95 \
        algorithm.gigpo.step_advantage_w=1.0 \
        algorithm.gigpo.mode=$mode \
        env.env_name=alfworld/AlfredTWEnv \
        env.seed=$seed \
        env.max_steps=50 \
        env.rollout.n=$group_size \
        env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.rollout_data_dir=outputs/${project_name}/${experiment_name} \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=$NUM_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=5 \
        trainer.total_epochs=200 \
        trainer.max_actor_ckpt_to_keep=3 \
        trainer.val_before_train=False $@
done
