# How to Add AEPO and GSPO to Your Task

This guide explains how to integrate AEPO (Adaptive Entropy Policy Optimization) and GSPO (Group Sequence Policy Optimization) into your task's ray trainer. The implementation is based on the `simpletir` task implementation in `/home/xw27/agent/ARLArena/recipe/simpletir/simpletir_ray_trainer.py`.

## Overview

- **AEPO**: Uses GRPO for advantage computation, but applies adaptive entropy balancing in the loss function (`compute_loss_aepo` in verl)
- **GSPO**: Uses GRPO for advantage computation, but applies sequence-level importance ratio in the loss function (`compute_loss_gspo` in verl)
- The loss computation logic is already implemented in `verl` - you just need to configure your trainer to use it

## Implementation Steps

### Step 1: Add AEPO and GSPO to AdvantageEstimator Enum

In your ray trainer file (e.g., `recipe/your_task/ppo/ray_trainer.py`), add AEPO and GSPO to the `AdvantageEstimator` enum:

```python
class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'
    AEPO = "aepo"  # Adaptive Entropy Policy Optimization
    GSPO = "gspo"  # Group Sequence Policy Optimization
```

### Step 2: Update compute_advantage Function

In your `compute_advantage` function, modify the GRPO branch to also handle AEPO and GSPO:

```python
elif adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.AEPO, AdvantageEstimator.GSPO):
    # GRPO, AEPO, and GSPO use the same advantage computation
    # AEPO's entropy balancing is handled in the loss function (compute_loss_aepo)
    # GSPO's sequence-level importance ratio is handled in the loss function (compute_loss_gspo)
    grpo_calculation_mask = data.batch["response_mask"]
    if multi_turn:
        # If multi-turn, replace the mask with the relevant part of loss_mask
        response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
        grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
    # Call compute_grpo_outcome_advantage with parameters matching its definition
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],
        traj_index=data.non_tensor_batch['traj_uid'],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

**Note**: AEPO and GSPO use the same advantage computation as GRPO. The difference is in the loss function, which is handled automatically by verl when `loss_mode` is set correctly.

### Step 3: Add Logic in __init__ Method (Math)

In your trainer's `__init__` method, add logic to automatically set `loss_mode` when AEPO or GSPO is used:

```python
def __init__(
    self,
    config,
    tokenizer,
    role_worker_mapping: dict[Role, WorkerType],
    resource_pool_manager: ResourcePoolManager,
    ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
    processor=None,
    reward_fn=None,
    val_reward_fn=None,
):
    # ... existing initialization code ...
    
    # Convert config value to enum for consistent comparison
    adv_estimator = AdvantageEstimator(self.config.algorithm.adv_estimator)
    
    # Automatically set loss_mode to "aepo" when using AEPO advantage estimator
    if adv_estimator == AdvantageEstimator.AEPO:
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.policy_loss.loss_mode = "aepo"
    
    # Automatically set loss_mode to "gspo" when using GSPO advantage estimator
    if adv_estimator == AdvantageEstimator.GSPO:
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.policy_loss.loss_mode = "gspo"
    
    # Update use_critic based on adv_estimator
    if adv_estimator == AdvantageEstimator.GAE:
        self.use_critic = True
    elif adv_estimator in [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.REMAX,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.GiGPO,
        AdvantageEstimator.AEPO,
        AdvantageEstimator.GSPO,
    ]:
        self.use_critic = False
    else:
        raise NotImplementedError(
            f"Unsupported advantage estimator: {self.config.algorithm.adv_estimator}"
        )
    
    # ... rest of initialization ...
```



**Key Points**:
- When `adv_estimator` is set to `"aepo"`, the code automatically sets `loss_mode = "aepo"` in the config
- When `adv_estimator` is set to `"gspo"`, the code automatically sets `loss_mode = "gspo"` in the config
- Both AEPO and GSPO don't require a critic (like GRPO), so `use_critic = False`

### Step 4: Update Configuration in Bash File

In your training bash script, add the advantage estimator as a command-line argument when calling the Python training script:

For AEPO:
```bash
python -m recipe.your_task.main_your_task \
    --config-name $CONFIG_NAME \
    algorithm.adv_estimator=aepo \
    # ... other config parameters ...
```

For GSPO:
```bash
python -m recipe.your_task.main_your_task \
    --config-name $CONFIG_NAME \
    algorithm.adv_estimator=gspo \
    # ... other config parameters ...
```

**Example from `train_aepo.sh`:**
```bash
PYTHONUNBUFFERED=1 python -m recipe.simpletir.main_simpletir \
    --config-name $CONFIG_NAME \
    algorithm.adv_estimator=aepo \
    data.train_files=$TRAIN_FILES \
    # ... rest of config ...
```

**Example from `train_gspo.sh`:**
```bash
PYTHONUNBUFFERED=1 python -m recipe.simpletir.main_simpletir \
    --config-name $CONFIG_NAME \
    algorithm.adv_estimator=gspo \
    data.train_files=$TRAIN_FILES \
    # ... rest of config ...
```

The `loss_mode` will be automatically set by the trainer's `__init__` method, so you don't need to manually set it in the config or bash file.

## How It Works

1. **Advantage Computation**: Both AEPO and GSPO use GRPO's advantage computation method (`compute_grpo_outcome_advantage`). This is handled in the `compute_advantage` function.

2. **Loss Computation**: The actual loss computation is implemented in `verl`:
   - For AEPO: `compute_loss_aepo` in `verl/trainer/ppo/core_algos.py`
   - For GSPO: `compute_loss_gspo` in `verl/trainer/ppo/core_algos.py`
   
   These functions are automatically called when `loss_mode` is set to `"aepo"` or `"gspo"` respectively.

3. **Automatic Configuration**: The trainer automatically sets `loss_mode` based on the `adv_estimator` value, so you only need to specify `adv_estimator: "aepo"` or `adv_estimator: "gspo"` in your config.

## Example Reference

See the implementation in:
- `/home/xw27/agent/ARLArena/recipe/simpletir/simpletir_ray_trainer.py` (lines 457-472, 343-361, 562-570)

## Summary

To add AEPO/GSPO support to your task:

1. ✅ Add `AEPO` and `GSPO` to `AdvantageEstimator` enum
2. ✅ Update `compute_advantage` to handle AEPO/GSPO (they use GRPO's advantage computation)
3. ✅ Add logic in `__init__` to set `loss_mode` automatically when AEPO/GSPO is used
4. ✅ Update `use_critic` logic to include AEPO/GSPO (both set `use_critic = False`)
5. ✅ Set `algorithm.adv_estimator: "aepo"` or `"gspo"` in your config

The loss computation logic is already implemented in verl, so no additional code is needed for that part!

