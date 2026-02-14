# eval_world_agent_api.py
# Version for API-based evaluation (DeepSeek/OpenAI)
# Enhanced version with full result saving and high-score sample extraction

from verl import DataProto
import hydra
import os
import time
import pickle
from pathlib import Path
from numbers import Number
from torchdata.stateful_dataloader import StatefulDataLoader
from recipe.shop_agent.llm_agent.agent_proxy import ApiCallingWrapperWg
from agent_system.multi_turn_rollout.rollout_loop_eval import TrajectoryCollector
from agent_system.environments import make_envs
from verl.utils import hf_processor
from recipe.world_agent.main_world_agent import create_rl_dataset, create_rl_sampler
from transformers import AutoTokenizer


def to_serializable(obj):
    """Convert common numpy/torch scalars and arrays to JSON friendly types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    # Handle numpy/torch scalar with .item()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    # Handle numpy arrays
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (str, Number)) or obj is None:
        return obj
    return str(obj)


def safe_load_tokenizer(model_path: str):
    """安全加载 tokenizer，路径或 HF ID 都可."""
    # HF 仓库 ID（形如 org/model）
    if "/" in model_path and len(model_path.split("/")) == 2 and not any(x in model_path for x in ["~", "\\", "..", ":"]):
        try:
            return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        except Exception:
            pass

    # 本地路径
    try:
        expanded = os.path.expanduser(os.path.expandvars(model_path))
        if Path(expanded).exists():
            return AutoTokenizer.from_pretrained(expanded, trust_remote_code=True, use_fast=True)
    except Exception:
        pass

    # 兜底
    print("⚠️  Loading fallback tokenizer: Qwen/Qwen2.5-7B-Instruct")
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, use_fast=True)


@hydra.main(version_base=None, config_path="config", config_name="base_eval")
def main(config):
    # 禁止 tokenizer 的多线程，避免多进程报错
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[CONFIG] Using API model: {config.model_config.model_name}")
    print(f"[CONFIG] Model provider: {config.model_info[config.model_config.model_name].provider_name}")
    print(config.data)

    tokenizer = safe_load_tokenizer(config.actor_rollout_ref.model.path)
    actor_wg = ApiCallingWrapperWg(config, tokenizer)

    envs, val_envs = make_envs(config)
    processor = hf_processor(config.actor_rollout_ref.model.path, trust_remote_code=True, use_fast=True)

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)

    train_sampler = create_rl_sampler(config.data, train_dataset)
    from verl.utils.dataset.rl_dataset import collate_fn

    val_bs = config.data.val_batch_size or 64
    val_workers = min(8, os.cpu_count() or 2)
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        num_workers=val_workers,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    for test_data in val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        # repeat test batch
        test_batch = test_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

        start_time = time.time()
        # Returns a single dict with keys: batch_output, success_rate, success, step_io_history
        result = traj_collector.multi_turn_loop_for_eval(
            gen_batch=test_gen_batch,
            actor_rollout_wg=actor_wg,
            envs=envs,
        )
        end_time = time.time()
        print(f"rollout time: {end_time - start_time} seconds")

        # Extract metrics from result
        success_rate = result.get("success_rate", {})
        success = result.get("success", {})
        print("Success rate:")
        for k, v in success_rate.items():
            print(f"{k}: {v}")

        # Persist every evaluation result for later inspection
        snapshot_dir = Path("outputs_world") / "eval_results"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = config.model_config.model_name.replace("/", "-")  # sanitize for filename
        snapshot_path = snapshot_dir / f"result_{model_name}_seed{config.env.seed}_{timestamp}.pkl"
        try:
            with snapshot_path.open("wb") as f:
                pickle.dump(result, f)
            print(f"Saved evaluation result to {snapshot_path}")
        except Exception as exc:
            print(f"Warning: failed to save evaluation result to {snapshot_path}: {exc}")

        # Save human-readable metrics to JSON alongside the pickle
        metrics_json = snapshot_dir / f"result_{model_name}_seed{config.env.seed}_{timestamp}.json"
        try:
            import json
            metrics_payload = {
                "seed": config.env.seed,
                "timestamp": timestamp,
                "model": config.model_config.model_name,  # Use actual API model name
                "tokenizer": config.actor_rollout_ref.model.path,  # Keep tokenizer path for reference
                "data": {
                    "train": config.data.train_files,
                    "val": config.data.val_files,
                },
                "success_rate": to_serializable(success_rate),
                "success": to_serializable(success),
                "step_io_history_len": len(result.get("step_io_history", [])),
                "pkl_path": str(snapshot_path),
            }
            with metrics_json.open("w", encoding="utf-8") as jf:
                json.dump(metrics_payload, jf, ensure_ascii=False, indent=2)
            print(f"Saved metrics summary to {metrics_json}")
        except Exception as exc:
            print(f"Warning: failed to save metrics JSON to {metrics_json}: {exc}")

        # Extract high-score samples for Alfworld
        # Note: Alfworld metric keys might be different from Webshop
        # success_rate is a dict, success is a dict with per-sample success indicators
        try:
            import json

            # Extract success indicators from the success dict
            # success dict typically has boolean indicators for each sample
            task_scores = None
            score_key = None

            # Check if success dict contains task-level success indicators
            if success:
                # Get the first key's values (typically 'success' or similar)
                for key, values in success.items():
                    if values is not None and len(values) > 0:
                        task_scores = values
                        score_key = key
                        break

            # If we found scores and have step history
            step_io_history = result.get("step_io_history", [])
            if task_scores is not None and step_io_history:
                output_path = f'high_score_multiturn_texts_alfworld_{model_name}_seed{config.env.seed}.json'
                all_turns = []

                # Alfworld: extract samples with score > threshold (0.5 for success)
                for sample_idx, score in enumerate(task_scores):
                    if score is not None and float(score) > 0.5:
                        sample_data = {
                            "sample_idx": sample_idx,
                            "task_score": float(score),
                            "score_key": score_key,
                        }

                        # Extract dialogue history from step_io_history
                        # step_io_history is a list of dicts with 'step', 'inputs', 'outputs'
                        if step_io_history:
                            turns = []
                            for turn_data in step_io_history:
                                turn_result = {
                                    "step": turn_data.get('step', None),
                                }
                                # Extract data for this specific sample
                                if turn_data.get('inputs') is not None and sample_idx < len(turn_data['inputs']):
                                    turn_result["input"] = turn_data['inputs'][sample_idx]
                                if turn_data.get('outputs') is not None and sample_idx < len(turn_data['outputs']):
                                    turn_result["output"] = turn_data['outputs'][sample_idx]
                                turns.append(turn_result)
                            sample_data["turns"] = turns

                        all_turns.append(sample_data)

                if all_turns:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(all_turns, f, ensure_ascii=False, indent=2)
                    print(f"Saved {len(all_turns)} high-score samples to {output_path}")
                else:
                    print(f"No samples with score > 0.9 found (checked {score_key})")
            else:
                print(f"Could not extract high-score samples: task_scores={task_scores is not None}, rollouts={rollouts is not None}")
        except Exception as exc:
            print(f"Warning: failed to extract high-score samples: {exc}")

        print(f'Total rollout time: {end_time - start_time} seconds')


if __name__ == "__main__":
    main()
