#!/usr/bin/env python3
"""
Math Agent API Evaluation Script

Evaluates API models (GPT-4o, Claude, DeepSeek, etc.) on math problems
with multi-turn code interpreter interaction.
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import pickle
import json
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from recipe.math_agent.eval.llm_agent.agent_proxy import ApiCallingWrapperWg
from recipe.math_agent.eval.llm_agent.evaluation_loop import MathEvaluationLoop
from recipe.math_agent.workers.reward_manager.math_verify import MathRewardManager
from recipe.math_agent.workers.reward_manager.math_verify_with_exec import MathRewardExecManager
from recipe.math_agent.workers.reward_manager.code import CodeRewardManager

from verl import DataProto


class Tee:
    """Duplicate writes to multiple streams (e.g., console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self):
        for stream in self._streams:
            if hasattr(stream, "fileno"):
                try:
                    return stream.fileno()
                except Exception:
                    continue
        raise OSError("No fileno on Tee streams")


def setup_log_tee() -> Path:
    """Send stdout/stderr to both console and ARLArena/logs."""
    base_dir = Path(__file__).resolve().parents[3]
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"eval_math_agent_api_{timestamp}.log"
    log_file = open(log_path, "a", buffering=1)

    stdout_console = sys.stdout
    stderr_console = sys.stderr
    sys.stdout = Tee(stdout_console, log_file)
    sys.stderr = Tee(stderr_console, log_file)
    sys._codex_log_file = log_file

    return log_path


def safe_load_tokenizer(model_path: str, fallback_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load tokenizer with fallback for API evaluation"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        print(f"[INFO] Loaded tokenizer from {model_path}")
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {model_path}: {e}")
        print(f"[INFO] Using fallback tokenizer: {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_model,
            trust_remote_code=True,
            padding_side='left'
        )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def create_reward_manager(config, tokenizer):
    """Create reward manager based on config"""
    reward_manager_type = config.reward_model.reward_manager
    num_examine = config.get('num_examine', 5)  # Number of samples to examine/print

    if reward_manager_type == "math":
        return MathRewardManager(tokenizer, num_examine)
    elif reward_manager_type == "math_exec":
        return MathRewardExecManager(tokenizer, num_examine)
    elif reward_manager_type == "code":
        return CodeRewardManager(tokenizer, num_examine)
    else:
        raise ValueError(f"Unknown reward_manager: {reward_manager_type}")


class SimpleApiEvalDataset:
    """Simple dataset for API evaluation - no tokenization needed."""

    def __init__(self, parquet_files, **kwargs):
        """Load parquet file(s) for API evaluation.

        Args:
            parquet_files: Path or list of paths to parquet files
            **kwargs: Ignored (for compatibility with create_rl_dataset signature)
        """
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        # Load all parquet files
        dataframes = [pd.read_parquet(f) for f in parquet_files]
        self.df = pd.concat(dataframes, ignore_index=True)

        print(f"[INFO] Loaded {len(self.df)} samples from {len(parquet_files)} file(s)")

    def __len__(self):
        return len(self.df)

    def _extract_prompt_text(self, prompt):
        """Extract plain question text from various prompt formats.

        Handles:
        - Qwen chat template format: [{'content': '<|im_start|>...<|im_end|>', 'role': '...'}]
        - Plain string
        - Other array formats
        """
        import re

        # If already a string, return as-is
        if isinstance(prompt, str):
            return prompt

        # Handle numpy array or list format
        if isinstance(prompt, (list, np.ndarray)) and len(prompt) > 0:
            first_item = prompt[0]
            if isinstance(first_item, dict) and 'content' in first_item:
                content = first_item['content']

                # Extract question from Qwen chat template format
                # Pattern: <|im_start|>user\n{question}<|im_end|>
                user_match = re.search(
                    r'<\|im_start\|>user\n(.*?)<\|im_end\|>',
                    content,
                    re.DOTALL
                )
                if user_match:
                    return user_match.group(1).strip()

                # If no match, try to extract text after removing special tokens
                clean_text = re.sub(r'<\|[^|]+\|>', '', content)
                # Remove role markers like "system\n" or "user\n"
                clean_text = re.sub(r'^(system|user|assistant)\n', '', clean_text.strip())
                return clean_text.strip()

            # If it's a simple string in array
            return str(first_item)

        return str(prompt)

    def __getitem__(self, idx):
        """Return sample as dict with prompt, ground_truth, data_source."""
        row = self.df.iloc[idx]

        # Extract prompt and convert to plain text for API evaluation
        prompt = row['prompt']
        prompt_text = self._extract_prompt_text(prompt)

        # Support both formats: direct 'solution' field or nested 'reward_model.ground_truth'
        if 'reward_model' in row and isinstance(row['reward_model'], dict):
            ground_truth = row['reward_model'].get('ground_truth', '')
        else:
            ground_truth = row.get('solution', row.get('output', ''))

        return {
            'prompt': prompt_text,
            'ground_truth': ground_truth,
            'output': ground_truth,
            'data_source': row.get('data_source', 'math')
        }


def create_rl_dataset(parquet_files, tokenizer, processor, prompt_key, reward_fn_key, config, is_sft):
    """Create dataset for API evaluation.

    Note: For API evaluation, we don't need tokenization or RL-specific features.
    This is a simplified wrapper for compatibility with the eval script.

    Args:
        parquet_files: Path(s) to parquet data files
        tokenizer: Ignored for API evaluation
        processor: Ignored for API evaluation
        prompt_key: Ignored for API evaluation
        reward_fn_key: Ignored for API evaluation
        config: Ignored for API evaluation
        is_sft: Ignored for API evaluation

    Returns:
        SimpleApiEvalDataset instance
    """
    return SimpleApiEvalDataset(parquet_files)


def save_results(results: dict, config: DictConfig, output_dir: Path):
    """Save evaluation results to pickle and JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = config.data.seed
    model_name = config.model_config.model_name.replace("/", "_")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle (full trajectories)
    pickle_path = output_dir / f"result_{model_name}_seed{seed}_{timestamp}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"[INFO] Saved full results to {pickle_path}")

    # Save JSON (metrics summary)
    json_data = {
        "model": config.model_config.model_name,
        "dataset": config.data.val_files,
        "timestamp": timestamp,
        "seed": seed,
        "num_problems": len(results['problems']) // config.actor_rollout_ref.rollout.val_kwargs.n,
        "num_samples": len(results['problems']),
        "metrics": {
            "accuracy": results['accuracy'],
            "pass_at_k": results['pass_at_k'],
            "avg_turns": results['avg_turns']
        },
        "config": {
            "max_turns": config.agent.max_turns,
            "num_samples_per_problem": config.actor_rollout_ref.rollout.val_kwargs.n,
            "temperature": config.model_info[config.model_config.model_name].generation_kwargs.temperature,
            "max_completion_tokens": config.model_info[config.model_config.model_name].generation_kwargs.get(
                'max_completion_tokens',
                config.model_info[config.model_config.model_name].generation_kwargs.get('max_tokens', None)
            ),
            "reward_manager": config.reward_model.reward_manager
        }
    }

    json_path = output_dir / f"result_{model_name}_seed{seed}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[INFO] Saved metrics summary to {json_path}")

    # Optionally save high-score trajectories
    if results['accuracy'] > 0:
        high_score_trajectories = []
        n = config.actor_rollout_ref.rollout.val_kwargs.n
        for i, (reward, conv, problem, gt) in enumerate(zip(
            results['rewards'],
            results['conversation_histories'],
            results['problems'],
            results['ground_truths']
        )):
            if reward >= 0:  # all trajectory
                high_score_trajectories.append({
                    "sample_id": i,
                    "problem": problem,
                    "ground_truth": gt,
                    "conversation": conv,
                    "reward": float(reward),
                    "turns": results['turn_counts'][i]
                })

        if high_score_trajectories:
            high_score_path = output_dir / f"high_score_{model_name}_seed{seed}_{timestamp}.json"
            with open(high_score_path, 'w') as f:
                json.dump(high_score_trajectories, f, indent=2)
            print(f"[INFO] Saved {len(high_score_trajectories)} high-score trajectories to {high_score_path}")


@hydra.main(version_base=None, config_path="config", config_name="base_eval")
def main(config: DictConfig):
    """Main evaluation entry point"""
    log_path = setup_log_tee()
    print(f"[INFO] Logging to file: {log_path}")

    print("\n" + "="*80)
    print("Math Agent API Evaluation")
    print("="*80)
    print(f"Model: {config.model_config.model_name}")
    print(f"Reward Manager: {config.reward_model.reward_manager}")
    print(f"Max Turns: {config.agent.max_turns}")
    print(f"Val Files: {config.data.val_files}")
    print(f"Batch Size: {config.data.val_batch_size}")
    print(f"Num Samples (n): {config.actor_rollout_ref.rollout.val_kwargs.n}")
    print("="*80 + "\n")

    # 1. Load tokenizer (for logging only, API handles its own tokenization)
    model_path = config.actor_rollout_ref.model.path
    tokenizer = safe_load_tokenizer(model_path)
    processor = None  # Math agent doesn't use processor

    # 2. Initialize API wrapper
    actor_wrapper = ApiCallingWrapperWg(config, tokenizer)

    # 3. Initialize reward manager
    reward_manager = create_reward_manager(config, tokenizer)
    print(f"[INFO] Reward manager initialized: {config.reward_model.reward_manager}")

    # 4. Initialize evaluation loop
    eval_loop = MathEvaluationLoop(
        config=config,
        tokenizer=tokenizer,
        actor_wrapper=actor_wrapper,
        reward_manager=reward_manager
    )
    print("[INFO] MathEvaluationLoop initialized\n")

    # 5. Load validation dataset
    print("[INFO] Loading validation dataset...")
    val_dataset = create_rl_dataset(
        parquet_files=config.data.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.data.prompt_key,
        reward_fn_key=config.data.get('reward_fn_key', 'data_source'),
        config=config.data,
        is_sft=False
    )
    print(f"[INFO] Loaded {len(val_dataset)} validation samples\n")

    # 6. Run evaluation in batches
    all_results = []
    batch_size = config.data.val_batch_size

    for batch_idx in range(0, len(val_dataset), batch_size):
        batch_end = min(batch_idx + batch_size, len(val_dataset))
        batch_samples = [val_dataset[i] for i in range(batch_idx, batch_end)]

        # Extract batch data
        batch_data = {
            'prompt': [s['prompt'] for s in batch_samples],
            'ground_truth': [s.get('ground_truth', s.get('output', '')) for s in batch_samples],
            'data_source': [s.get('data_source', 'math') for s in batch_samples]
        }

        print(f"\n{'='*80}")
        print(f"Batch {batch_idx//batch_size + 1}/{(len(val_dataset) + batch_size - 1)//batch_size}")
        print(f"Samples {batch_idx} - {batch_end-1} ({batch_end - batch_idx} problems)")
        print(f"{'='*80}")

        # Run evaluation
        batch_results = eval_loop.run_evaluation(
            batch_data=batch_data,
            num_samples=config.actor_rollout_ref.rollout.val_kwargs.n
        )
        all_results.append(batch_results)

    # 7. Aggregate results
    print(f"\n\n{'='*80}")
    print("Aggregating Results")
    print(f"{'='*80}\n")

    aggregated_results = {
        'rewards': np.concatenate([r['rewards'] for r in all_results]),
        'trajectories': [t for r in all_results for t in r['trajectories']],
        'conversation_histories': [c for r in all_results for c in r['conversation_histories']],
        'turn_counts': [t for r in all_results for t in r['turn_counts']],
        'problems': [p for r in all_results for p in r['problems']],
        'ground_truths': [g for r in all_results for g in r['ground_truths']],
        'data_sources': [d for r in all_results for d in r['data_sources']]
    }

    # Recompute overall metrics
    aggregated_results['accuracy'] = float(np.mean(aggregated_results['rewards']))
    aggregated_results['avg_turns'] = float(np.mean(aggregated_results['turn_counts']))

    # Recompute pass@k
    n = config.actor_rollout_ref.rollout.val_kwargs.n
    num_problems = len(aggregated_results['rewards']) // n
    rewards_matrix = aggregated_results['rewards'].reshape(num_problems, n)
    aggregated_results['pass_at_k'] = {
        1: float((rewards_matrix.sum(axis=1) >= 1).mean()),
        4: float((rewards_matrix.sum(axis=1) >= 1).mean()) if n >= 4 else None
    }

    print("\n" + "="*80)
    print("Final Results")
    print("="*80)
    print(f"Total Problems: {num_problems}")
    print(f"Total Samples: {len(aggregated_results['rewards'])}")
    print(f"Accuracy: {aggregated_results['accuracy']:.4f}")
    print(f"Pass@1: {aggregated_results['pass_at_k'][1]:.4f}")
    if aggregated_results['pass_at_k'][4] is not None:
        print(f"Pass@4: {aggregated_results['pass_at_k'][4]:.4f}")
    print(f"Avg Turns: {aggregated_results['avg_turns']:.2f}")
    print("="*80 + "\n")

    # 8. Save results
    output_dir = Path(config.evaluation.output_dir)
    save_results(aggregated_results, config, output_dir)

    print("\n[INFO] Evaluation complete!\n")


if __name__ == "__main__":
    main()
