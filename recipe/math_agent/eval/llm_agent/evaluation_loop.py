import asyncio
import re
from typing import Dict, List, Tuple
import numpy as np

from verl import DataProto
from sandbox.local_sandbox import parallel_sandbox


class MathEvaluationLoop:
    """Core evaluation loop for math agent API evaluation"""

    def __init__(self, config, tokenizer, actor_wrapper, reward_manager):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_wrapper = actor_wrapper
        self.reward_manager = reward_manager
        self.system_prompt = config.model_config.eval_system_prompt

    def run_evaluation(self, batch_data: Dict, num_samples: int = 4) -> Dict:
        """
        Run multi-turn evaluation on a batch of problems

        Args:
            batch_data: dict with 'prompt', 'ground_truth', 'data_source'
            num_samples: n for pass@k (repeat each problem n times)

        Returns:
            dict with rewards, trajectories, metrics
        """
        # 1. Repeat batch for pass@k
        problems = batch_data['prompt'] * num_samples
        ground_truths = batch_data['ground_truth'] * num_samples
        data_sources = batch_data.get('data_source', ['math'] * len(batch_data['prompt'])) * num_samples

        print(f"\n[INFO] Evaluating {len(problems)} samples ({len(batch_data['prompt'])} problems x {num_samples} samples each)")

        # 2. Initialize conversation histories
        conversation_histories = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem}
            ]
            for problem in problems
        ]

        trajectories = []
        active_mask = [True] * len(problems)
        turn_counts = [0] * len(problems)

        # 3. Multi-turn loop
        for turn in range(self.config.agent.max_turns):
            if not any(active_mask):
                print(f"[INFO] All samples terminated at turn {turn}")
                break

            print(f"\n[INFO] Turn {turn + 1}/{self.config.agent.max_turns}: {sum(active_mask)}/{len(problems)} samples active")

            # Build messages for active problems
            active_messages = [
                conv for i, conv in enumerate(conversation_histories)
                if active_mask[i]
            ]

            # Create DataProto for API call
            lm_inputs = DataProto()
            lm_inputs.non_tensor_batch = {
                'messages_list': np.array(active_messages, dtype=object),
                'env_ids': np.arange(len(active_messages), dtype=object),
                'group_ids': np.zeros(len(active_messages), dtype=object)
            }
            lm_inputs.meta_info = {}

            # Generate responses
            print(f"[INFO] Calling API for {len(active_messages)} active samples...")
            lm_outputs = self.actor_wrapper.generate_sequences(lm_inputs)
            responses = lm_outputs.non_tensor_batch['response_texts']

            # Execute code and update conversations
            active_idx = 0
            turn_results = []

            for i in range(len(problems)):
                if not active_mask[i]:
                    turn_results.append(None)
                    continue

                response = responses[active_idx]
                active_idx += 1
                turn_counts[i] += 1

                # Append assistant response
                conversation_histories[i].append({
                    "role": "assistant", "content": response
                })

                # Check termination condition 1: \boxed{} in response
                if "\\boxed{" in response:
                    active_mask[i] = False
                    turn_results.append({
                        "response": response,
                        "code": None,
                        "execution": None,
                        "terminated": True,
                        "reason": "boxed_in_response"
                    })
                    continue

                # Extract and execute code
                code = self._extract_code(response)
                if code:
                    stdout, stderr, success = self._execute_code(code)
                    obs = self._format_observation(stdout, stderr, success)

                    # Append observation
                    conversation_histories[i].append({
                        "role": "user", "content": obs
                    })

                    # Check if boxed in stdout
                    if "\\boxed{" in stdout:
                        active_mask[i] = False
                        turn_results.append({
                            "response": response,
                            "code": code,
                            "execution": {"stdout": stdout, "stderr": stderr, "success": success},
                            "terminated": True,
                            "reason": "boxed_in_stdout"
                        })
                    else:
                        turn_results.append({
                            "response": response,
                            "code": code,
                            "execution": {"stdout": stdout, "stderr": stderr, "success": success},
                            "terminated": False,
                            "reason": None
                        })
                else:
                    # No code, no boxed -> terminate
                    active_mask[i] = False
                    turn_results.append({
                        "response": response,
                        "code": None,
                        "execution": None,
                        "terminated": True,
                        "reason": "no_code_no_boxed"
                    })

            # Record trajectory for this turn
            trajectories.append({
                "turn": turn,
                "turn_results": turn_results,
                "active_mask": active_mask.copy()
            })

        # 4. Compute rewards
        final_responses = []
        for i, conv in enumerate(conversation_histories):
            # Get last assistant message
            for msg in reversed(conv):
                if msg["role"] == "assistant":
                    final_responses.append(msg["content"])
                    break
            else:
                final_responses.append("")  # Fallback if no assistant message

        rewards = self._compute_rewards(final_responses, ground_truths, data_sources)

        # 5. Compute pass@k
        pass_at_k = self._compute_pass_at_k(rewards, num_samples)

        # 6. Aggregate statistics
        avg_turns = np.mean([t for t, active in zip(turn_counts, [True]*len(turn_counts))])

        print(f"\n[INFO] Evaluation complete:")
        print(f"  - Accuracy: {np.mean(rewards):.4f}")
        print(f"  - Pass@1: {pass_at_k.get(1, 'N/A')}")
        print(f"  - Pass@4: {pass_at_k.get(4, 'N/A')}")
        print(f"  - Avg turns: {avg_turns:.2f}")

        return {
            "rewards": rewards,
            "trajectories": trajectories,
            "conversation_histories": conversation_histories,
            "turn_counts": turn_counts,
            "pass_at_k": pass_at_k,
            "accuracy": float(np.mean(rewards)),
            "avg_turns": float(avg_turns),
            "problems": problems,
            "ground_truths": ground_truths,
            "data_sources": data_sources
        }

    def _extract_code(self, response: str) -> str:
        """Extract code block from response"""
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    def _execute_code(self, code: str) -> Tuple[str, str, bool]:
        """Execute code in sandbox and return stdout, stderr, success"""
        # Inject final_answer function if configured
        if self.config.agent.append_final_answer_func:
            code = """
def final_answer(result):
    print(f"\\\\boxed{{{result}}}")

""" + code

        # Run sandbox
        try:
            success, stdout, stderr = asyncio.run(
                parallel_sandbox(
                    [code],
                    num_processes=1,
                    run_timeout=self.config.agent.sandbox_run_timeout
                )
            )
            return stdout[0], stderr[0], success[0]
        except Exception as e:
            return "", str(e), False

    def _format_observation(self, stdout: str, stderr: str, success: bool) -> str:
        """Format observation from code execution result"""
        if stderr:
            return f"\nCode execution result: {stderr}\n"
        elif stdout:
            return f"\nCode execution result: {stdout}\n"
        elif not success:
            return "\nCode execution result: interpreter timeout\n"
        else:
            return "\nCode execution result: \n"

    def _compute_rewards(self, responses: List[str], ground_truths: List[str], data_sources: List[str]) -> np.ndarray:
        """Compute rewards using reward manager - directly call scoring function for API evaluation"""
        # For API evaluation, directly call the scoring function with text responses
        # This bypasses __call__ which expects tokenized data
        extra_infos = [None] * len(responses)

        scores, _ = self.reward_manager.math_compute_score_parallel_with_ray(
            solution_strs=responses,
            ground_truths=ground_truths,
            data_sources=data_sources,
            extra_infos=extra_infos
        )

        return np.array(scores)

    def _compute_pass_at_k(self, rewards: np.ndarray, n: int) -> Dict[int, float]:
        """Compute pass@k metric"""
        # rewards: [batch_size * n]
        # Reshape to [batch_size, n]
        batch_size = len(rewards) // n
        rewards_matrix = rewards.reshape(batch_size, n)

        pass_at_k = {}
        for k in [1, 4]:
            if k <= n:
                # At least 1 correct sample means pass
                if k == 1:
                    pass_at_k[k] = float((rewards_matrix.sum(axis=1) >= 1).mean())
                else:
                    # For k>1, we use the standard pass@k formula
                    # But here we simplify: if any sample is correct, count as pass
                    pass_at_k[k] = float((rewards_matrix.sum(axis=1) >= 1).mean())

        return pass_at_k
