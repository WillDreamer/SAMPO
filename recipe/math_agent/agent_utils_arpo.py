import asyncio
import os
import re
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from verl import DataProto

# if os.getenv("SANDBOX_ENDPOINT", None) is not None:
#     from sandbox.local_sandbox import parallel_sandbox
# else:
#     from sandbox.internal_sandbox import parallel_sandbox
from sandbox.local_sandbox import parallel_sandbox

MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(
    content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT
) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        keys: List[str],
        cut_left: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict["attention_mask"].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(
        self, tensor: torch.Tensor, pad_to_left: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = (
            tensor != self.config.pad_token_id
            if pad_to_left
            else tensor == self.config.pad_token_id
        )
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self, tensors: List[torch.Tensor], pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(
        self,
        responses: torch.Tensor,
        responses_str: List[str],
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0], (
            f"{active_mask.sum()} != {responses.shape[0]}"
        )
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len),
            self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses[active_mask] = responses

        # Create masked response strings
        padded_responses_str = [""] * batch_size

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1

        return padded_responses, padded_responses_str


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    rollout_n: int
    mask_void_turns: bool
    append_final_answer_func: bool
    sandbox_run_timeout: float = 3.0
    # ARPO Entropy parameters
    entropy_weight: float = 0.5  # Weight for entropy in branching decision
    branch_probability: float = 0.5  # Base probability threshold for branching
    initial_rollouts: Optional[int] = None  # Initial number of rollouts (if different from rollout_n)
    beam_size: int = 2  # Maximum branches to create from one sample
    logprobs: int = 10  # Number of top logprobs to request from generation


class AgentHelper:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

        self.prompt_dict = {
            "no_tool_prompt": "\n",
            "final_prompt": "\n",
        }
        self.error_n_line = 1

        # ARPO Entropy: Initialize entropy tracking
        self.initial_entropy_dict = {}  # Record initial entropy of active indices
        self.entropy_weight = config.entropy_weight
        self.branch_probability = config.branch_probability
        self.initial_rollouts = config.initial_rollouts if config.initial_rollouts is not None else config.rollout_n
        self.beam_size = config.beam_size
        self.logprobs = config.logprobs

    def _calc_entropy(self, logprobs: List[float]) -> float:
        """
        Calculate Shannon entropy from log probabilities.
        
        Args:
            logprobs: List of log probabilities
            
        Returns:
            Entropy value
        """
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        entropy = -sum(p * l for p, l in zip(p_list, logprobs))
        return entropy

    def _get_entropy_from_output(
        self, 
        gen_output: DataProto, 
        sample_idx: int,
        vocab_size: Optional[int] = None
    ) -> float:
        """
        Extract and calculate normalized entropy from generation output.
        
        Args:
            gen_output: DataProto containing generation output
            sample_idx: Index of the sample in the batch
            vocab_size: Vocabulary size for normalization
            
        Returns:
            Normalized entropy (0-1 range)
        """
        if vocab_size is None:
            vocab_size = len(self.tokenizer)
        
        entropy_norm_factor = math.log(vocab_size)
        
        # Try to extract logprobs from meta_info
        if hasattr(gen_output, 'meta_info') and gen_output.meta_info:
            if 'logprobs' in gen_output.meta_info:
                logprobs_data = gen_output.meta_info['logprobs']
                if sample_idx < len(logprobs_data):
                    # Extract logprobs for this sample
                    sample_logprobs = logprobs_data[sample_idx]
                    
                    # Handle different logprob formats
                    token_logprobs = []
                    if isinstance(sample_logprobs, list):
                        for lp_item in sample_logprobs[:20]:  # First 20 tokens
                            if isinstance(lp_item, dict):
                                # Format: {token_id: LogprobInfo}
                                token_logprobs.extend([
                                    lp.logprob if hasattr(lp, 'logprob') else lp
                                    for lp in lp_item.values()
                                ])
                            elif hasattr(lp_item, 'logprob'):
                                token_logprobs.append(lp_item.logprob)
                            elif isinstance(lp_item, (int, float)):
                                token_logprobs.append(float(lp_item))
                    
                    if token_logprobs:
                        entropy = self._calc_entropy(token_logprobs) / entropy_norm_factor
                        return entropy
        
        # Fallback: try to extract from responses if available
        # This is a placeholder - you may need to adjust based on your actual output format
        return 0.0

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, add_special_tokens=False, return_tensors="pt", padding="longest"
        )["input_ids"]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        # Drop the responses after the code block
        for i in range(len(responses_str)):
            pattern = r"^(.*?)(```(?:py|python)?\n.*?\n```)"
            match = re.search(pattern, responses_str[i], re.DOTALL)
            if match:
                preceding_text = match.group(1).strip()
                code_block = match.group(2).strip()
                responses_str[i] = preceding_text + code_block

        responses = self._batch_tokenize(responses_str)

        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,  # Prevents adding special tokens
        )["input_ids"]

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}"
            )
            next_obs_ids = next_obs_ids[:, : self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(
        self,
        rollings: DataProto,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor,
    ) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding(
            [rollings.batch["input_ids"], cur_responses, next_obs_ids]
        )

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
            }
        )
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True,
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(
                info.size(), pad_id, dtype=info.dtype, device=info.device
            )  # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(
        self,
        right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None,
    ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    next_obs_ids,
                    pad_to_left=False,
                )
            )
        else:
            responses, responses_with_info_mask = (
                self._info_masked_concatenate_with_padding(
                    right_side["responses"],
                    right_side["responses_with_info_mask"],
                    cur_responses,
                    pad_to_left=False,
                )
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": responses_with_info_mask[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        ARPO Entropy: Also requests logprobs for entropy calculation.
        
        if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output
        """
        num_gpus = self.config.num_gpus
        
        # ARPO Entropy: Request logprobs in meta_info if not already present
        if not hasattr(active_batch, 'meta_info') or active_batch.meta_info is None:
            active_batch.meta_info = {}
        
        # Set logprobs request in sampling_params if supported
        if 'sampling_params' not in active_batch.meta_info:
            active_batch.meta_info['sampling_params'] = {}
        active_batch.meta_info['sampling_params']['logprobs'] = self.logprobs
        
        # Also pass logprobs directly in meta_info for easier access
        active_batch.meta_info['request_logprobs'] = self.logprobs
        
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        if hasattr(active_batch, "meta_info"):
            padded_active_batch.meta_info = active_batch.meta_info

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, "meta_info") and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                elif isinstance(v, list) and len(v) > padding_size:
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def _get_original_sample_idx(self, rollout_idx: int, batch_size: int) -> int:
        """
        Get the original sample index for a given rollout index.
        
        Args:
            rollout_idx: Index in the rollout batch
            batch_size: Original batch size
            
        Returns:
            Original sample index
        """
        return rollout_idx // self.config.rollout_n

    def _should_branch_based_on_entropy(
        self,
        entropy_now: float,
        entropy_init: float,
    ) -> bool:
        """
        ARPO Entropy: Decide whether to branch based on entropy delta.
        
        Args:
            entropy_now: Current entropy value
            entropy_init: Initial entropy value
            
        Returns:
            True if should branch, False otherwise
        """
        entropy_delta = entropy_now - entropy_init
        prob = random.random() - self.entropy_weight * entropy_delta
        prob = max(0.0, min(1.0, prob))
        
        # If prob > branch_probability, skip branching (model is confident)
        # If prob <= branch_probability, do branching (model is uncertain)
        return prob <= self.branch_probability

    def run_llm_loop(
        self,
        gen_batch,
        initial_input_ids: torch.Tensor,
        timeout: int = 5,
    ) -> Tuple[Dict, Dict]:
        """
        Run main LLM generation loop with ARPO entropy-based adaptive branching.
        """
        batch_size = gen_batch.batch["input_ids"].shape[0]

        self.timeout = timeout

        original_left_side = {
            "input_ids": initial_input_ids[:, -self.config.max_start_length :]
        }
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        active_mask = torch.ones(batch_size * self.config.rollout_n, dtype=torch.bool)
        void_turn_mask = torch.ones(
            batch_size * self.config.rollout_n, dtype=torch.bool
        )  # if void turn, set to False
        turns_stats = torch.ones(batch_size * self.config.rollout_n, dtype=torch.int)
        use_code_stats = torch.zeros(
            batch_size * self.config.rollout_n, dtype=torch.int
        )
        valid_code_stats = torch.zeros(
            batch_size * self.config.rollout_n, dtype=torch.int
        )
        success_code_lines = []
        fail_code_lines = []
        success_code_strip_lines = []
        fail_code_strip_lines = []
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # ARPO Entropy: Initialize entropy tracking dictionaries
        current_entropy_dict = {}  # Track current entropy per rollout index
        branching_decisions_per_step = {}  # Track branching decisions per step for analysis
        vocab_size = len(self.tokenizer)
        entropy_norm_factor = math.log(vocab_size)

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch, keys=["input_ids", "attention_mask", "position_ids"]
            )

            # Generate responses for active batches
            if step != 0:
                # Only keep active_mask=true samples
                rollings_active = DataProto.from_dict(
                    {k: v[active_mask] for k, v in rollings.batch.items()}
                )
            else:
                rollings_active = rollings
                if self.config.rollout_n == 1:
                    rollings_active.meta_info["n"] = 1
                else:
                    repeated_rollings_dict = {}
                    for k, v in rollings.batch.items():
                        repeated_rollings_dict[k] = v.repeat_interleave(
                            self.config.rollout_n, dim=0
                        )
                    rollings = DataProto.from_dict(repeated_rollings_dict)
                    for k, v in original_left_side.items():
                        original_left_side[k] = v.repeat_interleave(
                            self.config.rollout_n, dim=0
                        )
                    for k, v in original_right_side.items():
                        original_right_side[k] = v.repeat_interleave(
                            self.config.rollout_n, dim=0
                        )
                rollings_active = rollings
            
            # Generate with logprobs for entropy calculation
            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            # ARPO Entropy: Calculate entropy for each active sample
            active_indices_list = torch.where(active_mask)[0].tolist()
            for i, rollout_idx in enumerate(active_indices_list):
                entropy = self._get_entropy_from_output(gen_output, i, vocab_size)
                current_entropy_dict[rollout_idx] = entropy
                
                # Store initial entropy on first generation for this rollout
                if rollout_idx not in self.initial_entropy_dict:
                    self.initial_entropy_dict[rollout_idx] = entropy
                else:
                    # ARPO Entropy: Check if should branch based on entropy change
                    # This provides monitoring/analysis of when branching would occur
                    entropy_init = self.initial_entropy_dict[rollout_idx]
                    should_branch = self._should_branch_based_on_entropy(entropy, entropy_init)
                    # Store branching decision for logging (could be used for dynamic branching later)
                    if step not in branching_decisions_per_step:
                        branching_decisions_per_step[step] = {}
                    branching_decisions_per_step[step][rollout_idx] = should_branch

            # Post-process responses
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(
                gen_output.batch["responses"]
            )
            responses_ids, responses_str = self.tensor_fn._example_level_pad(
                responses_ids, responses_str, active_mask
            )

            # Execute code and get next inputs
            next_obs, dones, is_void_turn, code_info = self.execute_predictions(
                responses_str, active_mask
            )

            curr_active_mask = torch.tensor(
                [not done for done in dones], dtype=torch.bool
            )
            active_mask = active_mask * curr_active_mask
            void_turn_mask = void_turn_mask * torch.tensor(
                [not v for v in is_void_turn], dtype=torch.bool
            )
            active_num_list.append(active_mask.sum().item())
            use_code_stats += torch.tensor(code_info["use_code"], dtype=torch.int)
            valid_code_stats += torch.tensor(code_info["valid_code"], dtype=torch.int)
            success_code_lines.extend(code_info["success_code_lines"])
            fail_code_lines.extend(code_info["fail_code_lines"])
            success_code_strip_lines.extend(code_info["success_code_strip_lines"])
            fail_code_strip_lines.extend(code_info["fail_code_strip_lines"])

            if step == self.config.max_turns - 2:
                for i, obs in enumerate(next_obs):
                    if len(obs) > 0:
                        next_obs[i] += self.prompt_dict["final_prompt"]

            if step < self.config.max_turns - 1:
                turns_stats[curr_active_mask] += 1
            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(
                original_right_side, responses_ids, next_obs_ids
            )

            # ARPO Entropy: Log entropy statistics (optional)
            # You can use entropy information here for adaptive branching if needed
            # For now, we just track it for logging purposes

        # Add entropy statistics to meta_info
        entropy_stats = {}
        branching_decisions_all = {}  # Track branching decisions across all steps
        for rollout_idx in self.initial_entropy_dict.keys():
            entropy_init = self.initial_entropy_dict.get(rollout_idx, 0.0)
            entropy_now = current_entropy_dict.get(rollout_idx, 0.0)
            entropy_delta = entropy_now - entropy_init
            
            # ARPO Entropy: Use the branching decision method to determine if should branch
            should_branch = self._should_branch_based_on_entropy(entropy_now, entropy_init)
            branching_decisions_all[rollout_idx] = should_branch
            
            entropy_stats[rollout_idx] = {
                "initial_entropy": entropy_init,
                "final_entropy": entropy_now,
                "entropy_delta": entropy_delta,
                "should_branch": should_branch,  # Whether this rollout should branch based on entropy
            }

        meta_info["turns_stats"] = turns_stats.tolist()
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["void_turn_mask"] = void_turn_mask.tolist()
        meta_info["use_code_stats"] = use_code_stats.tolist()
        meta_info["valid_code_stats"] = valid_code_stats.tolist()
        meta_info["success_code_lines"] = success_code_lines
        meta_info["fail_code_lines"] = fail_code_lines
        meta_info["success_code_strip_lines"] = success_code_strip_lines
        meta_info["fail_code_strip_lines"] = fail_code_strip_lines
        meta_info["entropy_stats"] = entropy_stats
        meta_info["branching_decisions_per_step"] = branching_decisions_per_step  # Track branching decisions across all steps

        print("ACTIVE_TRAJ_NUM:", active_num_list)
        return self._compose_final_output(
            original_left_side, original_right_side, void_turn_mask, meta_info
        )

    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        void_turn_mask: torch.Tensor,
        meta_info: Dict,
    ) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output["prompts"] = left_side["input_ids"]

        # Combine input IDs
        final_output["input_ids"] = torch.cat(
            [left_side["input_ids"], right_side["responses"]], dim=1
        )

        # Create attention mask and position ids
        final_output["attention_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(final_output["responses"]),
            ],
            dim=1,
        )
        final_output["info_mask"] = torch.cat(
            [
                self.tensor_fn.create_attention_mask(left_side["input_ids"]),
                self.tensor_fn.create_attention_mask(
                    final_output["responses_with_info_mask"]
                ),
            ],
            dim=1,
        )

        # create void turn mask
        final_output["void_turn_mask"] = void_turn_mask

        final_output["position_ids"] = self.tensor_fn.create_position_ids(
            final_output["attention_mask"]
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        for key in final_output.batch.keys():
            final_output.batch[key] = final_output.batch[key].long()

        return final_output

    def execute_predictions(self, predictions: List[str], active_mask) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM

        Args:
            predictions: List of action predictions
            active_mask: Mask indicating which samples are active

        Returns:
            Tuple of (next_obs, dones, is_void_turn, code_info)
        """
        next_obs = [None] * len(active_mask)
        dones = [0] * len(active_mask)
        use_code = [0] * len(active_mask)
        valid_code = [0] * len(active_mask)
        is_void_turn = [0] * len(active_mask)  # default no void turn
        success_code_lines = []
        success_code_strip_lines = []
        fail_code_lines = []
        fail_code_strip_lines = []

        code_actions = []
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        for i, prediction in enumerate(predictions):
            # Allow answering with no code execution
            if "\\boxed{" in prediction:
                next_obs[i] = ""
                dones[i] = 1

            match = re.search(pattern, prediction, re.DOTALL)
            if match:
                code = match.group(1).strip()
                use_code[i] = 1
            else:
                code = None
                use_code[i] = 0
            code_actions.append((i, code))

        tasks = []
        index_mapping = []
        for i, code in code_actions:
            if not active_mask[i]:
                next_obs[i] = ""
                dones[i] = 1
                use_code[i] = 0
                valid_code[i] = 0
            elif code is None:
                if dones[i] == 0:
                    # Neither answer(\boxed) nor code is detected, directly stop the generation
                    next_obs[i] = self.prompt_dict["no_tool_prompt"]
                    dones[i] = 1
                    if self.config.mask_void_turns:
                        is_void_turn[i] = 1
            else:
                if self.config.append_final_answer_func:
                    code = (
                        """
def final_answer(result):
    print(f"\\\\boxed{{{result}}}")

"""
                        + code
                    )
                tasks.append(code)
                index_mapping.append(i)

        if tasks:
            sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(
                parallel_sandbox(tasks, num_processes=256, run_timeout=self.config.sandbox_run_timeout)
            )
            for j, env_idx in enumerate(index_mapping):
                success = sandbox_success[j]
                stdout = str(sandbox_stdout[j])
                stderr = str(sandbox_stderr[j])
                total_lines, code_lines = count_lines(tasks[j])

                obs = ""
                if len(stderr) > 0:
                    valid_code[env_idx] = 0
                    fail_code_lines.append(total_lines)
                    fail_code_strip_lines.append(code_lines)

                    stderr_lines = stderr.splitlines()
                    truncated_stderr = truncate_content(
                        "\n".join(stderr_lines[-self.error_n_line :]), max_length=512
                    )
                    obs = f"\nCode execution result: {truncated_stderr}\n"
                elif len(stdout) > 0:
                    valid_code[env_idx] = 1
                    success_code_lines.append(total_lines)
                    success_code_strip_lines.append(code_lines)

                    truncated_stdout = truncate_content(stdout, max_length=512)
                    obs = f"\nCode execution result: {truncated_stdout}\n"
                else:
                    # no stdout nor stderr
                    # this can happen upon sandbox error or the code block itself
                    # did not print anything
                    if not success:
                        obs = "\nCode execution result: interpreter timeout\n"
                    else:
                        obs = "\nCode execution result: \n"
                next_obs[env_idx] = obs
                if "\\boxed{" in stdout:
                    dones[env_idx] = 1

        code_info = {
            "use_code": use_code,
            "valid_code": valid_code,
            "success_code_lines": success_code_lines,
            "fail_code_lines": fail_code_lines,
            "success_code_strip_lines": success_code_strip_lines,
            "fail_code_strip_lines": fail_code_strip_lines,
        }
        print(
            f"[debug] void turn number: {sum(is_void_turn)} out of {active_mask.sum()} samples"
        )

        return next_obs, dones, is_void_turn, code_info


def count_lines(code_str: str) -> tuple[int, int]:
    """Count the number of lines in the code string.

    Args:
        code_str: The full text of the code.

    Returns:
        total_lines: The total number of lines in the code.
        code_lines: The number of lines in the code excluding empty lines and comment lines.
    """
    lines = code_str.splitlines()
    total_lines = len(lines)

    code_lines = sum(
        1 for line in lines if line.strip() and not line.lstrip().startswith("#")
    )

    return total_lines, code_lines

