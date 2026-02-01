import asyncio
import logging
import math
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from verl import DataProto

# if os.getenv("SANDBOX_ENDPOINT", None) is not None:
#     from sandbox.local_sandbox import parallel_sandbox
# else:
#     from sandbox.internal_sandbox import parallel_sandbox
from sandbox.local_sandbox import parallel_sandbox

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

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
    # AEPO parameters
    entropy_weight: float = 0.5  # Weight for entropy in branching decision
    branch_probability: float = 0.5  # Base probability threshold for branching
    initial_rollouts: int = 1  # Initial number of rollouts (if different from rollout_n)
    beam_size: int = 2  # Maximum branches to create from one sample
    logprobs: int = 10  # Number of top logprobs to request from generation
    # AEPO-specific knobs
    enable_dynamic_rollouts: bool = False
    initial_entropy_tokens: int = 50
    dynamic_rollout_min: int = 1
    dynamic_rollout_max: Optional[int] = None
    consecutive_branch_penalty: float = 0.05


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

        # AEPO-specific scheduling parameters
        self.initial_entropy_tokens = config.initial_entropy_tokens
        self.enable_dynamic_rollouts = config.enable_dynamic_rollouts
        self.dynamic_rollout_min = max(1, config.dynamic_rollout_min)
        self.dynamic_rollout_max = (
            config.dynamic_rollout_max
            if config.dynamic_rollout_max is not None
            else config.rollout_n
        )
        self.initial_rollouts = max(
            1, config.initial_rollouts if config.initial_rollouts else 1
        )
        self.entropy_weight = config.entropy_weight
        self.branch_probability = config.branch_probability
        self.consecutive_branch_penalty = max(config.consecutive_branch_penalty, 0.0)
        self.beam_size = getattr(config, "beam_size", 1)
        self.logprobs = config.logprobs

        # Runtime trackers
        self.initial_entropy_dict: Dict[int, float] = {}
        self.consecutive_branches: Dict[int, int] = {}

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
                    # Calculate entropy per token, then average (more accurate than joint entropy)
                    token_entropies = []
                    if isinstance(sample_logprobs, list):
                        for lp_item in sample_logprobs[:20]:  # First 20 tokens
                            token_logprobs = []
                            if isinstance(lp_item, dict):
                                # Format: {token_id: LogprobInfo}
                                token_logprobs = [
                                    lp.logprob if hasattr(lp, 'logprob') else lp
                                    for lp in lp_item.values()
                                ]
                            elif hasattr(lp_item, 'logprob'):
                                token_logprobs = [lp_item.logprob]
                            elif isinstance(lp_item, (int, float)):
                                token_logprobs = [float(lp_item)]
                            
                            # Calculate entropy for this token position
                            if token_logprobs:
                                token_entropy = self._calc_entropy(token_logprobs)
                                token_entropies.append(token_entropy)
                    
                    # Average token-level entropies and normalize
                    if token_entropies:
                        avg_entropy = sum(token_entropies) / len(token_entropies)
                        normalized_entropy = avg_entropy / entropy_norm_factor
                        return normalized_entropy
        
        # Fallback: try to extract from responses if available
        # This is a placeholder - you may need to adjust based on your actual output format
        return 0.0

    def _clone_dataproto(self, proto: DataProto) -> DataProto:
        batch = {k: v.clone() for k, v in proto.batch.items()}
        cloned = DataProto.from_dict(batch)
        cloned.meta_info = deepcopy(proto.meta_info) if proto.meta_info else {}
        cloned.non_tensor_batch = deepcopy(proto.non_tensor_batch)
        return cloned

    def _repeat_tensor_by_counts(self, tensor: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if tensor.size(0) != counts.size(0):
            raise ValueError("repeat counts must align with batch dimension")
        repeats = counts.to(tensor.device)
        return torch.repeat_interleave(tensor, repeats, dim=0)

    def _repeat_batch_by_counts(self, batch: Dict[str, torch.Tensor], counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        repeated = {}
        for key, value in batch.items():
            if not torch.is_tensor(value):
                raise TypeError(f"Unsupported type for batch key {key}: {type(value)}")
            repeated[key] = self._repeat_tensor_by_counts(value, counts)
        return repeated

    def _prepare_sampling_meta(self, data_proto: DataProto, max_tokens: Optional[int] = None, n: Optional[int] = None):
        data_proto.meta_info = data_proto.meta_info or {}
        sampling_params = data_proto.meta_info.get("sampling_params", {})
        sampling_params.setdefault("logprobs", self.logprobs)
        if max_tokens is not None:
            sampling_params["max_tokens"] = max_tokens
        if n is not None:
            sampling_params["n"] = n
        data_proto.meta_info["sampling_params"] = sampling_params

    def _calculate_dynamic_initial_rollouts(self, gen_batch: DataProto) -> List[int]:
        """
        Calculate initial rollouts per sample by generating full trajectories
        and averaging entropy across all turns. Processes entire batch in parallel.
        """
        batch_size = gen_batch.batch["input_ids"].shape[0]
        default_rollouts = [self.initial_rollouts] * batch_size
        if not self.enable_dynamic_rollouts:
            return default_rollouts

        try:
            vocab_size = len(self.tokenizer)
            initial_entropy_tokens = self.config.initial_entropy_tokens
            
            # Step 1: Calculate initial entropy (H_root) for all samples in parallel
            probe_batch = self._clone_dataproto(gen_batch)
            self._prepare_sampling_meta(
                probe_batch,
                max_tokens=max(1, initial_entropy_tokens),
                n=1,
            )
            initial_output = self._generate_with_gpu_padding(probe_batch)
            
            # Extract initial entropy (H_root) for each sample
            initial_entropies = []
            for sample_idx in range(batch_size):
                initial_entropy = self._get_entropy_from_output(initial_output, sample_idx, vocab_size=vocab_size)
                initial_entropies.append(max(0.0, min(1.0, initial_entropy)))
            
            # Step 2: Execute tool invocation inference flow in parallel
            # Initialize state for all samples in parallel
            rollings = self._clone_dataproto(gen_batch)
            active_mask = torch.ones(batch_size, dtype=torch.bool)
            
            # Track step entropy per sample (H_high)
            step_entropies_list = [[] for _ in range(batch_size)]
            
            # Generate full trajectories for all samples in parallel
            for step in range(self.config.max_turns):
                if not active_mask.sum():
                    break
                
                rollings.batch = self.tensor_fn.cut_to_effective_len(
                    rollings.batch, keys=["input_ids", "attention_mask", "position_ids"]
                )
                
                # Generate for all active samples in parallel
                rollings_active = DataProto.from_dict(
                    {k: v[active_mask] for k, v in rollings.batch.items()}
                )
                rollings_active.meta_info = deepcopy(rollings.meta_info)
                
                # Explicitly set n=1 for probe generation (one sample per batch item)
                self._prepare_sampling_meta(rollings_active, n=1)
                
                gen_output = self._generate_with_gpu_padding(rollings_active)
                
                # Extract entropy for each active sample
                active_indices_list = torch.where(active_mask)[0].tolist()
                for i, sample_idx in enumerate(active_indices_list):
                    entropy = self._get_entropy_from_output(gen_output, i, vocab_size=vocab_size)
                    step_entropies_list[sample_idx].append(entropy)
                
                # Post-process responses
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
                
                if not active_mask.sum():
                    break
                
                if step < self.config.max_turns - 1:
                    if step == self.config.max_turns - 2:
                        for i, obs in enumerate(next_obs):
                            if len(obs) > 0:
                                next_obs[i] += self.prompt_dict["final_prompt"]
                    
                    next_obs_ids = self._process_next_obs(next_obs)
                    rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            
            # Step 3: Calculate initial_rollouts using H_root and H_high (matching original AEPO)
            rollouts = []
            num_samples = self.config.rollout_n
            for i in range(batch_size):
                # Calculate average step entropy (H_high)
                if step_entropies_list[i]:
                    avg_step_entropy = sum(step_entropies_list[i]) / len(step_entropies_list[i])
                else:
                    # If no step entropy, use initial entropy as fallback
                    avg_step_entropy = initial_entropies[i]
                
                # Calculate initial_rollouts using original AEPO formula
                entropy_diff = initial_entropies[i] - avg_step_entropy
                sigmoid_input = 0.5 * entropy_diff
                sigmoid_value = 1.0 / (1.0 + math.exp(-sigmoid_input))
                
                # Round up
                initial_rollouts = int(num_samples * sigmoid_value) + 1
                initial_rollouts = max(1, min(initial_rollouts, num_samples))
                
                rollouts.append(initial_rollouts)
            
        except Exception as exc:
            logger.warning("Dynamic rollout probing failed: %s", exc)
            return default_rollouts

        return rollouts

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
        AEPO: Also requests logprobs for entropy calculation.
        
        if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
        if active_batch size is not divisible by num_gpus, pad with first sequence
        then remove padding from output
        """
        num_gpus = self.config.num_gpus
        
        if not hasattr(active_batch, 'meta_info') or active_batch.meta_info is None:
            active_batch.meta_info = {}

        sampling_params = active_batch.meta_info.get('sampling_params', {})
        sampling_params.setdefault('logprobs', self.logprobs)
        max_tokens_override = active_batch.meta_info.pop("max_tokens_override", None)
        if max_tokens_override is not None:
            sampling_params["max_tokens"] = max(1, int(max_tokens_override))
        active_batch.meta_info['sampling_params'] = sampling_params
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

    def _should_branch_aepo(
        self, entropy_now: float, entropy_init: float, sample_idx: int
    ) -> bool:
        entropy_delta = entropy_now - entropy_init
        prob = random.random() + self.entropy_weight * entropy_delta
        penalty_factor = max(
            0.0, 1.0 - self.consecutive_branch_penalty * self.consecutive_branches.get(sample_idx, 0)
        )
        prob *= penalty_factor
        prob = max(0.0, min(1.0, prob))
        return prob >= self.branch_probability

    def _update_consecutive_branch_counter(
        self, new_sample_origins: List[int], batch_size: int
    ):
        updated_samples = set(new_sample_origins)
        for sample_idx in range(batch_size):
            if sample_idx in updated_samples:
                self.consecutive_branches[sample_idx] = self.consecutive_branches.get(sample_idx, 0) + 1
            else:
                self.consecutive_branches[sample_idx] = 0

    def run_llm_loop(
        self,
        gen_batch,
        initial_input_ids: torch.Tensor,
        timeout: int = 5,
    ) -> Tuple[Dict, Dict]:
        """
        Run main LLM generation loop with AEPO entropy-based adaptive branching.
        """
        original_batch_size = gen_batch.batch["input_ids"].shape[0]
        batch_size = original_batch_size  # Keep for compatibility, but use original_batch_size for final size check

        self.timeout = timeout

        original_left_side = {
            "input_ids": initial_input_ids[:, -self.config.max_start_length :]
        }
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        self.initial_entropy_dict = {}
        rollouts_per_sample = self._calculate_dynamic_initial_rollouts(gen_batch)
        
        
        initial_rollout_plan = list(rollouts_per_sample)
        repeat_counts = torch.tensor(
            rollouts_per_sample,
            device=initial_input_ids.device if initial_input_ids.is_cuda else torch.device("cpu"),
            dtype=torch.long,
        )
        total_rollouts = int(repeat_counts.sum().item())
        
        # Final check: ensure we don't exceed budget
        if total_rollouts > total_budget:
            raise ValueError(f"Initial rollouts ({total_rollouts}) exceed total budget ({total_budget})")

        if total_rollouts == 0:
            raise ValueError("Dynamic rollout calculation produced zero rollouts.")

        expanded_batch = self._repeat_batch_by_counts(gen_batch.batch, repeat_counts)
        rollings = DataProto.from_dict(expanded_batch)
        rollings.meta_info = deepcopy(gen_batch.meta_info) if gen_batch.meta_info else {}
        rollings.non_tensor_batch = deepcopy(gen_batch.non_tensor_batch)

        base_left_inputs = initial_input_ids[:, -self.config.max_start_length :]
        expanded_left_inputs = self._repeat_tensor_by_counts(base_left_inputs, repeat_counts)
        original_left_side = {"input_ids": expanded_left_inputs}

        empty_responses = initial_input_ids.new_empty((total_rollouts, 0))
        original_right_side = {
            "responses": empty_responses.clone(),
            "responses_with_info_mask": empty_responses.clone(),
        }

        active_mask = torch.ones(total_rollouts, dtype=torch.bool)
        void_turn_mask = torch.ones(total_rollouts, dtype=torch.bool)
        turns_stats = torch.ones(total_rollouts, dtype=torch.int)
        use_code_stats = torch.zeros(total_rollouts, dtype=torch.int)
        valid_code_stats = torch.zeros(total_rollouts, dtype=torch.int)
        success_code_lines = []
        fail_code_lines = []
        success_code_strip_lines = []
        fail_code_strip_lines = []
        active_num_list = [active_mask.sum().item()]
        current_entropy_dict: Dict[int, float] = {}
        vocab_size = len(self.tokenizer)
        entropy_norm_factor = math.log(vocab_size)

        sample_to_indices: Dict[int, List[int]] = {}
        index_to_sample: List[int] = []
        cursor = 0
        for sample_idx, count in enumerate(rollouts_per_sample):
            indices = list(range(cursor, cursor + count))
            sample_to_indices[sample_idx] = indices
            index_to_sample.extend([sample_idx] * count)
            cursor += count
        self.consecutive_branches = {i: 0 for i in range(batch_size)}

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch, keys=["input_ids", "attention_mask", "position_ids"]
            )

            rollings_active = DataProto.from_dict(
                {k: v[active_mask] for k, v in rollings.batch.items()}
            )
            rollings_active.meta_info = deepcopy(rollings.meta_info)

            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            # AEPO: Calculate entropy for each active sample
            active_indices_list = torch.where(active_mask)[0].tolist()
            for i, rollout_idx in enumerate(active_indices_list):
                entropy = self._get_entropy_from_output(gen_output, i, vocab_size)
                current_entropy_dict[rollout_idx] = entropy
                
                # Store initial entropy on first generation for this rollout
                if rollout_idx not in self.initial_entropy_dict:
                    self.initial_entropy_dict[rollout_idx] = entropy

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

            # AEPO: Create new trajectories to satisfy budget M when needed
            if step < self.config.max_turns - 1:  # Only create new rollouts if not last step
                next_active_indices = torch.where(active_mask)[0].tolist()
                final_active_indices = next_active_indices

                # Separate active indices by original sample
                active_by_sample: Dict[int, List[int]] = {}
                for idx in final_active_indices:
                    orig_sample = index_to_sample[idx]
                    active_by_sample.setdefault(orig_sample, []).append(idx)
                
                # Create new rollouts using entropy-based branching
                new_rollout_indices = []
                new_rollout_input_ids = []
                new_rollout_responses = []
                new_rollout_responses_mask = []
                new_rollout_sample_origins = []
                
                current_num_rollouts = len(active_mask)
                
                # Branch from active trajectories using entropy-based decision (per-sample budget only)
                for orig_sample, active_idxs in active_by_sample.items():
                    # Per-sample budget: remaining_slots = rollout_n - current_rollouts_for_sample
                    remaining_slots = self.config.rollout_n - rollouts_per_sample[orig_sample]
                    if remaining_slots <= 0:
                        continue
                    
                    branches_created = 0
                    for source_idx in active_idxs:
                        branches_per_idx = min(self.beam_size - 1, remaining_slots - branches_created)
                        if branches_per_idx <= 0:
                            break
                        
                        for _ in range(branches_per_idx):
                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = self.initial_entropy_dict.get(source_idx, 0.0)
                            should_branch = self._should_branch_aepo(
                                entropy_now, entropy_init, orig_sample
                            )

                            if not should_branch:
                                continue

                            new_rollout_input_ids.append(
                                rollings.batch["input_ids"][source_idx].clone()
                            )
                            new_rollout_responses.append(
                                original_right_side["responses"][source_idx].clone()
                            )
                            new_rollout_responses_mask.append(
                                original_right_side["responses_with_info_mask"][source_idx].clone()
                            )
                            new_rollout_sample_origins.append(orig_sample)

                            new_idx = current_num_rollouts + len(new_rollout_indices)
                            new_rollout_indices.append(new_idx)

                            rollouts_per_sample[orig_sample] += 1
                            branches_created += 1
                            sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                            index_to_sample.append(orig_sample)
                        
                        if branches_created >= remaining_slots:
                            break
                
                # Add new rollouts for samples that have no active rollouts but need more
                for orig_sample in range(original_batch_size):
                    if orig_sample not in active_by_sample and rollouts_per_sample[orig_sample] < self.config.rollout_n:
                        # Sample has no active rollouts but still needs more (per-sample budget only)
                        branches_to_add = min(
                            1, 
                            self.config.rollout_n - rollouts_per_sample[orig_sample]
                        )
                        
                        if branches_to_add > 0:
                            current_indices = sample_to_indices.get(orig_sample, [])
                            source_idx = None
                            for candidate in current_indices:
                                if candidate < len(active_mask) and active_mask[candidate]:
                                    source_idx = candidate
                                    break

                            if source_idx is not None:
                                new_rollout_input_ids.append(
                                    rollings.batch["input_ids"][source_idx].clone()
                                )
                            else:
                                new_rollout_input_ids.append(base_left_inputs[orig_sample].clone())

                            empty_responses = torch.zeros_like(original_right_side["responses"][0])
                            new_rollout_responses.append(empty_responses)
                            new_rollout_responses_mask.append(empty_responses.clone())
                            new_rollout_sample_origins.append(orig_sample)

                            new_idx = current_num_rollouts + len(new_rollout_indices)
                            new_rollout_indices.append(new_idx)

                            sample_to_indices.setdefault(orig_sample, []).append(new_idx)
                            index_to_sample.append(orig_sample)
                            rollouts_per_sample[orig_sample] += 1
                    
                    # Add new rollouts to the batch
                    if new_rollout_indices:
                        # Extend rollings
                        new_input_ids = torch.stack([new_rollout_input_ids[i] for i in range(len(new_rollout_input_ids))])
                        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
                        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
                        
                        rollings.batch["input_ids"] = torch.cat([rollings.batch["input_ids"], new_input_ids], dim=0)
                        rollings.batch["attention_mask"] = torch.cat([rollings.batch["attention_mask"], new_attention_mask], dim=0)
                        rollings.batch["position_ids"] = torch.cat([rollings.batch["position_ids"], new_position_ids], dim=0)
                        
                        # Extend original_right_side - all responses should already have the same shape
                        new_responses = torch.stack([new_rollout_responses[i] for i in range(len(new_rollout_responses))])
                        new_responses_mask = torch.stack([new_rollout_responses_mask[i] for i in range(len(new_rollout_responses_mask))])
                        
                        original_right_side["responses"] = torch.cat([original_right_side["responses"], new_responses], dim=0)
                        original_right_side["responses_with_info_mask"] = torch.cat([original_right_side["responses_with_info_mask"], new_responses_mask], dim=0)
                        
                        # Extend original_left_side (duplicate if needed)
                        if len(new_rollout_indices) > 0:
                            # For new rollouts, use the original left side from their source sample
                            new_left_side_input_ids = []
                            for orig_sample in new_rollout_sample_origins:
                                new_left_side_input_ids.append(base_left_inputs[orig_sample].clone())
                            
                            if new_left_side_input_ids:
                                new_left_side = torch.stack(new_left_side_input_ids)
                                original_left_side["input_ids"] = torch.cat([original_left_side["input_ids"], new_left_side], dim=0)
                        
                        self._update_consecutive_branch_counter(new_rollout_sample_origins, original_batch_size)

                        # Extend all tracking arrays
                        new_active_mask = torch.ones(len(new_rollout_indices), dtype=torch.bool)
                        active_mask = torch.cat([active_mask, new_active_mask], dim=0)
                        
                        new_void_turn_mask = torch.ones(len(new_rollout_indices), dtype=torch.bool)
                        void_turn_mask = torch.cat([void_turn_mask, new_void_turn_mask], dim=0)
                        
                        new_turns_stats = torch.zeros(len(new_rollout_indices), dtype=torch.int)
                        turns_stats = torch.cat([turns_stats, new_turns_stats], dim=0)
                        
                        new_use_code_stats = torch.zeros(len(new_rollout_indices), dtype=torch.int)
                        use_code_stats = torch.cat([use_code_stats, new_use_code_stats], dim=0)
                        
                        new_valid_code_stats = torch.zeros(len(new_rollout_indices), dtype=torch.int)
                        valid_code_stats = torch.cat([valid_code_stats, new_valid_code_stats], dim=0)
                        
                        # Initialize entropy for new rollouts (will be set on next generation)
                        active_num_list.append(active_mask.sum().item())

        # Add entropy statistics to meta_info
        entropy_stats = {}
        for rollout_idx in self.initial_entropy_dict.keys():
            entropy_init = self.initial_entropy_dict.get(rollout_idx, 0.0)
            entropy_now = current_entropy_dict.get(rollout_idx, 0.0)
            entropy_delta = entropy_now - entropy_init
            
            sample_idx = index_to_sample[rollout_idx] if rollout_idx < len(index_to_sample) else 0
            should_branch = self._should_branch_aepo(entropy_now, entropy_init, sample_idx)
            
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
        meta_info["initial_rollout_plan"] = initial_rollout_plan
        meta_info["final_rollout_plan"] = rollouts_per_sample

        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # Extract token-level entropy from logprobs if available in meta_info
        # Store it in original_right_side so it can be passed to advantage computation
        if 'logprobs' in meta_info and meta_info['logprobs']:
            # Try to extract token-level entropy from logprobs
            # Note: In original AEPO, entropy is computed from logits during actor update,
            # but we can extract from logprobs here if available for early use
            batch_size = original_right_side["responses"].shape[0]
            response_length = original_right_side["responses"].shape[1]
            # Initialize entropy tensor with zeros
            token_entropy = torch.zeros((batch_size, response_length), dtype=torch.float32, device=original_right_side["responses"].device)
            
            # Extract entropy from logprobs for each sample
            logprobs_data = meta_info['logprobs']
            vocab_size = len(self.tokenizer)
            entropy_norm_factor = math.log(vocab_size)
            
            # If logprobs are stored as a list, try to extract token-level entropy
            # This is a simplified extraction - in practice, entropy should be computed from logits
            # during actor update for accurate token-level entropy
            if isinstance(logprobs_data, list) and len(logprobs_data) > 0:
                # For now, set to zeros - proper extraction requires detailed logprob structure
                # Entropy will be computed from logits during actor update (as in original AEPO)
                pass
        
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

