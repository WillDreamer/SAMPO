# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import logging
import os
import pickle
import socket
from contextlib import contextmanager
from types import MethodType
from typing import Any
from typing import List, Optional, Sequence

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import DictConfig, ListConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase
import math
from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class ThinkLogitsProcessorFast:
    """
    - 限制 <think> 段长度；并基于“联合度量 = α·confidence + (1-α)·(1-Entropy)”的停滞检测来触发停止。 扩展特性： 
    - min_think_monitor_tokens：达到该长度后才开始监控； 
    - follow_think_end_token：触发截停时，自动强制输出一串 token； 
    - 若 input_ids 中已包含 think_end_token，则不再触发任何截断或停滞检测。
    - 维护增量 since_think 计数器。
    """

    def __init__(
        self,
        think_start_token: int = 151667,
        think_end_token: int = 151668,
        num_think_tokens: int = 200,
        min_think_monitor_tokens: int = 100,
        enable_stagnation_stop: bool = True,
        stagnation_patience: int = 20,
        stagnation_eps: float = 0.0001,
        combine_alpha: float = 0.8,
        eos_token_id: Optional[int] = None,
        follow_think_end_token: Optional[Sequence[int]] = None,
        # 新增
        entropy_topk: Optional[int] = 100,     # None 表示用全 vocab
        monitor_stride: int = 1,               # 每几步检查一次
    ):
        self.num_think_tokens = int(num_think_tokens)
        self.think_start_token = int(think_start_token)
        self.think_end_token = int(think_end_token)
        self.min_think_monitor_tokens = int(min_think_monitor_tokens)

        self.enable_stagnation_stop = bool(enable_stagnation_stop)
        self.stagnation_patience = int(stagnation_patience)
        self.stagnation_eps = float(stagnation_eps)
        self.combine_alpha = float(combine_alpha)
        self.eos_token_id = eos_token_id if eos_token_id is None else int(eos_token_id)
        self.follow_think_end_token = list(follow_think_end_token or [])

        self.entropy_topk = None if entropy_topk in (None, 0) else int(entropy_topk)
        self.monitor_stride = max(1, int(monitor_stride))

        # 批量状态（张量/列表）
        self._last_score_t = None     # (B,) tensor or None
        self._stagnation_cnt = None   # (B,) int tensor
        self._force_queues = []       # List[List[int]]
        self._seen_end = None         # (B,) bool tensor
        self._since_think = None      # (B,) int tensor
        self._step_mod = 0            # 全局步计数用于 stride
        self._init_scanned = False

        # 常量缓存
        self._logV = None

    def _ensure_state(self, batch_size: int, device, dtype):
        if self._last_score_t is None or self._last_score_t.numel() != batch_size:
            self._last_score_t = torch.full((batch_size,), float("nan"), device=device, dtype=torch.float32)
            self._stagnation_cnt = torch.zeros((batch_size,), device=device, dtype=torch.int32)
            self._force_queues = [[] for _ in range(batch_size)]
            self._seen_end = torch.zeros((batch_size,), device=device, dtype=torch.bool)
            self._since_think = torch.zeros((batch_size,), device=device, dtype=torch.int32)
            self._init_scanned = False
        if self._logV is None or self._logV.device != device or self._logV.dtype != torch.float32:
            # 用 float32 存常量即可
            self._logV = None  # 延迟到拿到 vocab_size 再设

    @torch.no_grad()
    def __call__(self, input_ids, logits: torch.Tensor) -> torch.Tensor:
        single = False
        if logits.dim() == 1:
            logits = logits.unsqueeze(0); single = True
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, V = logits.shape[0], logits.shape[-1]
        device = logits.device
        self._ensure_state(B, device, logits.dtype)
        if self._logV is None:
            self._logV = torch.tensor(math.log(max(V, 2)), device=device, dtype=torch.float32)

        # 1) 处理“强制队列”的样本（优先）
        force_mask = torch.tensor([len(q) > 0 for q in self._force_queues], device=device, dtype=torch.bool)
        if force_mask.any():
            idx = force_mask.nonzero(as_tuple=True)[0]
            # 对这些样本取队首 token，并强制
            forced = torch.tensor([self._force_queues[i].pop(0) for i in idx.tolist()],
                                  device=device, dtype=torch.long)
            logits[idx] = -float('inf')
            logits[idx, forced] = 0.0
            # 重置状态
            self._last_score_t[idx] = float('nan')
            self._stagnation_cnt[idx] = 0

        # 2) 首次调用：仅扫描一次输入，确定是否已含 think_end / think_start
        if not self._init_scanned:
            if isinstance(input_ids, torch.Tensor):
                ids = input_ids
                self._seen_end |= (ids == self.think_end_token).any(dim=-1)
                # 若提示里包含 think_start，则以“最后一个 think_start 之后”的剩余长度初始化 since_think
                # 否则从 0 开始（等着看到 start 再清零）
                pos = (ids == self.think_start_token).int()
                # 找到每行最后一次出现的位置
                any_start = pos.any(dim=-1)
                last_idx = torch.where(
                    any_start,
                    torch.argmax(pos.flip(dims=[-1]), dim=-1),  # 从末尾找第一个 1 的偏移
                    torch.zeros(B, device=device, dtype=torch.long)
                )
                # since_think = 序列长度 - 最后一次出现位置（若无则 0）
                seq_len = ids.shape[-1]
                since = torch.where(
                    any_start,
                    torch.tensor(seq_len, device=device) - (last_idx + 1),  # +1 因为 flip 偏移
                    torch.zeros(B, device=device)
                ).to(torch.int32)
                self._since_think = since
            self._init_scanned = True

        # 3) 若 last token 是 think_end，则标记（避免后续监控/截断）
        if isinstance(input_ids, torch.Tensor):
            last_tok = input_ids[:, -1]
            self._seen_end |= (last_tok == self.think_end_token)

        # 4) 更新 since_think 计数器：看到 think_start 清零；其他样本 +1
        if isinstance(input_ids, torch.Tensor):
            last_tok = input_ids[:, -1]
            is_start = (last_tok == self.think_start_token)
            self._since_think = torch.where(is_start, torch.zeros_like(self._since_think), self._since_think + 1)

        # 5) 已经 seen_end 的样本直接跳过
        active = (~self._seen_end)

        # 6) 长度限制：since_think >= num_think_tokens -> 强制 think_end
        over_len = active & (self._since_think >= self.num_think_tokens)
        if over_len.any():
            idx = over_len.nonzero(as_tuple=True)[0]
            logits[idx] = -float('inf')
            logits[idx, self.think_end_token] = 0.0
            for i in idx.tolist():
                self._force_queues[i] = list(self.follow_think_end_token)
            self._last_score_t[idx] = float('nan')
            self._stagnation_cnt[idx] = 0
            active = active & (~over_len)

        # 7) 停滞监控（分步执行以降低频率）
        self._step_mod = (self._step_mod + 1) % self.monitor_stride
        do_monitor = (
            self.enable_stagnation_stop
            and self._step_mod == 0
        )
        monitor_mask = active & (self._since_think >= self.min_think_monitor_tokens)

        if do_monitor and monitor_mask.any():
            idx = monitor_mask.nonzero(as_tuple=True)[0]
            row_logits = logits[idx].to(torch.float32)

            if self.entropy_topk is not None and self.entropy_topk < V:
                # top-k 近似
                top_vals, top_idx = torch.topk(row_logits, k=self.entropy_topk, dim=-1)
                log_probs_top = torch.log_softmax(top_vals, dim=-1)
                probs_top = torch.exp(log_probs_top)
                # max_prob（在 top-k 上）
                max_log_prob = log_probs_top.max(dim=-1).values
                max_prob = torch.exp(max_log_prob)
                # 熵（在 top-k 上近似）
                entropy = -(probs_top * log_probs_top).sum(dim=-1)
            else:
                # 全 vocab
                log_probs = torch.log_softmax(row_logits, dim=-1)
                probs = torch.exp(log_probs)
                max_log_prob = log_probs.max(dim=-1).values
                max_prob = torch.exp(max_log_prob)
                entropy = -(probs * log_probs).sum(dim=-1)

            H_norm = entropy / self._logV  # 标准化熵
            one_minus_entropy = 1.0 - torch.clamp(H_norm, 0.0, 1.0)

            alpha = torch.as_tensor(self.combine_alpha, device=device, dtype=torch.float32)
            score = alpha * max_prob + (1.0 - alpha) * one_minus_entropy

            last = self._last_score_t[idx]
            # last 为 NaN 的第一次监控，视作“有变化”，清零计数
            changed = torch.isnan(last) | (torch.abs(score - last) > self.stagnation_eps)
            self._stagnation_cnt[idx] = torch.where(changed, torch.zeros_like(self._stagnation_cnt[idx]), self._stagnation_cnt[idx] + 1)
            self._last_score_t[idx] = score

        else:
            # 不监控时，把本轮的 last_score 标成 NaN，避免错误累计
            idx = monitor_mask.nonzero(as_tuple=True)[0] if monitor_mask.any() else None
            if idx is not None and idx.numel() > 0:
                self._last_score_t[idx] = float('nan')
                self._stagnation_cnt[idx] = 0

        # 8) 停滞触发：>= patience -> 强制 think_end（或 eos）
        if self.enable_stagnation_stop:
            trigger = active & (self._stagnation_cnt >= self.stagnation_patience)
            if trigger.any():
                idx = trigger.nonzero(as_tuple=True)[0]
                # 优先 think_end
                logits[idx] = -float('inf')
                ok_idx = idx
                logits[ok_idx, self.think_end_token] = 0.0
                for i in ok_idx.tolist():
                    self._force_queues[i] = list(self.follow_think_end_token)
                self._stagnation_cnt[ok_idx] = 0

        return logits[0] if single else logits



# class ThinkLogitsProcessor:
#     """
#     限制 <think> 段长度；并基于“联合度量 = α·confidence + (1-α)·(1-Entropy)”的停滞检测来触发停止。
#     扩展特性：
#     - min_think_monitor_tokens：达到该长度后才开始监控；
#     - follow_think_end_token：触发截停时，自动强制输出一串 token；
#     - 若 input_ids 中已包含 think_end_token，则不再触发任何截断或停滞检测。
#     """

#     def __init__(
#         self,
#         think_start_token: int = 151667,
#         think_end_token: int = 151668,
#         num_think_tokens: int = 200,
#         min_think_monitor_tokens: int = 100,
#         enable_stagnation_stop: bool = True,
#         stagnation_patience: int = 20,
#         stagnation_eps: float = 0.001,
#         combine_alpha: float = 0.5,
#         eos_token_id: Optional[int] = None,
#         follow_think_end_token: Optional[Sequence[int]] = None,
#     ):
#         self.num_think_tokens = num_think_tokens
#         self.think_start_token = think_start_token
#         self.think_end_token = think_end_token
#         self.min_think_monitor_tokens = int(min_think_monitor_tokens)

#         self.enable_stagnation_stop = enable_stagnation_stop
#         self.stagnation_patience = stagnation_patience
#         self.stagnation_eps = stagnation_eps
#         self.combine_alpha = float(combine_alpha)
#         self.eos_token_id = eos_token_id
#         self.follow_think_end_token = list(follow_think_end_token or [])

#         # 批量状态
#         self._last_score = []        # List[Optional[float]]
#         self._stagnation_count = []  # List[int]
#         self._force_queue = []       # List[List[int]]

#     def _ensure_state_size(self, batch_size: int):
#         cur = len(self._last_score)
#         if cur < batch_size:
#             self._last_score += [None] * (batch_size - cur)
#             self._stagnation_count += [0] * (batch_size - cur)
#             self._force_queue += [[] for _ in range(batch_size - cur)]
#         elif cur > batch_size:
#             self._last_score = self._last_score[:batch_size]
#             self._stagnation_count = self._stagnation_count[:batch_size]
#             self._force_queue = self._force_queue[:batch_size]

#     @torch.no_grad()
#     def __call__(self, input_ids, logits: torch.Tensor) -> torch.Tensor:
#         single = False
#         if logits.dim() == 1:
#             logits = logits.unsqueeze(0)
#             single = True
#         if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
#             input_ids = input_ids.unsqueeze(0)

#         batch_size = logits.shape[0]
#         vocab_size = logits.shape[-1]
#         self._ensure_state_size(batch_size)

#         def safe_force_token(row_logits: torch.Tensor, token_id: int):
#             if 0 <= token_id < vocab_size:
#                 row_logits.fill_(float("-inf"))
#                 row_logits[token_id] = 0.0
#                 return True
#             return False

#         logV = math.log(max(vocab_size, 2))

#         for b in range(batch_size):
#             # (0) 若存在后续强制队列，优先输出并跳过其它逻辑
#             if self._force_queue[b]:
#                 token_id = self._force_queue[b].pop(0)
#                 safe_force_token(logits[b], token_id)
#                 self._last_score[b] = None
#                 self._stagnation_count[b] = 0
#                 continue

#             ids_b = input_ids[b] if isinstance(input_ids, torch.Tensor) else input_ids

#             # >>> 如果已经生成过 think_end_token，就不再截断 <<<
#             if self.think_end_token in ids_b:
#                 self._last_score[b] = None
#                 self._stagnation_count[b] = 0
#                 continue

#             tokens_since_think = ids_b.shape[-1] if isinstance(ids_b, torch.Tensor) else len(ids_b)

#             # (1) 长度限制
#             if tokens_since_think >= self.num_think_tokens:
#                 if safe_force_token(logits[b], self.think_end_token):
#                     self._force_queue[b] = list(self.follow_think_end_token)
#                 self._last_score[b] = None
#                 self._stagnation_count[b] = 0
#                 continue

#             # (2) 停滞监控
#             monitoring_allowed = (
#                 self.enable_stagnation_stop
#                 and (tokens_since_think >= self.min_think_monitor_tokens)
#             )

#             if monitoring_allowed:
#                 row_logits = logits[b]
#                 log_probs = torch.log_softmax(row_logits, dim=-1)
#                 probs = torch.exp(log_probs)

#                 max_prob = torch.max(probs).item()
#                 entropy = -torch.sum(probs * log_probs).item()
#                 H_norm = entropy / logV
#                 one_minus_entropy = 1.0 - max(0.0, min(1.0, H_norm))

#                 alpha = self.combine_alpha
#                 score = alpha * max_prob + (1.0 - alpha) * one_minus_entropy

#                 last = self._last_score[b]
#                 if (last is None) or (abs(score - last) > self.stagnation_eps):
#                     self._stagnation_count[b] = 0
#                 else:
#                     self._stagnation_count[b] += 1
#                 self._last_score[b] = score
#             else:
#                 self._last_score[b] = None
#                 self._stagnation_count[b] = 0

#             # (3) 停滞触发截断
#             if self.enable_stagnation_stop and self._stagnation_count[b] >= self.stagnation_patience:
#                 if safe_force_token(logits[b], self.think_end_token):
#                     self._force_queue[b] = list(self.follow_think_end_token)
#                     self._stagnation_count[b] = 0
#                 else:
#                     if self.eos_token_id is not None and safe_force_token(logits[b], self.eos_token_id):
#                         self._stagnation_count[b] = 0
#                     else:
#                         self._stagnation_count[b] = 0

#         return logits[0] if single else logits


# class ThinkLogitsProcessor:
#     """
#     限制 <think> 段长度；并在 max log prob 连续若干步几乎不变时触发停止。
#     """

#     def __init__(
#         self,
#         think_start_token: int = 151667,
#         think_end_token: int = 151668,
#         num_think_tokens: int = 200,
#         # —— 置信度（停滞）停止相关参数 ——
#         enable_confidence_stop: bool = True,
#         stagnation_patience: int = 20,   # 连续多少步“几乎不变”后停止
#         stagnation_eps: float = 1e-4,    # 认为“几乎不变”的阈值
#         eos_token_id: Optional[int] = None,  # 不在 <think> 段时使用的停止 token（可选）
#     ):
#         self.num_think_tokens = num_think_tokens
#         self.think_start_token = think_start_token
#         self.think_end_token = think_end_token

#         self.enable_confidence_stop = enable_confidence_stop
#         self.stagnation_patience = stagnation_patience
#         self.stagnation_eps = stagnation_eps
#         self.eos_token_id = eos_token_id

#         # 下面两个状态用于“max log prob 停滞”检测（按 batch index 记录）
#         self._last_max_logprob = []  # List[Optional[float]]
#         self._stagnation_count = []  # List[int]

#     # —— 工具：找最后一次出现的位置（支持 list / 1D tensor） ——
#     def _find_last(self, ids: Sequence[int], token: int) -> int:
#         if isinstance(ids, torch.Tensor):
#             ids_1d = ids.view(-1)
#             eq = (ids_1d == token).nonzero(as_tuple=False)
#             return int(eq[-1].item()) if eq.numel() > 0 else -1
#         else:
#             for i in range(len(ids)-1, -1, -1):
#                 if ids[i] == token:
#                     return i
#             return -1

#     def _ensure_state_size(self, batch_size: int):
#         # 动态扩展/收缩状态数组到 batch_size
#         cur = len(self._last_max_logprob)
#         if cur < batch_size:
#             self._last_max_logprob += [None] * (batch_size - cur)
#             self._stagnation_count += [0] * (batch_size - cur)
#         elif cur > batch_size:
#             # 收缩（通常 vLLM 会保持顺序不变，这里直接截断对应多余条目）
#             self._last_max_logprob = self._last_max_logprob[:batch_size]
#             self._stagnation_count = self._stagnation_count[:batch_size]

#     def __call__(self, input_ids, logits: torch.Tensor) -> torch.Tensor:
#         """
#         input_ids: (batch, seq_len) 或 1D
#         logits:    (batch, vocab)   或 1D
#         """
#         single = False
#         if logits.dim() == 1:
#             logits = logits.unsqueeze(0)
#             single = True

#         if isinstance(input_ids, torch.Tensor):
#             if input_ids.dim() == 1:
#                 input_ids = input_ids.unsqueeze(0)

#         batch_size = logits.shape[0]
#         vocab_size = logits.shape[-1]
#         self._ensure_state_size(batch_size)

#         # 提前做越界保护
#         def safe_force_token(row_logits: torch.Tensor, token_id: int):
#             if 0 <= token_id < vocab_size:
#                 row_logits.fill_(float('-inf'))
#                 row_logits[token_id] = 0.0  # 其他都是 -inf，softmax 后该 token 概率=1
#                 return True
#             return False

#         for b in range(batch_size):
#             ids_b = input_ids[b] if isinstance(input_ids, torch.Tensor) else input_ids

#             # 1) 先计算本步的最大 log prob（稳定比较）
#             if self.enable_confidence_stop:
#                 # 用 log_softmax 的最大值来衡量“置信度”
#                 max_logprob = torch.log_softmax(logits[b], dim=-1).max().item()
#                 last = self._last_max_logprob[b]
#                 if last is None or abs(max_logprob - last) > self.stagnation_eps:
#                     # 有明显变化：重置计数
#                     self._stagnation_count[b] = 0
#                 else:
#                     # 基本不变：计数+1
#                     self._stagnation_count[b] += 1
#                 self._last_max_logprob[b] = max_logprob

#             # 2) 判断是否处于 <think> 段，若达到上限则强制 </think>
#             last_think_pos = self._find_last(ids_b, self.think_start_token)
#             last_end_pos = self._find_last(ids_b, self.think_end_token)
#             in_think = (last_think_pos >= 0 and last_end_pos < last_think_pos)

#             if in_think and self.num_think_tokens is not None and self.num_think_tokens >= 0:
#                 tokens_since_think = (ids_b.shape[-1] if isinstance(ids_b, torch.Tensor) else len(ids_b)) - last_think_pos - 1
#                 if tokens_since_think >= self.num_think_tokens:
#                     safe_force_token(logits[b], self.think_end_token)
#                     # 到这里即可继续下一个样本（不再做置信度判断的强制停止，以免覆盖）
#                     continue

#             # 3) 若启用“置信度停滞”检测并达到阈值，则强制停止
#             if self.enable_confidence_stop and self._stagnation_count[b] >= self.stagnation_patience:
#                 # 优先：如果在 think 段内，用 </think> 作为停止 token
#                 if in_think and safe_force_token(logits[b], self.think_end_token):
#                     # 重置计数，避免反复触发
#                     self._stagnation_count[b] = 0
#                 else:
#                     # 否则尝试用 eos_token_id（如果提供了）
#                     if self.eos_token_id is not None and safe_force_token(logits[b], self.eos_token_id):
#                         self._stagnation_count[b] = 0
#                     else:
#                         # 没有可用的停止 token：不强制修改，但避免持续触发
#                         self._stagnation_count[b] = 0

#         return logits[0] if single else logits


# class ThinkLogitsProcessor:
#     """A logits processor that limit the number of thinking tokens."""
    
#     def __init__(self, think_start_token: int = 151667, think_end_token: int = 151668, num_think_tokens: int = 200):
#         """
#         Initialize the think logits processor.
        
#         Args:
#             tokenizer: The tokenizer used for the model
#             num_think_tokens: Maximum number of tokens allowed in thinking section
#         """
#         self.num_think_tokens = num_think_tokens
#         self.think_start_token = think_start_token
#         self.think_end_token = think_end_token
        
#     def __call__(
#         self,
#         input_ids: List[int],
#         logits: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Process the logits to enforce </think> token when needed.
        
#         Args:
#             input_ids: List of input token IDs.
#             logits: Tensor of logits for the next token.
            
#         Returns:
#             Processed logits tensor.
#         """
#         # Check if we're in a thinking section
#         if self.think_start_token in input_ids and self.think_end_token not in input_ids:
#             # Find the position of the last <think> token
#             think_start_pos = len(input_ids) - 1 - input_ids[::-1].index(self.think_start_token)
            
#             # Calculate number of tokens since <think>
#             tokens_since_think = len(input_ids) - think_start_pos - 1
            
#             # If we've reached the maximum thinking length, force </think>
#             if tokens_since_think >= self.num_think_tokens:
#                 # Set all other logits to -inf except for </think>
#                 logits = torch.full_like(logits, float('-inf'))
#                 logits[self.think_end_token] = 1.0
                
#         return logits 


# # NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
# def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
#     # remove the left padding in the prompt token_id
#     # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
#     # is not None else self.llm_engine.tokenizer.eos_token_id
#     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
#     token_ids = prompt_token_ids[non_pad_index:].tolist()
#     return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(
            logits_processors=[ThinkLogitsProcessor(num_think_tokens=config.num_think_tokens)],
            **kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            message = await self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = await self._execute_method(method, *args, **kwargs)
            await self.socket.send(pickle.dumps(result))

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0 if not ray_noset_visible_devices() else int(os.environ.get("RAY_LOCAL_RANK", 0))
        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep":
            return await self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return await self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address

    async def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    async def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    async def generate(self, *args, **kwargs):
        """Generate sequence with token-in-token-out."""
        raise NotImplementedError

    async def chat_completion(self, json_request):
        """OpenAI chat completion API."""
        raise NotImplementedError
