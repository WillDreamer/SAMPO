# recipe/shop_agent/llm_agent/base_llm.py
# 只保留 OpenAI Provider 和 ConcurrentLLM
# DeepSeek, Anthropic, Together 先注释掉，方便未来扩展
from dotenv import load_dotenv
load_dotenv(override=True)
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Union
import os
import asyncio
import time
from openai import AsyncOpenAI, BadRequestError

# Gemini SDK (optional)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ===================== 基础结构 =====================

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str


class LLMProvider:
    """Minimal abstract class (for interface consistency)"""
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        raise NotImplementedError


# ===================== ✅ OpenAI Provider =====================

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or missing in environment variables")

        self.client = AsyncOpenAI(api_key=self.api_key)
        print(f"[DEBUG] Using OpenAI model: {self.model_name}")
        print(f"[DEBUG] OPENAI_API_KEY prefix: {str(self.api_key)[:16] if self.api_key else None}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # 推理模型不支持 temperature/top_p 等采样参数
        unsupported_models = ['o1', 'o1-mini', 'o1-preview', 'o3-mini', 'gpt-5']
        if any(m in self.model_name for m in unsupported_models):
            kwargs.pop('top_p', None)
            kwargs.pop('temperature', None)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        text = response.choices[0].message.content
        return LLMResponse(content=text, model_name=response.model)


# ===================== ✅ DeepSeek Provider =====================
class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider implementation (OpenAI-compatible SDK)"""
    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided or missing in environment variables")

        # DeepSeek uses OpenAI-compatible endpoint but different base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        print(f"[DEBUG] Using DeepSeek model: {self.model_name}")
        print(f"[DEBUG] DEEPSEEK_API_KEY prefix: {str(self.api_key)[:16] if self.api_key else None}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        text = response.choices[0].message.content
        return LLMResponse(content=text, model_name=response.model)


# ===================== ✅ Gemini Provider =====================

class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation using google-genai SDK"""

    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. Install with: pip install google-genai"
            )

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not provided or missing in environment variables "
                "(GOOGLE_API_KEY or GEMINI_API_KEY)"
            )

        self.client = genai.Client(api_key=self.api_key)
        print(f"[DEBUG] Using Gemini model: {self.model_name}")
        print(f"[DEBUG] GOOGLE_API_KEY prefix: {str(self.api_key)[:16] if self.api_key else None}")

    def _convert_messages_to_gemini_format(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[Optional[str], List]:
        """
        Convert OpenAI-style messages to Gemini format.

        Args:
            messages: List of OpenAI-style message dicts with 'role' and 'content'

        Returns:
            Tuple of (system_instruction, contents list)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Gemini handles system prompts via system_instruction parameter
                system_instruction = content
            elif role == "assistant":
                # Gemini uses 'model' instead of 'assistant'
                contents.append(
                    types.Content(
                        role='model',
                        parts=[types.Part(text=content)]
                    )
                )
            else:  # user or any other role defaults to user
                contents.append(
                    types.Content(
                        role='user',
                        parts=[types.Part(text=content)]
                    )
                )

        return system_instruction, contents

    def _map_generation_kwargs(self, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Map OpenAI-style generation kwargs to Gemini GenerateContentConfig.

        Parameter mapping:
        - temperature -> temperature (same)
        - max_tokens -> max_output_tokens
        - max_completion_tokens -> max_output_tokens
        - top_p -> top_p (same)
        - top_k -> top_k (Gemini-specific, passthrough)
        """
        config_params = {}

        if 'temperature' in kwargs:
            config_params['temperature'] = kwargs['temperature']

        if 'max_tokens' in kwargs:
            config_params['max_output_tokens'] = kwargs['max_tokens']
        elif 'max_completion_tokens' in kwargs:
            config_params['max_output_tokens'] = kwargs['max_completion_tokens']

        if 'top_p' in kwargs:
            config_params['top_p'] = kwargs['top_p']

        if 'top_k' in kwargs:
            config_params['top_k'] = kwargs['top_k']

        return types.GenerateContentConfig(**config_params) if config_params else None

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Generate response using Gemini API (async).

        Args:
            messages: List of OpenAI-style message dicts
            **kwargs: Generation parameters (temperature, max_tokens, top_p, etc.)

        Returns:
            LLMResponse with content and model_name
        """
        # Convert messages to Gemini format
        system_instruction, contents = self._convert_messages_to_gemini_format(messages)

        # Map generation parameters
        config = self._map_generation_kwargs(kwargs)

        # Build request parameters
        request_params = {
            'model': self.model_name,
            'contents': contents,
        }

        # Add system instruction if present
        if system_instruction:
            if config is None:
                config = types.GenerateContentConfig()
            config.system_instruction = system_instruction

        if config:
            request_params['config'] = config

        # Make async API call using client.aio
        response = await self.client.aio.models.generate_content(**request_params)

        # Extract text from response
        text = ""
        if response and response.text:
            text = response.text
        elif response and response.candidates:
            # Fallback: extract from candidates if .text is not available
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text += part.text

        return LLMResponse(content=text, model_name=self.model_name)


# ===================== ⚙️ ConcurrentLLM =====================

class ConcurrentLLM:
    """Unified concurrent interface for multiple providers (OpenAI + DeepSeek)"""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_concurrency: int = 4
    ):
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            p = provider.lower()
            if p == "openai":
                self.provider = OpenAIProvider(model_name, api_key)
            elif p == "deepseek":
                self.provider = DeepSeekProvider(model_name, api_key)
            elif p == "gemini" or p == "google":
                self.provider = GeminiProvider(model_name, api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        self.max_concurrency = max_concurrency
        self._semaphore = None

    @property
    def semaphore(self):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)

    def run_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Batch inference with retry"""
        results = [None] * len(messages_list)
        position_map = {id(m): i for i, m in enumerate(messages_list)}

        current_batch = messages_list.copy()
        retry_count = 0
        max_retries = kwargs.get("max_retries", 3)

        while current_batch and retry_count < max_retries:
            async def process_batch():
                self._semaphore = None
                batch_results, failures = [], []
                tasks = [(m, asyncio.create_task(self.generate(m, **kwargs)))
                         for m in current_batch]
                for messages, task in tasks:
                    try:
                        resp = await task
                        idx = position_map[id(messages)]
                        batch_results.append((idx, {
                            "messages": messages,
                            "response": resp.content if resp.content else "",
                            "model": resp.model_name,
                            "success": True
                        }))
                    except BadRequestError as e:
                        # 内容审核拦截，记录为空响应，不重试
                        print(f"[WARN] Content moderation blocked: {e}")
                        idx = position_map[id(messages)]
                        batch_results.append((idx, {
                            "messages": messages,
                            "response": "",
                            "model": self.provider.model_name,
                            "success": False,
                            "error": "content_moderation_blocked"
                        }))
                    except Exception as e:
                        print(f"[DEBUG] API error: {e}")
                        traceback.print_exc()
                        failures.append(messages)
                return batch_results, failures

            batch_results, next_batch = asyncio.run(process_batch())
            for idx, res in batch_results:
                results[idx] = res

            if next_batch:
                retry_count += 1
                print(f"[WARN] {len(next_batch)} failed, retry {retry_count}/{max_retries}")
                current_batch = next_batch
                time.sleep(2)
            else:
                current_batch = []
                break

        return results, current_batch


# ===================== ✅ 简单测试 =====================
# if __name__ == "__main__":
#     llm = ConcurrentLLM(provider="openai", model_name="gpt-4o-mini", max_concurrency=3)
#     messages = [
#         [{"role": "user", "content": "Say 1"}],
#         [{"role": "user", "content": "Say 2"}],
#         [{"role": "user", "content": "Say 3"}],
#     ]
#     results, failed = llm.run_batch(messages, temperature=0.7)
#     for r in results:
#         if r:
#             print(r["response"])
