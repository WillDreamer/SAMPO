from collections import Counter

from verl import DataProto
from recipe.math_agent.eval.llm_agent.base_llm import ConcurrentLLM
import numpy as np

class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # model_info adjust
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        # concurrent LLM
        self.llm = ConcurrentLLM(
            provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )

        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses,
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        if messages_list is not None:
            messages_list = messages_list.tolist()
            print("[INFO] Using messages_list with system prompt (OpenAI format)")
        else:
            # Fallback: construct single-turn chat messages from input_ids
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable to build fallback prompts")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] messages_list not provided; constructed fallback prompts from input_ids")

        # Debug: Print complete messages structure for first sample in batch
        if messages_list and len(messages_list) > 0:
            print("\n" + "="*80)
            print("[DEBUG] Complete messages sent to API (first sample in batch):")
            print("="*80)
            for i, msg in enumerate(messages_list[0]):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                print(f"\nMessage {i} [{role.upper()}]:")
                # Print first 300 chars of content
                if len(content) > 300:
                    print(f"  {content[:300]}...")
                    print(f"  ... (truncated, total length: {len(content)} chars)")
                else:
                    print(f"  {content}")
            print("="*80 + "\n")

        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        model_counts = Counter([r.get("model") for r in results if r])
        if model_counts:
            print(f"[INFO] API response model counts: {dict(model_counts)}")
            expected = self.config.model_config.model_name
            mismatched = {m: c for m, c in model_counts.items() if m and m != expected}
            if mismatched:
                print(f"[WARN] Response model mismatch expected={expected} counts={mismatched}")

        texts = [result["response"] for result in results]
        env_ids = lm_inputs.non_tensor_batch.get('env_ids')
        if env_ids is None:
            env_ids = np.arange(len(texts), dtype=object)
        group_ids = lm_inputs.non_tensor_batch.get('group_ids')
        if group_ids is None:
            group_ids = np.zeros(len(texts), dtype=object)
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            'response_texts': texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = lm_inputs.meta_info

        return lm_outputs
