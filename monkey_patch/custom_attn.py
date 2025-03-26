from monkey_patch.gpt2_attn import (
    GPT2AttentionCustom_forward,
    GPT2AttentionKVBiasCustom_forward,
)
from monkey_patch.llama_attn import (
    LlamaAttentionCustom_forward,
    LlamaAttentionKVBiasCustom_forward,
)


def get_custom_attn_func(model_name: str):
    if "gpt2" in model_name.lower():
        if "kvbias" in model_name.lower() or "kv_bias" in model_name.lower():
            return GPT2AttentionKVBiasCustom_forward
        else:
            return GPT2AttentionCustom_forward
    elif "llama" in model_name.lower():
        if "kvbias" in model_name.lower() or "kv_bias" in model_name.lower():
            return LlamaAttentionKVBiasCustom_forward
        else:
            return LlamaAttentionCustom_forward
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
