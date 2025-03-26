from monkey_patch.gpt2_attn import (
    gpt2_attention_custom_forward,
    gpt2_attention_kvbias_custom_forward,
)
from monkey_patch.llama_attn import (
    llama_attention_custom_forward,
    llama_attention_kvbias_custom_forward,
)


def get_custom_attn_func(model_name: str):
    if "gpt2" in model_name.lower():
        if "kvbias" in model_name.lower() or "kv_bias" in model_name.lower():
            return gpt2_attention_kvbias_custom_forward
        else:
            return gpt2_attention_custom_forward
    elif "llama" in model_name.lower():
        if "kvbias" in model_name.lower() or "kv_bias" in model_name.lower():
            return llama_attention_kvbias_custom_forward
        else:
            return llama_attention_custom_forward
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
