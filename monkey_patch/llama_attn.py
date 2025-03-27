import math
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def llama_attention_custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    ##################################################################
    self.input_attention = hidden_states.detach().cpu().clone()
    ##################################################################

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    ##################################################################
    self.query_states = query_states.detach().cpu().clone()
    self.key_states = key_states.detach().cpu().clone()
    self.value_states = value_states.detach().cpu().clone()
    # ###################################################

    # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ###################################################
    self.attn_logits = attn_weights.detach().cpu().clone()
    # ###################################################

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # ###################################################
    self.attn_probs = attn_weights.detach().cpu().clone()
    self.value_states_post = value_states.detach().cpu().clone()
    # ###################################################

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    ##################################################################
    self.intermediate_attention = attn_output.detach().cpu().clone()
    ##################################################################

    attn_output = self.o_proj(attn_output)

    ##################################################################
    self.output_attention = attn_output.detach().cpu().clone()
    ##################################################################

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attention_kvbias_custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    ##################################################################
    self.input_attention = hidden_states.detach().cpu().clone()
    ##################################################################
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    ##################################################################
    self.query_states = query_states.detach().cpu().clone()
    self.key_states = key_states.detach().cpu().clone()
    self.value_states = value_states.detach().cpu().clone()
    # ###################################################

    # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    ###########################################################
    k_bias = self.k_bias.weight.repeat(bsz, 1, 1, 1)  # (B, num_heads, 1, dim // num_heads)
    v_bias = self.v_bias.weight.repeat(bsz, 1, 1, 1)

    key_states = (
        torch.cat((k_bias, key_states), dim=2).to(query_states.dtype).to(query_states.device)
    )
    value_states = (
        torch.cat((v_bias, value_states), dim=2).to(query_states.dtype).to(query_states.device)
    )

    if attention_mask is not None:
        attention_mask = None
    ###########################################################

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # ###################################################
    self.attn_logits = attn_weights.detach().cpu().clone()
    # ###################################################

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # ###################################################
    self.attn_probs = attn_weights.detach().cpu().clone()
    self.value_states_post = value_states.detach().cpu().clone()
    # ###################################################

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    ##################################################################
    self.intermediate_attention = attn_output.detach().cpu().clone()
    ##################################################################

    attn_output = self.o_proj(attn_output)

    ##################################################################
    self.output_attention = attn_output.detach().cpu().clone()
    ##################################################################

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
