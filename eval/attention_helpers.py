from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

logger = logging.get_logger(__name__)



from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast

import transformers
import sys
sys.path.append('../gui_kv')

from gui_kv_utils import init_pyramidkv, init_vlcache, init_snapkv, init_gui_kv
from ui_tars_utils import smart_resize

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def unrepeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Undo repeat_kv
    """
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # Reshape from (batch, num_attention_heads, slen, head_dim) 
    # to (batch, num_key_value_heads, n_rep, slen, head_dim)
    num_key_value_heads = num_attention_heads // n_rep
    hidden_states = hidden_states.reshape(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # Take only the first repetition to get back the original
    # (batch, num_key_value_heads, slen, head_dim)
    return hidden_states[:, :, 0, :, :]

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    
    # if not repeat_kv already, repeat_kv
    if key.shape[1] != query.shape[1]:
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
    else:
        key_states = key
        value_states = value

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Ensure mask is at least as long as key_states
        if attention_mask.size(-1) < key_states.shape[-2]:
            pad_len = key_states.shape[-2] - attention_mask.size(-1)
            pad_shape = list(attention_mask.shape)
            pad_shape[-1] = pad_len
            padding = attention_mask.new_zeros(pad_shape)
            attention_mask = torch.cat([attention_mask, padding], dim=-1)

        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


############# Qwen2.5-VL ###############

def qwen2_5_vl_vision_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
    
    
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    kwargs.pop("attention_mask", None)
    if self.config._attn_implementation == "flash_attention_2":
        # Flash Attention 2: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


def qwen2_5_vt_forward_layer(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, n_layer: int, **kwargs) -> torch.Tensor:
    """
    Forward pass through first n_layer layers of vision transformer (early exit).
    
    Args:
        hidden_states: Input tensor of shape (seq_len, hidden_size)
        grid_thw: Tensor of shape (num_images_or_videos, 3) with temporal, height, width info
        n_layer: Number of layers to process before early exit
        
    Returns:
        Hidden states after n_layer transformer blocks
    """
    # Apply patch embedding
    hidden_states = self.patch_embed(hidden_states)
    
    # Get rotary position embeddings
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    
    # Reshape and reorder based on window indices
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    
    # Prepare cu_seqlens for attention
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    
    # Process through first n_layer transformer blocks only
    for layer_num, blk in enumerate(self.blocks):
        if layer_num >= n_layer:
            break  # Early exit after n_layer layers
            
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
            
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    
    # TODO: consider skipping merger and reverse indexing for early exit
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]
    
    return hidden_states

def qwen2_5_vt_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states    


def get_residual_stream_contrast(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, n_layer: int = 1):
    """
    Encodes images into continuous embeddings that can be forwarded to the language model.

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    
    # get outputs from the final layer of the visual encoder
    image_embeds_final = self.visual(pixel_values, grid_thw=image_grid_thw)
    
    # get outputs from the early layers of the visual encoder
    image_embeds_early = self.visual.forward_early_exit(pixel_values, grid_thw=image_grid_thw, n_layer=n_layer)
    
    
    image_embeds_contrast = image_embeds_final - image_embeds_early
    image_embeds_contrast = image_embeds_early
    # image_embeds_contrast = image_embeds_final
    
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds_contrast = torch.split(image_embeds_contrast, split_sizes)
    return image_embeds_contrast

    
def qwen2_5_vl_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    
    
    
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        # sliding_window=self.sliding_window,
        **kwargs,
    )
    # print("attn_weights.shape", attn_weights.shape)
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and self.move_attention_to_cpu:
        # Detach from computation graph and move to CPU immediately
        attn_weights = attn_weights.detach().cpu()
        
        # Don't call empty_cache() here as it's too frequent and can slow things down
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    
    
    return attn_output, attn_weights, past_key_value

def qwen2_5_vl_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
    
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    
    # Debug: print self_attn_weights device
    if self_attn_weights is not None:
        print(f"self_attn_weights device: {self_attn_weights.device}")
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def qwen2_5_vl_text_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        print(f"layer_outputs[1] device: {layer_outputs[1].device}")
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
     

########## PyramidKV ##########
        
def qwen2_5_vl_attention_forward_PyramidKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    # when use_cache is True, hidden_states is the last token
    # when use_cache is False, hidden_states represents the full input sequence 
    
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    
    
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
        
        
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    
        
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        
        # no compress
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)
        # print(f"key_states.shape[-2] : {key_states.shape[-2]}, kv_seq_len: {kv_seq_len}")
        #compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        # sliding_window=self.sliding_window,
        **kwargs,
    )

    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and self.move_attention_to_cpu:
        # Detach from computation graph and move to CPU immediately
        attn_weights = attn_weights.detach().cpu()
        
        # Don't call empty_cache() here as it's too frequent and can slow things down
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value

########## SnapKV ##########
        
def qwen2_5_vl_attention_forward_SnapKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    # when use_cache is True, hidden_states is the last token
    # when use_cache is False, hidden_states represents the full input sequence 
    
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    
    
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
        
        
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    
        
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        
        # no compress
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_snapkv(self)
        # print(f"key_states.shape[-2] : {key_states.shape[-2]}, kv_seq_len: {kv_seq_len}")
        #compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        # sliding_window=self.sliding_window,
        **kwargs,
    )

    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and self.move_attention_to_cpu:
        # Detach from computation graph and move to CPU immediately
        attn_weights = attn_weights.detach().cpu()
        
        # Don't call empty_cache() here as it's too frequent and can slow things down
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


########## GUIKV ##########
        
def qwen2_5_vl_attention_forward_GUIKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    # when use_cache is True, hidden_states is the last token
    # when use_cache is False, hidden_states represents the full input sequence 
    
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    
    
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
        
        
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    
        
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        
        # no compress
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        self.config.vision_start_idx = self.vision_start_idx
        self.config.vision_end_idx = self.vision_end_idx
        # only do this if token information scores exists
        if hasattr(self, 'token_information_scores'):
            self.config.token_information_scores = self.token_information_scores
        
        
        init_gui_kv(self)
        # print(f"key_states.shape[-2] : {key_states.shape[-2]}, kv_seq_len: {kv_seq_len}")
        #compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups, hidden_states)
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        # sliding_window=self.sliding_window,
        **kwargs,
    )

    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and self.move_attention_to_cpu:
        # Detach from computation graph and move to CPU immediately
        attn_weights = attn_weights.detach().cpu()
        
        # Don't call empty_cache() here as it's too frequent and can slow things down
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value

########### VLCache ##########


def qwen2_5_vl_attention_forward_VLCache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    # when use_cache is True, hidden_states is the last token
    # when use_cache is False, hidden_states represents the full input sequence 
    
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    # print("begging of qwen2_5_vl_attention_forward_VLCache")
    # print("query_states.shape", query_states.shape)
    # print("key_states.shape", key_states.shape)
    # print("value_states.shape", value_states.shape)
    # print("attention_mask.shape", attention_mask.shape)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    # print(f"key_states.shape[-2]: {key_states.shape[-2]}")
    # print(f"past_key_value: {past_key_value.get_usable_length(kv_seq_len, self.layer_idx)}")
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
        # Reset past_key_value cache for fresh start in prefilling
        # safe_reset_cache(past_key_value)
        # if past_key_value is not None:
        #     print("usable length", past_key_value.get_usable_length(kv_seq_len, self.layer_idx))
        
    if past_key_value is not None:
        
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                # Only get usable length if we're not in the middle of prefilling reset
                if q_len == 1:  # Single token generation - use cache length
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
                # For prefilling (q_len > 1), kv_seq_len stays as current key_states length
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    
        
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        
        # no compress
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        
        
        # if prefilling, compress
        if key_states.shape[-2] == kv_seq_len and hasattr(self, 'gammas') and self.gammas is not None:
            
            # only compute sparsity during the first forward pass of pre-filling
            # compute sparsity for each head and layer
        
            # not the first forward pass of pre-filling, already computed gammas and betas
            # self.betas is a tensor of shape [bsz]
            self.kv_seq_len = kv_seq_len
            if hasattr(self, 'betas') and self.betas is not None:
                max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100)) * self.config.num_hidden_layers
                self.config.layer_budget = (max_capacity_prompt * self.betas).int()
                
                # print(f"self.config.layer_budget: {self.config.layer_budget}")
                # 
                init_vlcache(self, num_hidden_layers=self.config.num_hidden_layers)
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
                self.kept_indices = self.kv_cluster.kept_indices
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                print(f"Warning: betas not set for layer {self.layer_idx}, skipping compression")
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    # print("layer_index", self.layer_idx)
    # print("seen tokens", past_key_value._seen_tokens if past_key_value else None)
    # print("kv_seq_len", kv_seq_len)
    # print("q_len", q_len)
    # print("self.kv_seq_len", getattr(self, "kv_seq_len", "not set"))
    
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        # sliding_window=self.sliding_window,
        **kwargs,
    )
    
    # print("attn_ouput.shape", attn_output.shape)
    # print("attn_weights.shape", attn_weights.shape)
    # update gammas for for the first forward pass of pre-filling
    if (not hasattr(self, 'gammas') or self.gammas is None):
                
        self.gammas = []  # [bsz]
        # attn_weights size: [bsz, num_heads, total_length, total_length]
        # self.last_vision_indices: [bsz]
        for batch_idx, (this_attn_weights, this_last_vision_indices) in enumerate(zip(attn_weights, self.last_vision_indices)):
            # Get region of interest for all heads: [num_heads, roi_len, total_length]
            roi_weights = this_attn_weights[:, this_last_vision_indices:, :]
            
            if attention_mask is not None:
                # Process mask - handle different dimensions
                mask = attention_mask[batch_idx]
                while mask.dim() > 2:
                    mask = mask.squeeze(0)
                roi_mask = mask[this_last_vision_indices:, :] > -1e4
                
                # Apply mask to each head and compute gamma
                gammas = []
                for head_weights in roi_weights:
                    valid_elements = head_weights[roi_mask]
                    if valid_elements.numel() > 0:
                        sparsity = 1.0 - (valid_elements > 0.01).float().mean().item()
                    else:
                        sparsity = 0.0
                    gammas.append(sparsity)
                avg_gamma = torch.tensor(gammas).mean().item()
            else:
                # No mask - fully vectorized
                flat_weights = roi_weights.flatten(1)  # [num_heads, roi_len * total_length]
                sparsity_per_head = 1.0 - (flat_weights > 0.01).float().mean(dim=1)  # [num_heads]
                avg_gamma = sparsity_per_head.mean().item()
            
            self.gammas.append(avg_gamma)
        # make self.gammas a tensor
        self.gammas = torch.tensor(self.gammas)
        
        # do not update kv_seq_len here, it will be updated in the next forward pass
        # self.kv_seq_len = kv_seq_len
        # do not update key_states and value_states here, it will be updated in the next forward pass
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
    
        
        
    
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and self.move_attention_to_cpu:
        # Detach from computation graph and move to CPU immediately
        attn_weights = attn_weights.detach().cpu()
        
        # Don't call empty_cache() here as it's too frequent and can slow things down
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


def qwen2_5_vl_text_model_forward_VLCache(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        
        

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # Two forward pass during pre-filling
    # 1. first for computing sparsity-aware budget 
    # 2. second for caching the proper key-value pairs
    if cache_position[0] == 0:  # pre-filling phase
        
        gammas = []
        # do one more forward pass
        for decoder_layer in self.layers:
            
            # clear out gammas and betas 
            decoder_layer.self_attn.betas = None
            decoder_layer.self_attn.gammas = None
            
            # force use_cache to False for the first forward pass
            decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # each decoder_layer.self_attn.gammas is a tensor of shape [bsz]
            gammas.append(decoder_layer.self_attn.gammas)
        # print("text model gammas", gammas)
        # Stack gammas to shape [bsz, n_layer]
        gammas = torch.stack(gammas, dim=1)  # Shape: [bsz, n_layer]
        Z = (1 - gammas).sum(dim=1) # Shape: [bsz]
        # Compute beta for each layer: beta_i = (1 - gamma_i) / Z
        betas = (1 - gammas) / Z.unsqueeze(1)  # Shape: [bsz, n_layer]
        
        # Clip betas to be between 0.001 and 1
        betas = torch.clamp(betas, min=0.001, max=1.0)
        
        for decoder_layer in self.layers:
            decoder_layer.self_attn.betas = betas[:, decoder_layer.self_attn.layer_idx]
        
    total_budget = 0
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        
        layer_budget = decoder_layer.self_attn.config.layer_budget[0]
        total_budget += layer_budget
        
        
    # if inputs_embeds.shape[1] > 1:
    #     # check the legnth of kv cache based on past_key_value
    #     kv_cache_length = 0
    #     if past_key_values is not None:
    #         for i in range(len(self.layers)):
    #             kv_cache_length += past_key_values.get_seq_length(i)
    #         print(f"KV cache length: {kv_cache_length}")
    #     else:
    #         print("KV cache is None")
        
    #     total_input_tokens = inputs_embeds.shape[1] * len(self.layers)  # sequence length times number of layers
    #     percentage_cached = (total_budget / total_input_tokens) * 100 if total_input_tokens > 0 else 0
        
        
    #     print(f"total_input_tokens: {total_input_tokens}")
    #     print(f"total budget: {total_budget}")
    #     print(f"Percentage cached: {percentage_cached:.2f}%")
            
        
        
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    # print(f"original query_states.shape: {query_states.shape}, key_states.shape: {key_states.shape}, value_states.shape: {value_states.shape}")
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## PyramidKV for Qwen2 ##########

def qwen2_attention_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat KV states for grouped-query attention
    if hasattr(self, 'num_key_value_groups'):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, 
                self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
            )
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            
            
            original_len = key_states.shape[-2]
            compressed_len = key_states_compress.shape[-2]
            length_diff = original_len - compressed_len
            compression_ratio = compressed_len / original_len if original_len > 0 else 0
            
                # print(f"Layer {self.layer_idx}: Original key_states length: {original_len}, "
                #       f"Compressed length: {compressed_len}, "
                #       f"Difference: {length_diff}, "
                #       f"Compression ratio: {compression_ratio:.3f}")
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    if self.config._attn_implementation == "sdpa":
        key_states = unrepeat_kv(key_states, self.num_key_value_groups)
        value_states = unrepeat_kv(value_states, self.num_key_value_groups)
        
    # if self.layer_idx == 0:
    #     print(f"key_states.shape: {key_states.shape}, value_states.shape: {value_states.shape}")
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## SnapKV for Qwen2 ##########

def qwen2_attention_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # if not repeat_kv already, repeat_kv
    if key_states.shape[1] != query_states.shape[1]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_snapkv(self)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, 
                self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
            )
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            
            # Compare length difference between original and compressed key states
            original_len = key_states.shape[-2]
            compressed_len = key_states_compress.shape[-2]
            length_diff = original_len - compressed_len
            compression_ratio = compressed_len / original_len if original_len > 0 else 0
            # if self.layer_idx == 0:
            #     print(f"Layer {self.layer_idx}: Original key_states length: {original_len}, "
            #         f"Compressed length: {compressed_len}, "
            #         f"Difference: {length_diff}, "
            #         f"Compression ratio: {compression_ratio:.3f}")
            
            
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

        
    
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    
    # print(f"key_states.shape: {key_states.shape}, value_states.shape: {value_states.shape}")
        
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## GUIKV for Qwen2 ##########

def qwen2_attention_forward_GUIKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat KV states for grouped-query attention
    # if hasattr(self, 'num_key_value_groups'):
    #     key_states = repeat_kv(key_states, self.num_key_value_groups)
    #     value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        self.config.vision_start_idx = self.vision_start_idx
        self.config.vision_end_idx = self.vision_end_idx
        # only do this if token information scores exists
        if hasattr(self, 'token_information_scores'):
            self.config.token_information_scores = self.token_information_scores
        
        init_gui_kv(self)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, 
                self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1,
                hidden_states
            )
            
            if key_states.shape[1] != key_states_compress.shape[1]:
                key_states_compress = unrepeat_kv(key_states_compress, self.num_key_value_groups)
                value_states_compress = unrepeat_kv(value_states_compress, self.num_key_value_groups)
            key_states_compress = key_states_compress.contiguous(); value_states_compress = value_states_compress.contiguous()
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    if self.config._attn_implementation == "sdpa":
        key_states = unrepeat_kv(key_states, self.num_key_value_groups)
        value_states = unrepeat_kv(value_states, self.num_key_value_groups)
    # if self.layer_idx == 0:
    #     print( "key_states.shape", key_states.shape)
        
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## VLCache for Qwen2 ##########

def qwen2_attention_forward_VLCache(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    
    # This is a temp fix
    # if kwargs["use_cache"] and past_key_value is None:
    #     past_key_value = DynamicCache()
        
    # Reset kv_seq_len if we're in prefilling phase (q_len > 1)
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                if q_len == 1:  # Single token generation - use cache length
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat KV states for grouped-query attention
    if hasattr(self, 'num_key_value_groups'):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    # print("Hi!!! first forward pass")
    
    if past_key_value is not None:
        
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        
        # If prefilling, compress
        if key_states.shape[-2] == kv_seq_len and hasattr(self, 'gammas') and self.gammas is not None:
            self.kv_seq_len = kv_seq_len
            if hasattr(self, 'betas') and self.betas is not None:
                max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100)) * self.config.num_hidden_layers
                self.config.layer_budget = (max_capacity_prompt * self.betas).int()
                
                init_vlcache(self, num_hidden_layers=self.config.num_hidden_layers)
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                    key_states, query_states, value_states, attention_mask, 
                    self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
                )
                self.kept_indices = self.kv_cluster.kept_indices
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                print(f"Warning: betas not set for layer {self.layer_idx}, skipping compression")
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    # Update gammas for the first forward pass of pre-filling
    if (not hasattr(self, 'gammas') or self.gammas is None):
        self.gammas = []
        for batch_idx, (this_attn_weights, this_last_vision_indices) in enumerate(zip(attn_weights, self.last_vision_indices)):
            # Get region of interest for all heads
            roi_weights = this_attn_weights[:, this_last_vision_indices:, :]
            
            if attention_mask is not None:
                # Process mask
                mask = attention_mask[batch_idx]
                while mask.dim() > 2:
                    mask = mask.squeeze(0)
                roi_mask = mask[this_last_vision_indices:, :] > -1e4
                
                # Apply mask to each head and compute gamma
                gammas = []
                for head_weights in roi_weights:
                    valid_elements = head_weights[roi_mask]
                    if valid_elements.numel() > 0:
                        sparsity = 1.0 - (valid_elements > 0.01).float().mean().item()
                    else:
                        sparsity = 0.0
                    gammas.append(sparsity)
                avg_gamma = torch.tensor(gammas).mean().item()
            else:
                # No mask - fully vectorized
                flat_weights = roi_weights.flatten(1)
                sparsity_per_head = 1.0 - (flat_weights > 0.01).float().mean(dim=1)
                avg_gamma = sparsity_per_head.mean().item()
            
            self.gammas.append(avg_gamma)
        self.gammas = torch.tensor(self.gammas)
    
    # Move attention weights to CPU to avoid OOM
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



def qwen2_text_model_forward_VLCache(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # Two forward pass during pre-filling
    # 1. first for computing sparsity-aware budget 
    # 2. second for caching the proper key-value pairs
    if cache_position[0] == 0:  # pre-filling phase
        gammas = []
        # do one more forward pass
        for decoder_layer in self.layers:
            # clear out gammas and betas 
            decoder_layer.self_attn.betas = None
            decoder_layer.self_attn.gammas = None
            
            # force use_cache to False for the first forward pass
            layer_attn_mask = causal_mask_mapping[decoder_layer.attention_type]
            decoder_layer(
                hidden_states,
                attention_mask=layer_attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # each decoder_layer.self_attn.gammas is a tensor of shape [bsz]
            gammas.append(decoder_layer.self_attn.gammas)
        
        # Stack gammas to shape [bsz, n_layer]
        gammas = torch.stack(gammas, dim=1)  # Shape: [bsz, n_layer]
        Z = (1 - gammas).sum(dim=1) # Shape: [bsz]
        # Compute beta for each layer: beta_i = (1 - gamma_i) / Z
        betas = (1 - gammas) / Z.unsqueeze(1)  # Shape: [bsz, n_layer]
        
        # Clip betas to be between 0.001 and 1
        betas = torch.clamp(betas, min=0.001, max=1.0)
        
        for decoder_layer in self.layers:
            decoder_layer.self_attn.betas = betas[:, decoder_layer.self_attn.layer_idx]
        
    total_budget = 0
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        layer_budget = decoder_layer.self_attn.config.layer_budget[0]
        total_budget += layer_budget
    
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )

    
def safe_reset_cache(past_key_value):
    """Safely reset cache in a way that's compatible with accelerate"""
    if past_key_value is None:
        return
    
    try:
        # Method 1: Use built-in crop method if available
        if hasattr(past_key_value, 'crop'):
            past_key_value.crop(0)
            past_key_value._seen_tokens = 0
            return
        
        # Method 2: Check if we're using DynamicCache
        if hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
            # Clear the cache lists completely
            past_key_value.key_cache.clear()
            past_key_value.value_cache.clear()
            past_key_value._seen_tokens = 0
            return
            
        # Method 3: Fallback - try to reset _seen_tokens only
        past_key_value._seen_tokens = 0
        
    except Exception as e:
        print(f"Warning: Could not fully reset cache: {e}")
        # At minimum, reset seen tokens
        try:
            past_key_value._seen_tokens = 0
        except:
            pass

def disable_accelerate_hooks_for_vlcache(model):
    """Disable accelerate hooks that interfere with VLCache cache management"""
    try:
        hooks_removed = 0
        def remove_hooks(module):
            nonlocal hooks_removed
            if hasattr(module, '_hf_hook') and module._hf_hook is not None:
                # Store reference to the hook before removing
                old_hook = module._hf_hook
                module._hf_hook = None
                hooks_removed += 1
                print(f"Removed accelerate hook from {module.__class__.__name__}")
            
            # Recursively remove from child modules
            for child in module.children():
                remove_hooks(child)
        
        remove_hooks(model)
        print(f"Removed {hooks_removed} accelerate hooks for VLCache compatibility")
        
    except Exception as e:
        print(f"Error removing accelerate hooks: {e}")

def configure_accelerate_skip_attention(model):
    """Configure accelerate to skip moving attention tensors back to GPU"""
    try:
        hooks_configured = 0
        # Recursively find and configure all hooks in the model
        def configure_hooks(module):
            nonlocal hooks_configured
            if hasattr(module, '_hf_hook') and module._hf_hook is not None:
                if hasattr(module._hf_hook, 'skip_keys'):
                    existing_skip_keys = module._hf_hook.skip_keys
                    print(f"Found hook on {module.__class__.__name__} with skip_keys: {existing_skip_keys} (type: {type(existing_skip_keys)})")
                    
                    # Handle different types of skip_keys
                    if existing_skip_keys is None:
                        skip_keys = {'attentions'}
                    elif isinstance(existing_skip_keys, str):
                        skip_keys = {existing_skip_keys, 'attentions'}
                    elif isinstance(existing_skip_keys, (list, tuple)):
                        skip_keys = set(existing_skip_keys).union({'attentions'})
                    elif isinstance(existing_skip_keys, set):
                        skip_keys = existing_skip_keys.union({'attentions'})
                    else:
                        # Try to convert to set
                        try:
                            skip_keys = set(existing_skip_keys).union({'attentions'})
                        except:
                            skip_keys = {'attentions'}
                    
                    module._hf_hook.skip_keys = skip_keys
                    hooks_configured += 1
                    print(f"Configured accelerate hook for {module.__class__.__name__} with skip_keys: {skip_keys}")
            
            # Recursively configure child modules
            for child in module.children():
                configure_hooks(child)
        
        configure_hooks(model)
        print(f"Configured accelerate to skip attention tensors on {hooks_configured} hooks")
        
        # Alternative approach: patch the model's forward method to keep attention tensors on CPU
        if hasattr(model, 'model') and hasattr(model.model, 'forward'):
            original_forward = model.model.forward
            
            def patched_forward(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                
                # If outputs contain attention tensors, move them to CPU
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    cpu_attentions = tuple(
                        attn.cpu() if attn is not None and attn.device.type == 'cuda' else attn 
                        for attn in outputs.attentions
                    )
                    # Create a new output with CPU attention tensors
                    outputs = type(outputs)(
                        **{k: v if k != 'attentions' else cpu_attentions for k, v in outputs.items()}
                    )
                
                return outputs
            
            model.model.forward = patched_forward
            print("Patched model forward method to keep attention tensors on CPU")
            
    except Exception as e:
        print(f"Error configuring accelerate skip keys: {e}")
def set_attention_implementation(model, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for block in model.model.visual.blocks:
            block.attn._attn_implementation = args.attention_implementation
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.config._attn_implementation = args.attention_implementation
    else:
        raise NotImplementedError("Model not supported")
    
def set_move_attention_to_cpu(model, args):
    # set move_attention_to_cpu to True
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer_name = layer.__class__.__name__
            if args.do_visualization or args.do_attention_sparsity_analysis:
                layer.self_attn.move_attention_to_cpu = True
                
                print(f"set move_attention_to_cpu to True for layer {layer_name}")
            else:
                layer.self_attn.move_attention_to_cpu = False
                
                print(f"set move_attention_to_cpu to False for layer {layer_name}")
                
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer_name = layer.__class__.__name__
            if args.do_visualization or args.do_attention_sparsity_analysis:
                layer.self_attn.move_attention_to_cpu = True
                
                print(f"set move_attention_to_cpu to True for layer {layer_name}")
            else:
                layer.self_attn.move_attention_to_cpu = False
                
                print(f"set move_attention_to_cpu to False for layer {layer_name}")
    else:
        raise NotImplementedError("Model not supported")
        

def set_kv_cache_budget(model, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.kv_cache_budget = args.kv_cache_budget
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.kv_cache_budget = args.kv_cache_budget
    else:
        raise NotImplementedError("Model not supported")

# set_torch_profiler removed per user request

                        

def setup_vlcache_compatibility(model, disable_accelerate_hooks=True):
    """Setup model for VLCache compatibility by handling accelerate conflicts"""
    if disable_accelerate_hooks:
        disable_accelerate_hooks_for_vlcache(model)
        print("Disabled accelerate hooks for VLCache compatibility")
    else:
        # Alternative: configure accelerate to be more lenient
        configure_accelerate_skip_attention(model)
        print("Configured accelerate to skip attention tensors")

def set_last_vision_indices(model, last_vision_indices, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.last_vision_indices = last_vision_indices
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.last_vision_indices = last_vision_indices
    else:
        raise NotImplementedError("Model not supported")
        
        
def set_token_information_scores(model, token_information_scores, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.token_information_scores = token_information_scores
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.token_information_scores = token_information_scores
    else:
        raise NotImplementedError("Model not supported")
        
def set_vision_start_idx(model, vision_start_idx, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.vision_start_idx = vision_start_idx
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.vision_start_idx = vision_start_idx
    else:
        raise NotImplementedError("Model not supported")

def set_vision_end_idx(model, vision_end_idx, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.vision_end_idx = vision_end_idx
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.vision_end_idx = vision_end_idx
    else:
        raise NotImplementedError("Model not supported")

def set_alpha(model, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.config.alpha = args.alpha
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.config.alpha = args.alpha
    else:
        raise NotImplementedError("Model not supported")

def set_temperature(model, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.config.temperature = args.temperature
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.config.temperature = args.temperature
    else:
        raise NotImplementedError("Model not supported")


def set_window_size(model, args):
    if args.model_path == "ByteDance-Seed/UI-TARS-1.5-7B":
        for layer in model.model.language_model.layers:
            layer.self_attn.config.window_size = args.window_size
    elif args.model_path == "xlangai/OpenCUA-7B":
        for layer in model.language_model.model.layers:
            layer.self_attn.config.window_size = args.window_size
    else:
        raise NotImplementedError("Model not supported")
        
def get_residual_stream_cosine_similarity(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, n_layer: int = 1):
    """
    Compute cosine similarity between final and early layer embeddings from visual encoder.
    
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        n_layer: Number of early layers to use for comparison
        
    Returns:
        Tuple of tensors containing cosine similarities in [0, 1] range
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    
    # Get outputs from the final layer of the visual encoder
    image_embeds_final = self.visual(pixel_values, grid_thw=image_grid_thw)
    
    # Get outputs from the early layers of the visual encoder
    image_embeds_early = self.visual.forward_early_exit(pixel_values, grid_thw=image_grid_thw, n_layer=n_layer)
    
    # Compute cosine similarity along hidden dimension
    cosine_sim = F.cosine_similarity(image_embeds_final, image_embeds_early, dim=-1)
    
    # Rescale from [-1, 1] to [0, 1]
    cosine_sim_scaled = (1 - cosine_sim) / 2
    
    # Split by image sizes
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    cosine_similarities = torch.split(cosine_sim_scaled, split_sizes)
    
    return cosine_similarities


def compute_patch_shannon_entropy(patch):
    """
    Compute Shannon entropy for a single image patch.
    
    Args:
        patch: numpy array of shape (H, W) or (H, W, C) representing an image patch
        
    Returns:
        float: Shannon entropy value
    """
    # Convert to grayscale if needed
    if len(patch.shape) == 3:
        # Convert RGB to grayscale
        patch = np.dot(patch[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Flatten the patch
    patch_flat = patch.flatten()
    
    # Compute histogram
    hist, _ = np.histogram(patch_flat, bins=256, range=(0, 255))
    
    # Normalize to get probabilities
    hist = hist / hist.sum()
    
    # Remove zero probabilities to avoid log(0)
    hist = hist[hist > 0]
    
    # Compute Shannon entropy: H = -sum(p * log2(p))
    if len(hist) == 0:
        return 0.0
    
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def compute_patch_shannon_entropy_color(patch):
    """
    Compute Shannon entropy for a color image patch by calculating entropy
    for each channel separately and averaging them.
    
    Args:
        patch: numpy array of shape (H, W) or (H, W, C) representing an image patch
              Handles RGB (3 channels) and RGBA (4 channels) images
        
    Returns:
        float: Average Shannon entropy value across color channels (RGB only, alpha ignored)
    """
    # Handle grayscale images
    if len(patch.shape) == 2:
        # If grayscale, use the original function
        return compute_patch_shannon_entropy(patch)
    
    # For color images, compute entropy per channel
    entropies = []
    num_channels = patch.shape[2]
    
    # Handle different number of channels
    if num_channels == 1:
        # Single channel image (grayscale with channel dimension)
        return compute_patch_shannon_entropy(patch[:, :, 0])
    elif num_channels == 2:
        # Grayscale + alpha, only process grayscale channel
        channels_to_process = 1
    elif num_channels >= 3:
        # RGB or RGBA, process only RGB channels (ignore alpha)
        channels_to_process = 3
    else:
        # Unexpected number of channels, process what we have
        channels_to_process = num_channels
    
    for channel_idx in range(channels_to_process):
        # Extract channel
        channel_data = patch[:, :, channel_idx].flatten()
        
        # Compute histogram for this channel
        hist, _ = np.histogram(channel_data, bins=256, range=(0, 255))
        
        # Normalize to get probabilities
        hist = hist / hist.sum()
        
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        
        # Compute Shannon entropy for this channel
        if len(hist) == 0:
            channel_entropy = 0.0
        else:
            channel_entropy = -np.sum(hist * np.log2(hist))
        
        entropies.append(channel_entropy)
    
    # Return average entropy across channels
    return np.mean(entropies) if entropies else 0.0


def compute_patch_sobel_magnitude(patch):
    """
    Compute Sobel edge magnitude for a single image patch using vectorized operations.
    
    Args:
        patch: numpy array of shape (H, W) or (H, W, C) representing an image patch
        
    Returns:
        float: Average Sobel edge magnitude value
    """
    # Convert to grayscale if needed
    if len(patch.shape) == 3:
        # Convert RGB to grayscale
        patch = np.dot(patch[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Ensure patch is float
    patch = patch.astype(np.float32)
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Vectorized convolution using numpy's correlate2d equivalent
    # First, create sliding windows of the patch
    h, w = patch.shape
    
    # Pad the patch to handle borders
    padded_patch = np.pad(patch, pad_width=1, mode='edge')
    
    # Create view-based sliding window for vectorized operation
    # This creates a 4D array where each position contains the 3x3 window
    from numpy.lib.stride_tricks import as_strided
    
    # Get strides of padded patch
    s0, s1 = padded_patch.strides
    
    # Create sliding windows view
    windows = as_strided(padded_patch, 
                        shape=(h, w, 3, 3),
                        strides=(s0, s1, s0, s1))
    
    # Apply Sobel filters using vectorized operations
    grad_x = np.sum(windows * sobel_x, axis=(2, 3))
    grad_y = np.sum(windows * sobel_y, axis=(2, 3))
    
    # Compute edge magnitude
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return average edge magnitude
    return np.mean(edge_magnitude)


def rgb_to_lab(rgb_image):
    """
    Convert RGB image to CIELAB color space.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) with values in [0, 255]
        
    Returns:
        numpy array of shape (H, W, 3) in CIELAB color space
    """
    # Normalize RGB to [0, 1]
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    # Convert RGB to XYZ using sRGB transformation matrix
    # First, apply gamma correction
    mask = rgb_normalized > 0.04045
    rgb_linear = np.where(mask, 
                          ((rgb_normalized + 0.055) / 1.055) ** 2.4,
                          rgb_normalized / 12.92)
    
    # RGB to XYZ transformation matrix (for sRGB with D65 illuminant)
    rgb_to_xyz_matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    
    # Reshape for matrix multiplication
    h, w, _ = rgb_linear.shape
    rgb_flat = rgb_linear.reshape(-1, 3).T
    xyz_flat = rgb_to_xyz_matrix @ rgb_flat
    xyz = xyz_flat.T.reshape(h, w, 3)
    
    # Normalize by D65 illuminant values
    xyz[:, :, 0] /= 0.95047  # X
    xyz[:, :, 1] /= 1.00000  # Y
    xyz[:, :, 2] /= 1.08883  # Z
    
    # XYZ to LAB conversion
    def f(t):
        delta = 6.0 / 29.0
        mask = t > delta ** 3
        return np.where(mask, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)
    
    fx = f(xyz[:, :, 0])
    fy = f(xyz[:, :, 1])
    fz = f(xyz[:, :, 2])
    
    # Calculate LAB values
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1)


def compute_integral_image(channel):
    """
    Compute integral image for a single channel.
    
    Args:
        channel: 2D numpy array representing a single channel
        
    Returns:
        2D numpy array representing the integral image
    """
    return np.cumsum(np.cumsum(channel, axis=0), axis=1)


def get_box_sum(integral_img, x1, y1, x2, y2):
    """
    Get sum of values in a box using integral image.
    
    Args:
        integral_img: 2D integral image
        x1, y1: top-left corner (inclusive)
        x2, y2: bottom-right corner (inclusive)
        
    Returns:
        Sum of values in the box
    """
    h, w = integral_img.shape
    
    # Clamp coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    
    # Handle edge cases
    if x1 > 0 and y1 > 0:
        return integral_img[y2, x2] - integral_img[y2, x1-1] - integral_img[y1-1, x2] + integral_img[y1-1, x1-1]
    elif x1 > 0:
        return integral_img[y2, x2] - integral_img[y2, x1-1]
    elif y1 > 0:
        return integral_img[y2, x2] - integral_img[y1-1, x2]
    else:
        return integral_img[y2, x2]


def compute_center_surround_saliency(image, center_size=5, surround_size=15):
    """
    Compute center-surround saliency using integral images for fast computation.
    
    Args:
        image: numpy array of shape (H, W, 3) with RGB values in [0, 255]
        center_size: radius of center region in pixels
        surround_size: radius of surround region in pixels
        
    Returns:
        2D numpy array of saliency values
    """
    # Convert to LAB color space
    lab_image = rgb_to_lab(image)
    
    # Compute integral images for each LAB channel
    integral_L = compute_integral_image(lab_image[:, :, 0])
    integral_a = compute_integral_image(lab_image[:, :, 1])
    integral_b = compute_integral_image(lab_image[:, :, 2])
    
    h, w = image.shape[:2]
    saliency = np.zeros((h, w), dtype=np.float32)
    
    # Precompute areas
    center_area = (2 * center_size + 1) ** 2
    surround_area = (2 * surround_size + 1) ** 2 - center_area
    
    # Compute saliency for each pixel
    for y in range(h):
        for x in range(w):
            # Center box coordinates
            c_x1 = x - center_size
            c_y1 = y - center_size
            c_x2 = x + center_size
            c_y2 = y + center_size
            
            # Surround box coordinates
            s_x1 = x - surround_size
            s_y1 = y - surround_size
            s_x2 = x + surround_size
            s_y2 = y + surround_size
            
            # Get sums for each channel
            center_L = get_box_sum(integral_L, c_x1, c_y1, c_x2, c_y2)
            center_a = get_box_sum(integral_a, c_x1, c_y1, c_x2, c_y2)
            center_b = get_box_sum(integral_b, c_x1, c_y1, c_x2, c_y2)
            
            surround_L = get_box_sum(integral_L, s_x1, s_y1, s_x2, s_y2) - center_L
            surround_a = get_box_sum(integral_a, s_x1, s_y1, s_x2, s_y2) - center_a
            surround_b = get_box_sum(integral_b, s_x1, s_y1, s_x2, s_y2) - center_b
            
            # Compute averages
            center_avg = np.array([center_L / center_area, 
                                  center_a / center_area, 
                                  center_b / center_area])
            
            # Handle edge case where surround might be partially outside
            actual_surround_area = max(1, surround_area)  # Avoid division by zero
            surround_avg = np.array([surround_L / actual_surround_area,
                                    surround_a / actual_surround_area,
                                    surround_b / actual_surround_area])
            
            # Euclidean distance in LAB space
            saliency[y, x] = np.linalg.norm(center_avg - surround_avg)
    
    return saliency


def compute_patch_center_surround_saliency(patch, center_ratio=0.3, surround_ratio=0.7):
    """
    Compute center-surround saliency for a single image patch.
    
    Args:
        patch: numpy array of shape (H, W) or (H, W, C) representing an image patch
        center_ratio: ratio of patch size for center region (default 0.3)
        surround_ratio: ratio of patch size for surround region (default 0.7)
        
    Returns:
        float: Average center-surround saliency value for the patch
    """
    # Ensure we have RGB image
    if len(patch.shape) == 2:
        # Convert grayscale to RGB
        patch = np.stack([patch, patch, patch], axis=-1)
    
    # Ensure values are in [0, 255] range
    if patch.max() <= 1.0:
        patch = (patch * 255).astype(np.uint8)
    
    h, w = patch.shape[:2]
    
    # Calculate center and surround sizes based on patch dimensions
    center_size = max(1, int(min(h, w) * center_ratio / 2))
    surround_size = max(center_size + 1, int(min(h, w) * surround_ratio / 2))
    
    # Compute saliency
    saliency_map = compute_center_surround_saliency(patch, center_size, surround_size)
    
    # Return average saliency
    return np.mean(saliency_map)


def compute_patch_level_center_surround_saliency(image, patch_size=28, surround_radius=1):
    """
    Compute center-surround saliency at the patch level by comparing each patch
    to its surrounding patches in LAB color space.
    
    Args:
        image: numpy array of shape (H, W, 3) with RGB values
        patch_size: size of each patch in pixels
        surround_radius: radius of surrounding patches to consider (1 = 3x3, 2 = 5x5, etc.)
        
    Returns:
        1D numpy array of saliency values, one per patch
    """
    # Ensure image is in correct format
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[-1] == 4:
        # Remove alpha channel if present
        image = image[:, :, :3]
    
    # Ensure values are in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    h, w = image.shape[:2]
    
    # Calculate number of patches
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    if num_patches_h == 0 or num_patches_w == 0:
        # Image too small, return single saliency value
        return np.array([1.0])
    
    # Convert entire image to LAB
    lab_image = rgb_to_lab(image)
    
    # Extract all patches and compute average LAB color for each
    patch_colors = np.zeros((num_patches_h, num_patches_w, 3), dtype=np.float32)
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = i * patch_size
            end_h = min((i + 1) * patch_size, h)
            start_w = j * patch_size
            end_w = min((j + 1) * patch_size, w)
            
            patch = lab_image[start_h:end_h, start_w:end_w]
            patch_colors[i, j] = np.mean(patch.reshape(-1, 3), axis=0)
    
    # Compute saliency for each patch
    saliencies = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Get current patch color
            center_color = patch_colors[i, j]
            
            # Get surrounding patches
            surround_colors = []
            
            for di in range(-surround_radius, surround_radius + 1):
                for dj in range(-surround_radius, surround_radius + 1):
                    # Skip center patch
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = i + di, j + dj
                    
                    # Check boundaries
                    if 0 <= ni < num_patches_h and 0 <= nj < num_patches_w:
                        surround_colors.append(patch_colors[ni, nj])
            
            # If no surrounding patches (edge case), compare to itself (results in 0)
            if len(surround_colors) == 0:
                surround_avg = center_color
            else:
                # Compute average surround color
                surround_avg = np.mean(surround_colors, axis=0)
            
            # Compute Euclidean distance in LAB space
            saliency = np.linalg.norm(center_color - surround_avg)
            saliencies.append(saliency)
    
    return np.array(saliencies, dtype=np.float32)


def compute_spectral_residual_map(image, sigma=5.0):
    """
    Compute spectral residual saliency map for the entire image.
    
    Args:
        image: numpy array of shape (H, W) or (H, W, C) representing an image
        sigma: Standard deviation for Gaussian blur (default 5.0)
        
    Returns:
        numpy array: 2D saliency map of the same spatial dimensions as the input
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        # Remove alpha channel if present
        if image.shape[-1] == 4:
            image = image[...,:3]
        # Convert RGB to grayscale
        image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Ensure image is float
    image = image.astype(np.float32)
    
    # Apply 2D FFT to the entire image
    fft_image = fft2(image)
    
    # Get amplitude and phase
    amplitude = np.abs(fft_image)
    phase = np.angle(fft_image)
    
    # Compute log-amplitude spectrum (add epsilon to avoid log(0))
    log_amplitude = np.log(amplitude + 1e-10)
    
    # Apply Gaussian blur to log-amplitude to get the "expected" pattern
    log_amplitude_smooth = gaussian_filter(log_amplitude, sigma=sigma)
    
    # Calculate spectral residual
    spectral_residual = log_amplitude - log_amplitude_smooth
    
    # Reconstruct in frequency domain
    # Combine exp(residual) with original phase
    reconstructed_fft = np.exp(spectral_residual) * np.exp(1j * phase)
    
    # Apply inverse FFT to get saliency map
    saliency_map = np.abs(ifft2(reconstructed_fft))
    
    # Post-process: square to emphasize peaks
    saliency_map = saliency_map ** 2
    
    return saliency_map


def compute_token_information_scores(image, patch_size=28, factor=28, min_pixels=None, max_pixels=None, temperature=1.5, hidden_states=None, cosine_similarities=None):
    """
    Compute token information scores using cosine similarities, L2 norm of hidden states, 
    or fall back to uniform scores.
    
    Args:
        image: PIL Image or numpy array or path to image
        patch_size: Size of each patch (default 28 for common vision transformers)
        factor: Factor for smart_resize (default 28 from IMAGE_FACTOR)
        min_pixels: Minimum pixels for smart_resize
        max_pixels: Maximum pixels for smart_resize
        temperature: Temperature parameter for softmax scaling (default 1.5)
        hidden_states: torch.Tensor of shape [num_patches, hidden_dim] from vision encoder
        cosine_similarities: torch.Tensor of shape [num_patches] containing cosine similarities in [0, 1] range
        
    Returns:
        torch.Tensor: 1D tensor of information scores for each patch
    """
   
    # Otherwise, fall back to uniform scores
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # Get original dimensions
    if len(image_array.shape) == 3:
        orig_height, orig_width, _ = image_array.shape
    else:
        orig_height, orig_width = image_array.shape
    
    # Use smart_resize to get appropriate dimensions
    resized_height, resized_width = smart_resize(
        orig_height, orig_width, 
        factor=factor, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    
    # Resize the image
    if isinstance(image, Image.Image):
        resized_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_array = np.array(resized_image)
    else:
        # Use PIL for resizing numpy array
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        resized_image = pil_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_array = np.array(resized_image)
    
    # Calculate number of patches
    num_patches_h = resized_height // patch_size
    num_patches_w = resized_width // patch_size
    
    # If cosine similarities are provided, use them directly
    if cosine_similarities is not None:
        # Cosine similarities are already in [0, 1] range
        # Apply softmax with temperature for normalization
        token_information_scores = torch.softmax(cosine_similarities / temperature, dim=0)
        
        # Ensure the number of patches matches
        expected_num_patches = num_patches_h * num_patches_w
        actual_num_patches = len(token_information_scores)
        assert expected_num_patches == actual_num_patches, \
            f"Mismatch between expected patches ({expected_num_patches}) and cosine similarities length ({actual_num_patches})"
        
        return token_information_scores.float()
    
    # If hidden states are provided, use L2 norm approach
    elif hidden_states is not None:
        # L2 norm across hidden dimension
        importance_scores = torch.norm(hidden_states, dim=-1)  # [num_patches]
        # Standardize the importance scores before applying softmax
        importance_scores = (importance_scores - importance_scores.mean()) / (importance_scores.std() + 1e-8)
        # Apply softmax with temperature for normalization
        token_information_scores = torch.softmax(importance_scores / temperature, dim=0)
        # Ensure the number of patches matches the length of token information scores
        expected_num_patches = num_patches_h * num_patches_w
        actual_num_patches = len(token_information_scores)
        assert expected_num_patches == actual_num_patches, \
            f"Mismatch between expected patches ({expected_num_patches} = {num_patches_h}h x {num_patches_w}w) " \
            f"and actual token information scores length ({actual_num_patches})"
        return token_information_scores.float()
    
    
    ##### Sobel #####
    # Ensure we have at least one patch
    # if num_patches_h == 0 or num_patches_w == 0:
    #     # If image is too small, treat the whole image as one patch
    #     edge_magnitude = compute_patch_sobel_magnitude(resized_array)
    #     # Single patch gets score of 1.0 (full attention)
    #     return torch.tensor([1.0], dtype=torch.float32)
    
    # Compute Sobel edge magnitude for each patch
    # edge_magnitudes = []
    
    # for i in range(num_patches_h):
    #     for j in range(num_patches_w):
    #         # Extract patch
    #         start_h = i * patch_size
    #         end_h = min((i + 1) * patch_size, resized_height)
    #         start_w = j * patch_size
    #         end_w = min((j + 1) * patch_size, resized_width)
            
    #         patch = resized_array[start_h:end_h, start_w:end_w]
            
    #         # Compute Sobel edge magnitude for this patch
    #         edge_magnitude = compute_patch_sobel_magnitude(patch)
    #         edge_magnitudes.append(edge_magnitude)
    
    # # Convert to torch tensor
    # edge_magnitudes_tensor = torch.tensor(edge_magnitudes, dtype=torch.float32)
    
    ##### Entropy #####
    # Ensure we have at least one patch
    # if num_patches_h == 0 or num_patches_w == 0:
    #     # If image is too small, treat the whole image as one patch
    #     entropy = compute_patch_shannon_entropy(resized_array)
    #     # Single patch gets score of 1.0 (full attention)
    #     return torch.tensor([1.0], dtype=torch.float32)
    
    # # Compute entropy for each patch
    # entropies = []
    
    # for i in range(num_patches_h):
    #     for j in range(num_patches_w):
    #         # Extract patch
    #         start_h = i * patch_size
    #         end_h = min((i + 1) * patch_size, resized_height)
    #         start_w = j * patch_size
    #         end_w = min((j + 1) * patch_size, resized_width)
            
    #         patch = resized_array[start_h:end_h, start_w:end_w]
            
    #         # Compute entropy for this patch
    #         entropy = compute_patch_shannon_entropy(patch)
    #         entropies.append(entropy)
    
    # # Convert to torch tensor
    # entropies_tensor = torch.tensor(entropies, dtype=torch.float32)
    # Original entropy-based computation (commented out)
    # Apply softmax with temperature scaling: S_i = exp(E_i/T) / sum_j(exp(E_j/T))
    # token_information_scores = torch.softmax(entropies_tensor / temperature, dim=0)
    
    # Apply min-max scaling: S_i = (E_i - min(E)) / (max(E) - min(E))
    # min_entropy = torch.min(entropies_tensor)
    # max_entropy = torch.max(entropies_tensor)
    
    # # Handle edge case where all entropies are the same
    # if max_entropy == min_entropy:
    #     token_information_scores = torch.ones_like(entropies_tensor) / len(entropies_tensor)
    # else:
    #     token_information_scores = (entropies_tensor - min_entropy) / (max_entropy - min_entropy)
        
    #     # Apply square to reduce high scores and increase contrast
    #     token_information_scores = token_information_scores ** 1
    
    # Previous binary mask approach (commented out)
    # Compute 40th percentile threshold (using edge magnitudes instead of entropies)
    # percentile_40 = torch.quantile(edge_magnitudes_tensor, 0.2)
    # Create binary mask: 1 for edge magnitude > 40th percentile, 0 otherwise
    # token_information_scores = (edge_magnitudes_tensor > percentile_40).float()
    
    # Apply min-max scaling: S_i = (E_i - min(E)) / (max(E) - min(E))
    # min_magnitude = torch.min(edge_magnitudes_tensor)
    # max_magnitude = torch.max(edge_magnitudes_tensor)
    
    # # Handle edge case where all magnitudes are the same
    # if max_magnitude == min_magnitude:
    #     # If all edge magnitudes are the same, give equal weight to all patches
    #     token_information_scores = torch.ones_like(edge_magnitudes_tensor) / len(edge_magnitudes_tensor)
    # else:
    #     # Min-max scaling to [0, 1] range
    #     token_information_scores = (edge_magnitudes_tensor - min_magnitude) / (max_magnitude - min_magnitude)
    
    ##### Center-surround saliency #####
    # # Compute patch-level center-surround saliency
    # saliencies = compute_patch_level_center_surround_saliency(resized_array, patch_size=patch_size, surround_radius=2)
    # 
    # # Convert to torch tensor
    # saliencies_tensor = torch.tensor(saliencies, dtype=torch.float32)
    # 
    # # Apply softmax with temperature scaling: S_i = exp(S_i/T) / sum_j(exp(S_j/T))
    # token_information_scores = torch.softmax(saliencies_tensor / temperature, dim=0)
    
    ##### Spectral Residual #####
    # # Compute spectral residual saliency map for the entire image
    # saliency_map = compute_spectral_residual_map(resized_array)
    # 
    # # Ensure we have at least one patch
    # if num_patches_h == 0 or num_patches_w == 0:
    #     # If image is too small, treat the whole image as one patch
    #     # Single patch gets score of 1.0 (full attention)
    #     return torch.tensor([1.0], dtype=torch.float32)
    # 
    # # Extract average saliency value for each patch from the full saliency map
    # spectral_saliencies = []
    # 
    # for i in range(num_patches_h):
    #     for j in range(num_patches_w):
    #         # Extract patch region from saliency map
    #         start_h = i * patch_size
    #         end_h = min((i + 1) * patch_size, resized_height)
    #         start_w = j * patch_size
    #         end_w = min((j + 1) * patch_size, resized_width)
    #         
    #         # Get the saliency values for this patch region
    #         patch_saliency = saliency_map[start_h:end_h, start_w:end_w]
    #         
    #         # Compute average saliency for this patch
    #         avg_saliency = np.mean(patch_saliency)
    #         spectral_saliencies.append(avg_saliency)
    # 
    # # Convert to torch tensor
    # spectral_saliencies_tensor = torch.tensor(spectral_saliencies, dtype=torch.float32)
    # 
    # # Standardize to zero mean and unit variance
    # mean = spectral_saliencies_tensor.mean()
    # std = spectral_saliencies_tensor.std() + 1e-8  # avoid division by zero
    # standardized = (spectral_saliencies_tensor - mean) / std
    # 
    # # Apply softmax with temperature scaling: S_i = exp(S_i/T) / sum_j(exp(S_j/T))
    # token_information_scores = torch.softmax(standardized / temperature, dim=0)
    
    # Fallback: Return uniform scores
    # Calculate total number of patches
    total_patches = num_patches_h * num_patches_w
    
    # Ensure we have at least one patch
    if total_patches == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    
    # Return uniform scores for all patches
    token_information_scores = torch.ones(total_patches, dtype=torch.float32) / total_patches
    
    return token_information_scores 


def replace_qwen2_5_vl(kv_cache_mode="original", disable_accelerate_for_vlcache=False):
    
    assert kv_cache_mode in ["original", "pyramid_kv", "vl_cache", "snap_kv", "gui_kv"]
    
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = qwen2_5_vl_vision_attention_forward    
    if kv_cache_mode == "original":
        
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = qwen2_5_vl_decoder_layer_forward
    # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = qwen2_5_vl_text_model_forward        
    elif kv_cache_mode == "pyramid_kv":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_PyramidKV
    elif kv_cache_mode == "snap_kv":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_SnapKV
    elif kv_cache_mode == "gui_kv":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_GUIKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward_early_exit = qwen2_5_vt_forward_layer
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_residual_stream_contrast = get_residual_stream_contrast
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_residual_stream_cosine_similarity = get_residual_stream_cosine_similarity
    elif kv_cache_mode == "vl_cache":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_VLCache
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = qwen2_5_vl_text_model_forward_VLCache
        
        
        if disable_accelerate_for_vlcache:
            print("Warning: VLCache mode with accelerate hooks disabled. This may affect memory management.")
            
            
            
def replace_opencua(kv_cache_mode="original", disable_accelerate_for_vlcache=False):
    
    assert kv_cache_mode in ["original", "pyramid_kv", "vl_cache", "snap_kv", "gui_kv"]
    
    if kv_cache_mode == "original":
        pass
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward
    elif kv_cache_mode == "pyramid_kv":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_PyramidKV
    elif kv_cache_mode == "snap_kv":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_SnapKV
    elif kv_cache_mode == "gui_kv":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_GUIKV
    elif kv_cache_mode == "vl_cache":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_VLCache
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_text_model_forward_VLCache
        
        if disable_accelerate_for_vlcache:
            print("Warning: VLCache mode with accelerate hooks disabled for OpenCUA. This may affect memory management.")        
