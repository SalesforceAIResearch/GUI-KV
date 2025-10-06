"""
Adapted from Pyramid KV's repo: https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py
"""

import torch
import math
import torch.nn.functional as F
import torch.nn as nn

from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

class DynamicCacheSplitHeadFlatten(Cache):
    """
    Adapted from https://github.com/FFY0/AdaKV.
    """
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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

def merge_kv(key_states, value_states, indices, window_size, merge):
    """Merge KV cache using LOOK-M methods."""
    bsz, num_heads, k_len, head_dim = key_states.shape

    # KV-selected
    selected_keys = key_states.gather(dim=2, index=indices)
    selected_values = value_states.gather(dim=2, index=indices)

    # KV-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()
    selected_indices_flattened = indices.flatten()
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len)
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # KV-recent
    recent_keys = key_states[:, :, -window_size:, :]

    # Prepare for merge
    k_hh_pruned = drop_keys
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)
    v_hh_pruned = drop_values
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)

    # Similarity matrix (cosine)
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2))
    max_values, max_indices = similarity.max(dim=-1)

    # Pivot merge
    if merge == "pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected) / 2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True)
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected) / 2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    return k_hh_recent, v_hh_recent


class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices

            past_seq_len = key_states.size(2) - self.window_size
            indices = torch.clamp(indices, min=0, max=past_seq_len - 1)

            self.kept_indices = indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices

            past_seq_len = key_states.size(2) - self.window_size
            indices = torch.clamp(indices, min=0, max=past_seq_len - 1)

            self.kept_indices = indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            self.kept_indices = indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states


class VLCacheCluster():
    def __init__(self, num_hidden_layers = 32, last_vision_indices = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', layer_idx=None, layer_budget=None, merge=None):
        self.num_hidden_layers = num_hidden_layers
        self.steps = -1
        self.last_vision_indices = last_vision_indices
        self.max_capacity_prompt = max_capacity_prompt
        
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.layer_budget = layer_budget
        self.layer_idx = layer_idx

    def reset(self, last_vision_indices = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, layer_budget=None):
        self.last_vision_indices = last_vision_indices
        self.max_capacity_prompt = max_capacity_prompt
        
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.layer_budget = layer_budget

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        last_vision_index = self.last_vision_indices[0]
        seq_len = key_states.shape[-2]
        self.window_size = seq_len - last_vision_index

        if self.layer_budget[0] > self.window_size:
            max_capacity_prompt = self.layer_budget - self.window_size
        else:
            max_capacity_prompt = torch.ones_like(self.layer_budget)
            self.window_size = self.layer_budget[0] - 1

        max_capacity_prompt = torch.clamp(max_capacity_prompt, min=0)

        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')
        
        # Handle batch-wise max_capacity_prompt
        key_states_list = []
        value_states_list = []
        
        for batch_idx in range(bsz):
            batch_capacity = max_capacity_prompt[batch_idx].item()
            
            batch_attn_cache = attn_cache[batch_idx:batch_idx+1]
            if batch_capacity > 0 and batch_capacity <= batch_attn_cache.shape[-1]:
                indices = batch_attn_cache.topk(batch_capacity, dim=-1).indices
            else:
                if batch_capacity <= 0:
                    indices = torch.empty(batch_attn_cache.shape[:-1] + (0,), dtype=torch.long, device=batch_attn_cache.device)
                else:
                    indices = torch.arange(batch_attn_cache.shape[-1], device=batch_attn_cache.device).unsqueeze(0).unsqueeze(0).expand(batch_attn_cache.shape[:-1] + (-1,))
            self.kept_indices = indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            batch_key_states = key_states[batch_idx:batch_idx+1]
            batch_value_states = value_states[batch_idx:batch_idx+1]
            
            if self.merge is not None:
                batch_key_states, batch_value_states = merge_kv(batch_key_states, batch_value_states, indices, self.window_size, self.merge)
                key_states_list.append(batch_key_states)
                value_states_list.append(batch_value_states)
            else:
                k_past_compress = batch_key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                v_past_compress = batch_value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
                k_cur = batch_key_states[:, :, -self.window_size:, :]
                v_cur = batch_value_states[:, :, -self.window_size:, :]
                batch_key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                batch_value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                key_states_list.append(batch_key_states)
                value_states_list.append(batch_value_states)

        key_states = torch.cat(key_states_list, dim=0)
        value_states = torch.cat(value_states_list, dim=0)
        return key_states, value_states

class GUIKVCluster():
    """
    GUI-KV: Accelerate MLLMs for GUI Agents with Adaptive KV Cache Compression

    This class implements the GUI-KV method for compressing KV caches in multimodal language models
    specifically optimized for GUI (Graphical User Interface) agents. GUI-KV exploits both spatial
    and temporal redundancies in GUI screenshots to reduce memory usage and computational costs.

    Key Features:
    1. Spatial Saliency Guidance: Uses L2 norms of hidden states to identify semantically important
       visual tokens, preserving critical GUI elements like buttons, text fields, and interactive elements.

    2. Temporal Redundancy Detection: Uses QR decomposition to identify redundant information across
       multiple GUI screenshots in a sequence, removing duplicate visual content from previous frames.

    3. Adaptive Budget Allocation: Dynamically allocates KV cache budget based on the relative importance
       of different spatial regions and temporal frames.

    Paper: https://arxiv.org/abs/2510.00536
    """

    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4, vision_start_idx=None, vision_end_idx=None, alpha=0.1, temperature=1.0):
        """
        Initialize GUI-KV cluster for efficient KV cache compression in GUI agents.

        Args:
            window_size: Size of the recent token window to always preserve (typically the most recent tokens
                        that are crucial for maintaining context continuity)
            max_capacity_prompt: Maximum total KV cache capacity after compression. This determines the
                                memory budget for storing key-value pairs.
            kernel_size: Kernel size for pooling attention weights when aggregating importance scores
                        across neighboring tokens (smooths the importance distribution)
            pooling: Pooling method for attention aggregation ('avgpool' or 'maxpool')
                    - 'avgpool': Averages importance scores in local neighborhoods
                    - 'maxpool': Takes maximum importance in local neighborhoods
            merge: Optional merge strategy for combining similar KV pairs (e.g., 'pivot' merging)
            recent_size: Size of recent context (legacy parameter for backward compatibility)
            ratio: Ratio for pruning (legacy parameter for backward compatibility)
            vision_start_idx: List of starting indices for each image/screenshot in the sequence.
                             Essential for identifying boundaries between different GUI frames.
            vision_end_idx: List of ending indices for each image/screenshot in the sequence.
                           Used together with vision_start_idx to isolate individual images.
            alpha: Weight parameter for combining attention scores with spatial importance scores.
                  - Higher alpha (e.g., 0.3): More weight on spatial saliency from hidden states
                  - Lower alpha (e.g., 0.1): More weight on learned attention patterns
                  Controls the balance between model's attention and spatial importance guidance.
            temperature: Temperature for softmax normalization of spatial importance scores.
                        - Higher temperature (>1.0): Smoother importance distribution (less aggressive pruning)
                        - Lower temperature (<1.0): Sharper importance distribution (more aggressive pruning)
                        - Default 1.0: Standard softmax behavior
        """
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio
        self.alpha = alpha  # Spatial importance weight
        self.temperature = temperature  # Softmax temperature for importance scores
        self.vision_start_idx = vision_start_idx  # Track where each image starts
        self.vision_end_idx = vision_end_idx  # Track where each image ends

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio = 0.4, alpha=0.1, temperature=1.0, vision_start_idx=None, vision_end_idx=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size
        self.alpha = alpha
        self.temperature = temperature
        self.vision_start_idx = vision_start_idx
        self.vision_end_idx = vision_end_idx

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, hidden_states=None):
        """
        Core method for GUI-KV cache compression that combines spatial and temporal redundancy detection.

        This method performs intelligent KV cache compression by:
        1. Computing attention-based importance scores
        2. Augmenting with spatial saliency from hidden states (if provided)
        3. Detecting temporal redundancy across multiple GUI screenshots using QR decomposition
        4. Selectively preserving the most important KV pairs within the memory budget

        Args:
            key_states: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            query_states: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value_states: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            attention_mask: Attention mask for the sequence
            num_key_value_groups: Number of key-value groups for grouped-query attention
            hidden_states: Optional hidden states tensor for computing spatial importance.
                          Shape: (batch_size, seq_len, hidden_dim)
                          When provided, enables spatial saliency guidance.

        Returns:
            Tuple of (compressed_key_states, compressed_value_states) with reduced sequence length
        """
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # Handle grouped-query attention by repeating KV pairs
        if key_states.shape[1] != query_states.shape[1]:
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

        # Calculate compression budget as a ratio of max capacity to current sequence length
        budget = float(self.max_capacity_prompt / q_len)

        # No compression needed if sequence is within capacity
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            # Step 1: Compute attention-based importance scores
            # Use recent window queries to compute attention to all past tokens
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            # Create causal mask for recent window to prevent attending to future tokens
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            # Apply causal mask to attention weights
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            # Normalize attention weights with softmax
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Aggregate attention scores from recent queries to past tokens
            # This gives us importance scores for each position in the past sequence
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)

            # Apply pooling to smooth importance scores across neighboring tokens
            # This helps maintain local context coherence
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            # Step 2: Spatial Saliency Guidance (if hidden states are provided)
            # This is a key innovation of GUI-KV that leverages spatial importance in GUI elements
            if hidden_states is not None:
                # Extract hidden states for the current (most recent) image
                visual_hidden_states = hidden_states[:, self.vision_start_idx[-1]: self.vision_end_idx[-1], :]

                # Compute L2 norm of hidden states as spatial importance scores
                # Higher L2 norm indicates more semantically important visual tokens
                # This helps preserve critical GUI elements like buttons, text, interactive widgets
                importance_scores = torch.norm(visual_hidden_states, p=2, dim=-1)

                # Normalize importance scores using standardization and temperature-controlled softmax
                # Standardization ensures consistent scale across different batches and images
                normalized_scores = torch.zeros_like(importance_scores)
                for batch_idx in range(importance_scores.shape[0]):
                    batch_scores = importance_scores[batch_idx]
                    # Standardize scores to zero mean and unit variance
                    standardized_scores = (batch_scores - batch_scores.mean()) / (batch_scores.std() + 1e-8)
                    # Apply temperature-controlled softmax for score sharpening/smoothing
                    normalized_scores[batch_idx] = torch.softmax(standardized_scores / self.temperature, dim=0)

                # Weight the spatial importance scores by alpha and add to attention-based scores
                # This combines learned attention patterns with spatial saliency guidance
                hidden_state_scores = normalized_scores.unsqueeze(1) * self.alpha

                # Augment attention cache with spatial importance for current image region
                attn_cache[:, :, self.vision_start_idx[-1]: self.vision_end_idx[-1] ] += hidden_state_scores.to(attn_cache.device)

                # Step 3: Temporal Redundancy Detection using QR Decomposition
                # This identifies and removes duplicate visual content across multiple GUI screenshots
                if len(self.vision_start_idx) > 1:
                    # Extract keys from the most recent (current) GUI screenshot
                    last_image_keys = key_states[:, :, self.vision_start_idx[-1]:self.vision_end_idx[-1], :]

                    # Process each batch and attention head independently
                    for batch_idx in range(bsz):
                        for head_idx in range(num_heads):
                            # Get keys for current image for this batch/head
                            L = last_image_keys[batch_idx, head_idx, :, :]

                            # Convert to float32 for numerical stability in QR decomposition
                            L_float32 = L.float()

                            # Determine rank for low-rank approximation (capped at 32 for efficiency)
                            # This creates a subspace that captures the main visual patterns of the current image
                            rank = min(32, L_float32.shape[0], L_float32.shape[1])

                            # Perform QR decomposition to create orthonormal basis for current image's key space
                            # Q contains orthonormal basis vectors that span the key subspace
                            # This basis will be used to detect redundancy in previous images
                            L_T = L_float32.T
                            Q, R = torch.linalg.qr(L_T, mode='reduced')
                            Q = Q[:, :rank]  # Keep only top 'rank' basis vectors

                            # Check all previous images for temporal redundancy
                            all_residual_norms = []
                            all_positions = []

                            for img_idx in range(len(self.vision_start_idx) - 1):
                                prev_start = self.vision_start_idx[img_idx]
                                prev_end = self.vision_end_idx[img_idx]

                                # Get keys from previous image
                                prev_image_keys = key_states[batch_idx:batch_idx+1, head_idx:head_idx+1, prev_start:prev_end, :].squeeze(0).squeeze(0)
                                V_prev_float32 = prev_image_keys.float()

                                # Project previous image keys onto current image's subspace
                                # This finds how much of the previous image is "explained" by current image
                                projections = Q.T @ V_prev_float32.T

                                # Reconstruct previous keys using current image's basis
                                V_proj = (Q @ projections).T

                                # Calculate residuals: parts of previous image NOT explained by current image
                                # Small residuals = high redundancy (can be safely pruned)
                                # Large residuals = unique content (should be preserved)
                                residuals = V_prev_float32 - V_proj
                                residual_norms = torch.norm(residuals, p=2, dim=-1)

                                all_residual_norms.append(residual_norms)
                                positions = torch.arange(prev_start, prev_end, device=residual_norms.device)
                                all_positions.append(positions)

                            # Prune tokens with smallest residual norms (highest redundancy)
                            if len(all_residual_norms) > 0:
                                all_residual_norms = torch.cat(all_residual_norms)
                                all_positions = torch.cat(all_positions)

                                # Calculate how many tokens to prune based on compression budget
                                num_tokens_to_zero = int((1.0 - budget) * len(all_residual_norms))

                                if num_tokens_to_zero > 0:
                                    # Find positions with smallest residual norms (most redundant)
                                    _, redundant_indices = torch.topk(all_residual_norms, num_tokens_to_zero, largest=False)

                                    # Zero out importance scores for redundant positions
                                    # These will be pruned in the final selection step
                                    positions_to_zero = all_positions[redundant_indices]
                                    attn_cache[batch_idx, head_idx, positions_to_zero] = 0

            # Step 4: Select top-k tokens based on combined importance scores
            # After spatial saliency augmentation and temporal redundancy filtering,
            # select the most important tokens to keep within the memory budget
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices

            # Store selected indices for potential analysis or debugging
            self.kept_indices = indices

            # Expand indices to match head dimension for gathering
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            # Optional: Apply merge strategy to combine similar KV pairs
            # This can further reduce memory while preserving information
            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            # Step 5: Compress KV cache by gathering selected tokens
            # Split the sequence into two parts:
            # 1. Compressed past: Selected important tokens from history (excluding recent window)
            # 2. Recent window: Always preserved in full for context continuity

            # Gather selected tokens from the past (excluding recent window)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)

            # Keep the entire recent window unchanged
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]

            # Concatenate compressed past with recent window to form final compressed cache
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states


def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = PyramidKVCluster(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )
 
def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = SnapKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )



def init_vlcache(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = VLCacheCluster(
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        last_vision_indices = self.last_vision_indices,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        layer_budget = self.config.layer_budget,
        )        
    
def init_gui_kv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

        assert hasattr(self.config, 'vision_start_idx')
        assert hasattr(self.config, 'vision_end_idx')
        assert hasattr(self.config, 'alpha')
        assert hasattr(self.config, 'max_capacity_prompt')
        assert hasattr(self.config, 'window_size')
        assert hasattr(self.config, 'temperature')

    self.kv_cluster = GUIKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        alpha = self.config.alpha,
        vision_start_idx = self.config.vision_start_idx,
        vision_end_idx = self.config.vision_end_idx,
        temperature = self.config.temperature,
        )    
