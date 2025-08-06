import torch
import torch.nn as nn
import math
from packaging import version
from typing import Optional, List, Tuple, ClassVar
from transformers import LlamaConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from specforge.modeling.draft.llama3_eagle import (
    LlamaRotaryEmbedding, 
    LlamaLinearScalingRotaryEmbedding, 
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb
)
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache, DynamicCache
from specforge.modeling.draft.eagle3_flex_mask import generate_eagle3_mask





if __name__ == "__main__":
    config_dict = {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-05,
        "vocab_size": 32000,
        "intermediate_size": 1376,
        "hidden_act": "silu",
    }
    config = LlamaConfig(**config_dict)
    llama_flex_attention = LlamaFlexAttention(config)
    llama_flex_attention.eval()
    batch_size = 1
    seq_len = 128 * 4
    hidden_size = config.hidden_size * 2
    # Simulate inputs
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    # Switch to StaticCache
    past_key_values = DynamicCache()
    attention_mask = torch.ones(batch_size, seq_len)
    # First 128 is padding
    attention_mask[:, 512:] = False
    for i in range(4):
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        flex_output = llama_flex_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
    print(flex_output.shape)
    