import unittest
import torch
import torch.nn as nn
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention, 
    prepare_decoder_attention_mask,
)
from specforge.modeling.draft.llama3_flex_attention import LlamaFlexAttention
from specforge.utils import padding




class TestLlamaAttention(unittest.TestCase):
    """Comprehensive test suite for LlamaAttention with simulated inputs."""

    def setUp(self):
        """Set up test configurations and common parameters."""
        # Basic configuration
        self.config_dict = {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 1376,
            "hidden_act": "silu",
        }
        self.config = LlamaConfig(**self.config_dict)



    def test_forward_pass_with_cache(self):
        """Test forward pass with caching mechanism."""
        attention = LlamaAttention(self.config)
        attention.eval()
        
        batch_size = 1
        seq_len = 128 * 4
        hidden_size = self.config.hidden_size * 2
        
        # Simulate inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        # Initialize cache, skip the sdpa
        cache_hidden = [[], []]  # [cache_k, cache_v]
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        # torch.matmul(query_states, k0.transpose(2, 3)) expects original to be B, H, T, D
        cache_hidden[0].append(torch.randn(batch_size, num_heads, seq_len, self.config.head_dim))
        cache_hidden[1].append(torch.randn(batch_size, num_heads, seq_len, self.config.head_dim))
        
        attention_mask = torch.ones(batch_size, seq_len)
        # First 128 is padding
        attention_mask[:, 128:] = False
        input_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size)
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )
        
        key_cache = None
        value_cache = None

        for idx in range(7):
            is_last = idx == 6

            with torch.no_grad():
                output = attention(
                    hidden_states=hidden_states,
                    attention_mask=decoder_attention_mask,
                    position_ids=position_ids,
                    cache_hidden=cache_hidden,
                    output_attentions=False,
                    use_cache=True
                )
                
            if not is_last:
                # Step 5.7: we need to update the loss mask
                ind = torch.arange(seq_len, device=decoder_attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_len - idx]
                decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(decoder_attention_mask.dtype).min
        
        # Check output shape
        expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_output_shape)
        
        # Check cache is populated
        self.assertGreater(len(cache_hidden[0]), 0)  # Keys cached
        self.assertGreater(len(cache_hidden[1]), 0)  # Values cached
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


if __name__ == "__main__":
     unittest.main(verbosity=2)



