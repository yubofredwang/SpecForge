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
from transformers.cache_utils import DynamicCache



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
        flex_attention = LlamaFlexAttention(self.config)
        
        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight) 
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)
        
        attention.eval()
        flex_attention.eval()
        batch_size = 2
        seq_len = 128 * 4
        hidden_size = self.config.hidden_size * 2
        
        ############### Attention Inputs ##############

        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len)
        # Simulate one item in the batch is masked and not taking a full block.
        padding_start_index = 128 * 4 - 200
        attention_mask[1, padding_start_index:] = False
        input_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size)
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        past_key_values = DynamicCache()

        ttt_length = 7
        for idx in range(ttt_length):
            is_last = idx == 6
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            flex_hidden_states = hidden_states.clone()
            with torch.no_grad():
                output = attention(
                    hidden_states=hidden_states,
                    attention_mask=decoder_attention_mask,
                    position_ids=position_ids,
                    cache_hidden=cache_hidden,
                    output_attentions=False,
                    use_cache=True
                )
            with torch.no_grad():
                output_flex = flex_attention(
                    hidden_states=flex_hidden_states,
                    attention_mask=attention_mask,
                    position_ids=flex_position_ids,
                    past_key_values=past_key_values,
                )
            torch.testing.assert_close(output[0][:-1-idx], output_flex[0][:-1-idx], atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                output[1][:padding_start_index-idx], 
                output_flex[1][:padding_start_index-idx], 
                atol=1e-3, 
                rtol=1e-3
            )
            if not is_last:
                # Step 5.7: we need to update the loss mask
                ind = torch.arange(seq_len, device=decoder_attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_len - idx]
                decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(decoder_attention_mask.dtype).min
            # Check output shape
            expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
            self.assertEqual(output_flex.shape, expected_output_shape)
            # Check output is not NaN or Inf
            self.assertFalse(torch.isnan(output_flex).any())
            self.assertFalse(torch.isinf(output_flex).any())


if __name__ == "__main__":
     unittest.main(verbosity=2)
