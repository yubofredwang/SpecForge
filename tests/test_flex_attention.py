import sys
import unittest
import torch
import torch.nn as nn
import torch._dynamo as dynamo
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlexAttention,
    prepare_decoder_attention_mask,
)
from specforge.utils import padding
from transformers.cache_utils import DynamicCache


dynamo.config.recompile_limit = 64

TTT_LENGTH = 7

class TestFlexAttention(unittest.TestCase):
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
            "num_hidden_layers": 1,
            "torch_dtype": "float32",
        }
        self.config = LlamaConfig(**self.config_dict)
        
        # Define sequence lengths to test (multiples of 128)
        # TODO: Expand this to longer
        # max_length = 128 * 20
        self.seq_lengths = [4096]
        self.dtype = torch.float32

    def test_forward_pass_comparison(self):
        """Test forward pass comparison between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_forward_pass_comparison_for_seq_len(seq_len)
    
    def _test_forward_pass_comparison_for_seq_len(self, seq_len):
        """Helper method to test forward pass comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)
        
        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight) 
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)
        
        attention.eval()
        flex_attention.eval()
        batch_size = 2
        hidden_size = self.config.hidden_size * 2
        
        ############### Attention Inputs ##############

        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        padding_start_index = seq_len - min(200, seq_len // 3)  # Adjust padding based on seq_len
        attention_mask[1, padding_start_index:] = False
        input_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size, dtype=self.dtype).to("cuda")
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        ttt_length = TTT_LENGTH
        past_key_values = DynamicCache()
        for idx in range(ttt_length):
            is_last = idx == ttt_length - 1
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=self.dtype)
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
                rtol=1e-3,
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

    def test_backward_pass_gradient_comparison(self):
        """Test backward pass comparing gradients between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_backward_pass_gradient_comparison_for_seq_len(seq_len)

    def _test_backward_pass_gradient_comparison_for_seq_len(self, seq_len):
        """Helper method to test backward pass gradient comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)
        
        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight) 
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        batch_size = 2
        hidden_size = self.config.hidden_size * 2
        
        ############### Attention Inputs ##############
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        # padding_start_index = seq_len - 50
        # attention_mask[1, padding_start_index:] = False
        input_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size, dtype=self.dtype).to("cuda")
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        ttt_length = TTT_LENGTH
        past_key_values = DynamicCache()
        loss_mask = torch.ones(batch_size, seq_len, dtype=self.dtype, requires_grad=False).to("cuda")

        # Create input tensors that require gradients
        loss_list = []
        loss_flex_list = []
        
        for idx in range(ttt_length):
            is_last = idx == ttt_length - 1
            
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            output = attention(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                output_attentions=False,
                use_cache=True
            )
            output_flex = flex_attention(
                hidden_states=flex_hidden_states,
                attention_mask=attention_mask,
                position_ids=flex_position_ids,
                past_key_values=past_key_values,
            )
            # Apply loss mask on calculation over batch
            loss = (output * loss_mask[..., None]).sum().mean()
            loss_flex = (output_flex * loss_mask[..., None]).sum().mean()
            torch.testing.assert_close(loss, loss_flex, atol=1e-3, rtol=1e-3)
            loss_list.append(loss)
            loss_flex_list.append(loss_flex)
            # Compare gradients

            if not is_last:
                # Step 5.7: we need to update the loss mask
                ind = torch.arange(seq_len, device=decoder_attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_len - idx]
                decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(decoder_attention_mask.dtype).min
                loss_mask = padding(loss_mask, left=False)
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_flex = sum(loss_flex_list) / len(loss_flex_list)
        mean_loss.backward()
        mean_loss_flex.backward()
        projections = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        for proj_name in projections:
            torch.testing.assert_close(
                getattr(attention, proj_name).weight.grad, 
                getattr(flex_attention, proj_name).weight.grad,
                atol=1e-3,
                rtol=2e-3,
            )

    @unittest.skip("Skipping flex attention basic test")
    def test_flex_attention_basic(self):
        """Helper method to test flex attention with large sequence lengths."""
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)
        batch_size = 1
        hidden_size = self.config.hidden_size * 2
        seq_len = 128
        ttt_length = 3
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("cuda")
        input_embeds = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=self.dtype)
        # Create past_key_values cache
        past_key_values = DynamicCache()
        # Run flex attention forward pass (fewer iterations for large sequences)
        for i in range(ttt_length):  # Reduced iterations to avoid memory issues
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=self.dtype)
            output = flex_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        
        # Check output shape
        expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_output_shape)
        
        # Check output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Check that cache was populated
        self.assertIsNotNone(past_key_values.key_cache)
        self.assertIsNotNone(past_key_values.value_cache)
        self.assertEqual(len(past_key_values.key_cache), 1)  # One layer
        self.assertEqual(len(past_key_values.value_cache), 1)  # One layer


if __name__ == "__main__":
    # Run test_flex_attention_basic only
    unittest.main(verbosity=2)
