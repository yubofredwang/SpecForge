import torch
import time
import gc
from transformers import LlamaConfig
from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlexAttention,
    prepare_decoder_attention_mask,
)
from transformers.cache_utils import Cache, StaticCache
from specforge.utils import padding


config_dict = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-05,
    "vocab_size": 32000,
    "hidden_act": "silu",
}

config = LlamaConfig(**config_dict)


def run_attention(seq_len: int, attention_backend: str = "sdpa"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    hidden_size = config.hidden_size * 2
    ttt_length = 7
    # Initialize cache and attention function based on backend
    if attention_backend == "sdpa":
        cache_hidden = [[], []]
        past_key_values = None
        attn_func = LlamaAttention(config).to(device)
    elif attention_backend == "flex_attention":
        cache_hidden = None
        past_key_values = StaticCache(
            config,
            max_batch_size=batch_size,
            max_cache_len=ttt_length * seq_len,
            device=device,
            dtype=torch.bfloat16,
        )
        attn_func = LlamaFlexAttention(config).to(device)
    else:
        raise ValueError(f"Unknown attention backend: {attention_backend}")



    # Simulate inputs - move to device
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    input_embeds = torch.randn(batch_size, seq_len, config.hidden_size).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    decoder_attention_mask = prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        input_shape=(batch_size, seq_len),
        inputs_embeds=input_embeds,
        past_key_values_length=0,
    )

    loss_list = []

    
    for idx in range(ttt_length):
        is_last = idx == ttt_length - 1
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True).to(device)
        
        # Call attention function with appropriate parameters
        if attention_backend == "sdpa":
            output = attn_func(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                output_attentions=False,
                use_cache=True
            )
        else:  # flex_attention
            output = attn_func(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True
            )
        
        # Compute a simple loss for benchmarking
        loss = output[0].sum()
        loss_list.append(loss)
        
        if not is_last:
            # Step 5.7: we need to update the loss mask
            ind = torch.arange(seq_len, device=decoder_attention_mask.device)
            ind0 = ind[idx:]
            ind1 = ind[: seq_len - idx]
            decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(decoder_attention_mask.dtype).min
    
    # Compute mean loss and backward pass
    if loss_list:
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss.backward()


def benchmark_function(attention_backend: str, seq_lengths: list):
    """Benchmark a function for speed and GPU memory usage."""
    print(f"\n=== Benchmarking {attention_backend} ===")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Warm up runs
    print("Warming up...")
    for _ in range(3):
        run_attention(seq_lengths[0], attention_backend)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    print("Warmup done, clearing cache...")
    # Clear cache again after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Initialize profiler for flex attention backend
    profiler = None
    if attention_backend == "flex_attention":
        profiler = torch.profiler.profile(
            activities=[
                # torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./profiler_logs/{attention_backend}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    # Start timer
    start_time = time.time()
    
    # Record initial memory
    initial_memory = 0
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()

    step_count = 0
    for i, seq_len in enumerate(seq_lengths):
        if profiler and step_count < 3:
            step_count += 1
            profiler.step(step_count)
        run_attention(seq_len, attention_backend)

    end_time = time.time()
    
    # Stop profiler for flex attention
    if profiler:
        profiler.stop()
        print(f"Profiler traces saved to ./profiler_logs/{attention_backend}")
    
    peak_memory = 0
    current_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        print(f"Peak GPU memory: {peak_memory / 1024**3:.3f} GB")
        print(f"Current GPU memory: {current_memory / 1024**3:.3f} GB")
        print(f"Memory increase: {(current_memory - initial_memory) / 1024**3:.3f} GB")
    
    return {
        'time': end_time - start_time,
        'peak_memory': peak_memory,
        'memory_increase': current_memory - initial_memory
    }


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name())
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    else:
        print("CUDA not available - running on CPU")
    
    # Define sequence lengths to test
    seq_lengths = [128 * i for i in range(1, 10, 2)]
    print(f"Testing sequence lengths: {seq_lengths}")
    
    # Run each benchmark 5 times
    eagle_results_list = []
    flex_results_list = []
    
    run_num = 1
    for run_idx in range(run_num):
        print(f"\n--- Run {run_idx + 1}/{run_num} ---")
        eagle_results = benchmark_function("sdpa", seq_lengths)
        flex_results = benchmark_function("flex_attention", seq_lengths)
        
        eagle_results_list.append(eagle_results)
        flex_results_list.append(flex_results)
    
    # Calculate averages
    avg_eagle_time = sum(r['time'] for r in eagle_results_list) / len(eagle_results_list)
    avg_flex_time = sum(r['time'] for r in flex_results_list) / len(flex_results_list)
    
    print(f"\n=== Final Results (averaged over {run_num} runs) ===")
    print(f"Eagle (SDPA) average time: {avg_eagle_time:.3f} seconds")
    print(f"Flex attention average time: {avg_flex_time:.3f} seconds")
    print(f"Speed ratio (Eagle/Flex): {avg_eagle_time / avg_flex_time:.2f}x")
    
    if torch.cuda.is_available() and eagle_results_list and flex_results_list:
        # Use the last run's memory results for comparison
        eagle_peak = eagle_results_list[-1]['peak_memory']
        flex_peak = flex_results_list[-1]['peak_memory']
        
        if flex_peak > 0:  # Avoid division by zero
            memory_ratio = eagle_peak / flex_peak
            print(f"\nMemory comparison (Eagle vs Flex):")
            print(f"  Eagle peak: {eagle_peak / 1024**3:.3f} GB")
            print(f"  Flex peak:  {flex_peak / 1024**3:.3f} GB")
            print(f"  Ratio: {memory_ratio:.2f}x ({'Eagle uses less' if memory_ratio < 1 else 'Flex uses less'})")
