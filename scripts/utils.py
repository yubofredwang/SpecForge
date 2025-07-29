import os
from datetime import datetime

import torch.distributed as dist
from torch.profiler import schedule


def profiler_schedule(wait=1, warmup=2, active=2, repeat=1):
    return schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

def export_profiler_trace(profiler):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_name = f"profiler_rank{dist.get_rank()}_{timestamp}.pt.trace.json"
    trace_path = os.path.join(os.environ["SGLANG_TORCH_PROFILER_DIR"], trace_name)
    profiler.export_chrome_trace(trace_path)
    print(f"Profiler data saved to {trace_path}")
    memory_timeline_name = f"profiler_rank{dist.get_rank()}_{timestamp}.html"
    memory_timeline_path = os.path.join(os.environ["SGLANG_TORCH_PROFILER_DIR"], memory_timeline_name)
    profiler.export_memory_timeline(memory_timeline_path)
    print(f"Memory timeline data saved to {memory_timeline_path}")
    print(f"Stopped PyTorch profiler")