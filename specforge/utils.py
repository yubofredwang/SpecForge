import json
import os
import re
import hashlib
from contextlib import contextmanager
from datetime import datetime

import torch
import torch.distributed as dist
from torch.profiler import schedule
from transformers import PretrainedConfig


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)


def print_with_rank(message):
    print(f"rank {dist.get_rank()}: {message}")


PREFIX_CHECKPOINT_DIR = "epoch"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"_(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )


def profiler_schedule(wait=1, warmup=2, active=1, repeat=0):
    return schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

def trace_handler(output_dir):
    def handler_fn(profiler) -> None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_name = f"profiler_rank{dist.get_rank()}_{timestamp}.pt.trace.json"
        trace_path = os.path.join(output_dir, trace_name)
        profiler.export_chrome_trace(trace_path)
        print(f"Profiler data saved to {trace_path}")
        memory_timeline_name = f"profiler_rank{dist.get_rank()}_{timestamp}.html"
        memory_timeline_path = os.path.join(output_dir, memory_timeline_name)
        profiler.export_memory_timeline(memory_timeline_path)
        print(f"Memory timeline data saved to {memory_timeline_path}")
        print(f"Stopped PyTorch profiler")
    return handler_fn


def generate_vocab_cache_key(config_path: str, dataset_cache_key: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    return hashlib.md5((json.dumps(config) + dataset_cache_key).encode()).hexdigest()