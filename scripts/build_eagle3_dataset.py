"""
Preprocess data for dataset generation. This runs faster without c10d comms.
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge.data import build_eagle3_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--chat-template", type=str, required=True, choices=["llama3", "qwen"])
    return parser.parse_args()


def main():
    """
    Separated script to build eagle3 dataset from the training.
    
    Usage:
    python ./scripts/build_eagle3_dataset.py  \
        --data-path "cache/dataset/sharegpt.jsonl" \
        --model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct \
        --chat-template llama3
    """
    args = parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.cache_dir, "processed_dataset")

    data_paths = args.data_path.split(",")
    dataset_list = []
    for data_path in data_paths:
        assert os.path.exists(
            data_path
        ), f"Dataset path {data_path} does not exist"
        dataset = load_dataset("json", data_files=data_path)["train"]
        dataset_list.append(dataset)
    if len(dataset_list) > 1:
        dataset = concatenate_datasets(dataset_list)
    else:
        dataset = dataset_list[0]

    if args.num_samples is not None:
        print(f"Selecting {args.num_samples} samples from {len(dataset)}")
        dataset = dataset.select(range(args.num_samples))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cache_key = hashlib.md5(args.data_path.encode()).hexdigest()
    eagle3_dataset = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
    )
    print(f"Built dataset")


if __name__ == "__main__":
    main()
