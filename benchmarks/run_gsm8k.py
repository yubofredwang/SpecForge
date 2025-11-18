"""
GSM8K benchmark evaluation script.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sglang.test.test_utils import add_common_sglang_args_and_parse
from sglang.utils import download_and_cache_file, read_jsonl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_benchmark import BaseBenchmark
from benchmarks.utils import create_few_shot_sgl_function

INVALID = -9999999


def get_one_example(lines: List[Dict], i: int, include_answer: bool) -> str:
    """Format a single example."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines: List[Dict], k: int) -> str:
    """Get few-shot examples as a string."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str: str) -> int:
    """Extract numeric answer from model output."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K benchmark implementation."""

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Load and preprocess GSM8K dataset."""
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

        # Construct prompts
        num_questions = self.args.num_questions
        num_shots = self.args.num_shots
        few_shot_examples = get_few_shot_examples(lines, num_shots)

        questions = []
        labels = []
        for i in range(min(len(lines), num_questions)):
            question_text = get_one_example(lines, i, False)
            questions.append({"question": question_text})
            labels.append(get_answer_value(lines[i]["answer"]))

        # Store few_shot_examples for use in create_sgl_function
        self.few_shot_examples = few_shot_examples

        assert all(l != INVALID for l in labels), "Some labels are invalid"
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[int]:
        """Extract numeric answer from model output."""
        return get_answer_value(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for GSM8K by comparing numeric answers."""
        if not labels or len(labels) == 0:
            return None
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if len(labels) > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for GSM8K with few-shot examples."""
        return create_few_shot_sgl_function(
            few_shot_examples=self.few_shot_examples,
            function_name="few_shot_gsm8k",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            stop=["Question", "Assistant:", "<|separator|>"],
        )


def main(args):
    """Main entry point."""
    benchmark = GSM8KBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--num-runs", type=int, default=1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
