"""
MATH-500 benchmark evaluation script.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from sglang.test.test_utils import add_common_sglang_args_and_parse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_benchmark import BaseBenchmark
from benchmarks.utils import create_simple_sgl_function


def extract_math_answer(output: str) -> Optional[str]:
    """Extract final answer from math problem solution.

    Tries to extract answer from \boxed{} format first, then looks for
    the last number in the output.
    """
    # Try to find answer in \boxed{} format
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(boxed_pattern, output)
    if match:
        return match.group(1).strip()

    # Try to find answer in \boxed format (without braces)
    boxed_pattern2 = r"\\boxed\s+([^\s]+)"
    match = re.search(boxed_pattern2, output)
    if match:
        return match.group(1).strip()

    # Try to find the last number (could be integer or decimal)
    # Look for patterns like "The answer is 42" or "Answer: 3.14"
    answer_patterns = [
        r"(?:answer|Answer|ANSWER)[\s:]+([-+]?\d*\.?\d+)",
        r"(?:is|equals?|=\s*)([-+]?\d*\.?\d+)\s*$",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Fallback: extract the last number in the text
    numbers = re.findall(r"[-+]?\d*\.?\d+", output)
    if numbers:
        return numbers[-1]

    return None


class Math500Benchmark(BaseBenchmark):
    """MATH-500 benchmark implementation."""

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess MATH-500 dataset."""
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        questions = []
        labels = []
        for idx, q in enumerate(dataset):
            if idx >= self.args.num_questions:
                break
            questions.append({"question": q["problem"]})
            # Extract answer from solution or answer field
            answer = None
            if "answer" in q:
                answer = str(q["answer"]).strip()
            elif "solution" in q:
                # Try to extract from solution
                answer = extract_math_answer(q["solution"])
            labels.append(answer)
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Extract answer from model output."""
        return extract_math_answer(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for MATH-500 by comparing answers."""
        if not labels or len(labels) == 0:
            return None
        if all(label is None for label in labels):
            return None

        correct = 0
        valid_count = 0
        for pred, label in zip(predictions, labels):
            if label is not None:
                valid_count += 1
                if pred is not None:
                    # Normalize answers for comparison (remove whitespace, handle different formats)
                    pred_normalized = str(pred).strip().lower()
                    label_normalized = str(label).strip().lower()
                    # Try exact match first
                    if pred_normalized == label_normalized:
                        correct += 1
                    else:
                        # Try numeric comparison if both are numbers
                        try:
                            pred_num = float(pred_normalized)
                            label_num = float(label_normalized)
                            if abs(pred_num - label_num) < 1e-6:
                                correct += 1
                        except ValueError:
                            pass

        return correct / valid_count if valid_count > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for MATH-500."""
        return create_simple_sgl_function(
            function_name="get_math500_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )


def main(args):
    """Main entry point."""
    benchmark = Math500Benchmark(args)
    benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--num-runs", type=int, default=1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
