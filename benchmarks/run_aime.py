"""
AIME benchmark evaluation script.
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


def extract_aime_answer(output: str) -> Optional[str]:
    """Extract final answer from AIME problem solution.

    AIME answers are typically integers between 0 and 999, and are usually
    in \boxed{} format.
    """
    # Try to find answer in \boxed{} format
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(boxed_pattern, output)
    if match:
        answer = match.group(1).strip()
        # Extract number from the boxed content
        numbers = re.findall(r"\d+", answer)
        if numbers:
            return numbers[-1]  # Take the last number (usually the final answer)
        return answer

    # Try to find answer in \boxed format (without braces)
    boxed_pattern2 = r"\\boxed\s+(\d+)"
    match = re.search(boxed_pattern2, output)
    if match:
        return match.group(1).strip()

    # Look for patterns like "The answer is 42" or "Answer: 123"
    answer_patterns = [
        r"(?:answer|Answer|ANSWER)[\s:]+(\d+)",
        r"(?:final\s+answer|Final\s+Answer)[\s:]+(\d+)",
        r"(?:is|equals?|=\s*)(\d+)\s*$",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Fallback: extract the last integer in the text
    numbers = re.findall(r"\b(\d+)\b", output)
    if numbers:
        # Filter to reasonable AIME answer range (0-999)
        valid_numbers = [n for n in numbers if 0 <= int(n) <= 999]
        if valid_numbers:
            return valid_numbers[-1]

    return None


class AIMEBenchmark(BaseBenchmark):
    """AIME benchmark implementation."""

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess AIME dataset."""
        dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
        questions = []
        labels = []
        for idx, q in enumerate(dataset):
            if idx >= self.args.num_questions:
                break
            questions.append({"question": q["Problem"]})
            # Extract answer from Answer field
            answer = None
            if "Answer" in q:
                answer = str(q["Answer"]).strip()
            elif "answer" in q:
                answer = str(q["answer"]).strip()
            labels.append(answer)
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Extract answer from model output."""
        return extract_aime_answer(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for AIME by comparing numeric answers."""
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
                    # Normalize answers for comparison
                    pred_normalized = str(pred).strip()
                    label_normalized = str(label).strip()
                    # Try exact match first
                    if pred_normalized == label_normalized:
                        correct += 1
                    else:
                        # Try numeric comparison
                        try:
                            pred_num = int(pred_normalized)
                            label_num = int(label_normalized)
                            if pred_num == label_num:
                                correct += 1
                        except ValueError:
                            pass

        return correct / valid_count if valid_count > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for AIME with reasoning prompt."""
        return create_simple_sgl_function(
            function_name="reasoning_gen",
            answer_key="answer",
            user_prefix="\nPlease reason step by step, and put your final answer within \\boxed{}.",
            max_tokens=self.get_max_new_tokens(),
        )

    def get_max_new_tokens(self) -> int:
        """AIME problems require more tokens."""
        return 32768


def main(args):
    """Main entry point."""
    benchmark = AIMEBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=1)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
