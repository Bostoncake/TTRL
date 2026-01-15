# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for SAT2 spatial reasoning task.
Performs exact text matching on answer choices.
"""

import re
import traceback


def extract_answer(response: str, answer_choices: list = None) -> str:
    """
    Extract the answer from the model's response.

    For SAT2, answers are typically in the format:
    - Direct text: "did not move"
    - Letter choice: "A" or "A."
    - Full choice: "A. did not move"

    Args:
        response: The model's generated response
        answer_choices: Optional list of valid answer choices for validation

    Returns:
        Extracted answer text, or None if extraction fails
    """
    response = response.strip()

    # Try to extract from common patterns
    # Pattern 1: "Answer: <text>" or "The answer is: <text>"
    match = re.search(r"(?:answer|Answer|ANSWER)(?:\s+is)?:\s*(.+?)(?:\.|$)", response, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        return clean_answer(answer)

    # Pattern 2: Look for letter choice (A, B, C, D) followed by period or text
    match = re.search(r"^([A-D])\.?\s*(.*?)$", response.strip(), re.MULTILINE)
    if match:
        letter = match.group(1)
        text = match.group(2).strip()
        # Return the full text if available, otherwise just the letter
        return clean_answer(text if text else letter)

    # Pattern 3: Just the text answer (last resort)
    # If response is short and looks like a direct answer
    if len(response) < 200:  # Reasonable length for a single answer
        return clean_answer(response)

    # Try to find the last sentence as the answer
    sentences = re.split(r'[.!?]+', response)
    if sentences:
        last_sentence = sentences[-1].strip()
        if last_sentence:
            return clean_answer(last_sentence)

    return None


def clean_answer(answer: str) -> str:
    """
    Clean and normalize an answer string.

    Args:
        answer: Raw answer text

    Returns:
        Cleaned answer text
    """
    # Remove leading/trailing whitespace
    answer = answer.strip()

    # Remove trailing periods
    answer = answer.rstrip('.')

    # Remove quotes
    answer = answer.strip('"\'')

    # Normalize whitespace
    answer = ' '.join(answer.split())

    return answer


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Answer text

    Returns:
        Normalized answer (lowercase, no extra spaces)
    """
    return clean_answer(answer).lower()


def grade_answer(model_answer: str, ground_truth: str) -> bool:
    """
    Grade a model answer against the ground truth.

    Uses normalized string matching with some flexibility:
    - Case-insensitive
    - Ignores extra whitespace
    - Checks if model answer contains ground truth or vice versa

    Args:
        model_answer: The model's extracted answer
        ground_truth: The correct answer

    Returns:
        True if correct, False otherwise
    """
    if model_answer is None or ground_truth is None:
        return False

    # Normalize both answers
    norm_model = normalize_answer(model_answer)
    norm_gt = normalize_answer(ground_truth)

    # Exact match
    if norm_model == norm_gt:
        return True

    # Check if one contains the other (for cases like "A" vs "A. did not move")
    if norm_gt in norm_model or norm_model in norm_gt:
        return True

    # Additional fuzzy matching for common variations
    # Remove common prefixes like "the answer is", "i think", etc.
    prefixes = ["the answer is", "i think", "i believe", "it is", "it's"]
    for prefix in prefixes:
        if norm_model.startswith(prefix):
            norm_model = norm_model[len(prefix):].strip()
            if norm_model == norm_gt or norm_gt in norm_model:
                return True

    return False


def compute_score(model_response: str, ground_truth: str) -> dict:
    """
    Compute the reward score for a model response.

    Args:
        model_response: The full model response
        ground_truth: The correct answer

    Returns:
        Dictionary with score and metadata:
        - score: 1.0 if correct, 0.0 if wrong
        - acc: Boolean correctness
        - extracted_answer: What was extracted from the response
        - ground_truth: The correct answer
    """
    # Extract answer from model response
    model_answer = extract_answer(model_response)

    # If we couldn't extract an answer, return 0
    if model_answer is None:
        return {
            "score": 0.0,
            "acc": False,
            "extracted_answer": "",
            "ground_truth": ground_truth,
            "format_score": 0.0,  # Couldn't parse the format
        }

    # Grade the answer
    is_correct = grade_answer(model_answer, ground_truth)

    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "extracted_answer": model_answer,
        "ground_truth": ground_truth,
        "format_score": 1.0,  # Successfully extracted an answer
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None
):
    """
    Main reward function interface for SAT2 spatial reasoning.

    This function is called by the TTRL training pipeline to compute rewards.

    Args:
        data_source: Source of the data (e.g., "SAT2_base")
        solution_str: The model's generated response
        ground_truth: The correct answer
        extra_info: Optional extra information about the example
        sandbox_fusion_url: Unused (for compatibility with other reward functions)
        concurrent_semaphore: Unused (for compatibility)

    Returns:
        Dictionary with score and metadata, or float score
    """
    try:
        res = compute_score(solution_str, str(ground_truth))

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            # Fallback
            return 0.0

    except Exception as e:
        print(f"[ERROR] Error in SAT2 reward function: {str(e)}")
        traceback.print_exc()
        # Return 0 score on error
        return {
            "score": 0.0,
            "acc": False,
            "extracted_answer": "",
            "ground_truth": str(ground_truth),
            "error": str(e),
        }
