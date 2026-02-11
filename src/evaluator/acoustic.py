#!/usr/bin/env python3
"""
WavBench Acoustic Evaluator

Evaluates Acoustic dataset responses including:
- Explicit Generation: Audio output evaluation
- Explicit Understanding: Text response evaluation
- Implicit Generation/Understanding: Audio and text evaluation
- Multi-round: Multi-turn conversation evaluation
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..api.utils import call_llm_api, safe_float, parse_json_response
import re


# ============================================================================
# Score Extraction Utilities
# ============================================================================

def extract_score_from_response(response: str, valid_scores: List[int] = None) -> Optional[float]:
    """
    Extract score from LLM response with multiple fallback strategies.

    Strategies (in order):
    1. Parse JSON format {"score": X, "reasoning": "..."}
    2. Find standalone number at the end of response
    3. Find any valid score number in the response

    Args:
        response: Raw LLM response text
        valid_scores: List of valid score values (e.g., [1, 3, 5] or [1, 2, 3, 4, 5])

    Returns:
        Extracted score or None if extraction failed
    """
    if valid_scores is None:
        valid_scores = [1, 2, 3, 4, 5]

    # Strategy 1: Try JSON parsing
    parsed = parse_json_response(response)
    if parsed and "score" in parsed:
        score = safe_float(parsed["score"])
        if score is not None and int(score) in valid_scores:
            return score

    # Strategy 2: Find number at the end of response (after thinking)
    # Look for patterns like "...\n1" or "...\n**5**" at the end
    lines = response.strip().split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip()
        if not line:
            continue
        # Clean markdown formatting
        cleaned = re.sub(r'[*_`#\-\[\]()]', '', line).strip()
        # Check if line is just a number
        if cleaned.isdigit():
            score = int(cleaned)
            if score in valid_scores:
                return float(score)
        # Check for "score: X" pattern
        match = re.search(r'score[:\s]+(\d+)', cleaned, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if score in valid_scores:
                return float(score)

    # Strategy 3: Find first word that's a valid score
    first_word = response.split()[0].strip().replace('*', '').replace(':', '') if response.split() else ''
    score = safe_float(first_word)
    if score is not None and int(score) in valid_scores:
        return score

    # Strategy 4: Search for any standalone number in valid_scores
    for num in sorted(valid_scores, reverse=True):  # Prefer higher scores in ambiguous cases
        pattern = rf'\b{num}\b'
        if re.search(pattern, response):
            return float(num)

    return None


def extract_judgment_from_response(response: str) -> Optional[str]:
    """
    Extract correct/incorrect judgment from LLM response.

    Strategies (in order):
    1. Parse JSON format {"judgment": "correct/incorrect", "reasoning": "..."}
    2. Search for "correct" or "incorrect" keywords in response

    Args:
        response: Raw LLM response text

    Returns:
        "correct", "incorrect", or None if extraction failed
    """
    # Strategy 1: Try JSON parsing
    parsed = parse_json_response(response)
    if parsed and "judgment" in parsed:
        judgment = str(parsed["judgment"]).lower().strip()
        if judgment in ["correct", "incorrect"]:
            return judgment

    # Strategy 2: Search for keywords in response
    reply = response.lower()
    # Check for "incorrect" first (since "correct" is a substring of "incorrect")
    if "incorrect" in reply:
        return "incorrect"
    elif "correct" in reply:
        return "correct"

    return None


# ============================================================================
# Explicit Generation Evaluator
# ============================================================================

def call_llm_judge_generation(
    text_instruction: str,
    generated_audio_path: Path,
    gt_label: str,
    task_type: str,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Tuple[bool, str, str]:
    """Judge if generated audio meets the required characteristic."""
    prompt = f"""You are an expert evaluator for audio generation tasks.

Task type: {task_type}
Text instruction: {text_instruction}
Required characteristic (label): {gt_label}

Your task is to listen to the generated audio and determine if it successfully meets the required characteristic.

Please strictly output the evaluation result in the following JSON format:
{{
    "judgment": "<correct or incorrect>",
    "reasoning": "<Detailed reasoning>"
}}

Judgment criteria:
- "correct": The audio meets the required characteristic
- "incorrect": The audio does not meet the required characteristic

Your evaluation (JSON only):"""

    if not generated_audio_path.exists():
        return False, "error", f"Audio file not found: {generated_audio_path}"

    success, response, error = call_llm_api(
        prompt=prompt,
        api_key=api_key,
        model=model,
        audio_path=generated_audio_path,
        max_retry=max_retry,
        sleep_between_retry=sleep_between_retry,
    )

    if success:
        judgment = extract_judgment_from_response(response)
        if judgment:
            return True, judgment, response
        else:
            return False, "error", f"Unrecognized response: {response}"

    return False, "error", error


def evaluate_explicit_generation_sample(
    sample: Dict,
    task_type: str,
    result_base_dir: Path,
    dataset_name: str,
    **kwargs
) -> Dict:
    """Evaluate a single explicit generation sample."""
    audio_path = result_base_dir / dataset_name / sample.get("model_response_audio", "")

    success, judgment, raw = call_llm_judge_generation(
        text_instruction=sample.get("text", ""),
        generated_audio_path=audio_path,
        gt_label=sample.get("label", ""),
        task_type=task_type,
        **kwargs
    )

    sample["eval_result"] = judgment if success else "error"
    sample["eval_correct"] = 1 if judgment == "correct" else 0
    sample["eval_raw_response"] = raw
    if not success:
        sample["eval_reason"] = raw

    return sample


# ============================================================================
# Explicit Understanding Evaluator
# ============================================================================

def call_llm_judge_understanding(
    model_response: str,
    gt_label: str,
    task_type: str,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Tuple[bool, str, str]:
    """Judge if model response correctly identifies the label."""
    prompt = f"""You are an expert evaluator for audio understanding tasks.

Task type: {task_type}
Ground truth label: {gt_label}
Model's text response: {model_response}

Your task is to determine if the model's response correctly identifies the ground truth label.

Rules:
1. The model's response should clearly mention or imply the ground truth label
2. For music instruments: accept synonyms
3. For emotions: accept similar emotions
4. For gender: "male"/"female" must match exactly
5. For age: accept age ranges that include the label
6. For language/accent: accept language name variations
7. Ignore extra information - focus on whether the key label is identified

Please strictly output the evaluation result in the following JSON format:
{{
    "judgment": "<correct or incorrect>",
    "reasoning": "<Detailed reasoning>"
}}

Judgment criteria:
- "correct": The model's response correctly identifies the label
- "incorrect": The model's response does not correctly identify the label

Your evaluation (JSON only):"""

    success, response, error = call_llm_api(
        prompt=prompt,
        api_key=api_key,
        model=model,
        max_retry=max_retry,
        sleep_between_retry=sleep_between_retry,
        max_tokens=512,
    )

    if success:
        judgment = extract_judgment_from_response(response)
        if judgment:
            return True, judgment, response
        else:
            return False, "error", f"Unable to parse: {response}"

    return False, "error", error


def evaluate_explicit_understanding_sample(sample: Dict, task_type: str, **kwargs) -> Dict:
    """Evaluate a single explicit understanding sample."""
    success, judgment, raw = call_llm_judge_understanding(
        model_response=sample.get("model_response_text", ""),
        gt_label=sample.get("label", ""),
        task_type=task_type,
        **kwargs
    )

    sample["eval_result"] = judgment if success else "error"
    sample["eval_correct"] = 1 if judgment == "correct" else 0
    sample["eval_raw_response"] = raw
    if not success:
        sample["eval_reason"] = raw

    return sample


# ============================================================================
# Implicit Audio Evaluator
# ============================================================================

def call_llm_judge_implicit_audio(
    audio_path: Path,
    label: str,
    task_type: str,
    user_input: str,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Tuple[bool, float, str]:
    """Judge if audio demonstrates the required characteristic (1-5 scale)."""
    task_criteria = {
        "age": f"Does the voice sound like it belongs to someone in the '{label}' age group?",
        "emotion": f"Does the speech convey the emotion of '{label}'?",
        "pitch": f"Does the speech have a '{label}' pitch level?",
        "speed": f"Does the speech have a '{label}' speaking speed?",
        "volume": f"Does the speech have a '{label}' volume level?",
        "gender": f"Does the voice have a '{label}' characteristic?",
        "accent": f"Does the voice have a '{label}' accent?",
    }

    criterion = task_criteria.get(task_type, f"Does the audio demonstrate: '{label}'?")

    prompt = f"""You are an expert evaluator for speech audio characteristics.
User asked: {user_input}
Required characteristic (label): {label}
Task type: {task_type}

Evaluation Question: {criterion}

Scoring Scale: 1 (Not at all) to 5 (Perfectly).

Please strictly output the evaluation result in the following JSON format:
{{
    "score": <1 or 5>,
    "reasoning": "<Detailed reasoning>"
}}

Your evaluation (JSON only):"""

    if not audio_path.exists():
        return False, 0.0, f"Audio file not found: {audio_path}"

    success, response, error = call_llm_api(
        prompt=prompt,
        api_key=api_key,
        model=model,
        audio_path=audio_path,
        max_retry=max_retry,
        sleep_between_retry=sleep_between_retry,
    )

    if success:
        score = extract_score_from_response(response, valid_scores=[1, 2, 3, 4, 5])
        if score is not None:
            return True, score, response
        else:
            return False, 0.0, f"Score parse failed: {response}"

    return False, 0.0, error


def evaluate_implicit_audio_sample(
    sample: Dict,
    task_type: str,
    result_base_dir: Path,
    dataset_name: str,
    **kwargs
) -> Dict:
    """Evaluate a single implicit audio sample."""
    audio_path = result_base_dir / dataset_name / sample.get("model_response_audio", "")

    success, score, raw = call_llm_judge_implicit_audio(
        audio_path=audio_path,
        label=sample.get("label", ""),
        task_type=task_type,
        user_input=sample.get("User", ""),
        **kwargs
    )

    sample["audio_eval_score"] = score
    sample["audio_eval_status"] = "success" if success else "error"
    sample["audio_eval_raw_response"] = raw
    if not success:
        sample["audio_eval_reason"] = raw

    return sample


# ============================================================================
# Implicit Text Evaluator
# ============================================================================

def call_llm_judge_implicit_text(
    user_input: str,
    model_response: str,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Tuple[bool, float, str]:
    """Judge text response relevance (1-5 scale)."""
    prompt = f"""You are an expert evaluator for conversational AI systems.

Evaluate whether the model's response is contextually appropriate given the user's input.

User Input: {user_input}
Model Response: {model_response}

Rate on a scale of 1 to 5:
1: Completely irrelevant.
2: Minimal relevance.
3: Somewhat relevant but generic/incomplete.
4: Relevant and addresses most aspects.
5: Highly relevant and contextually perfect.

Please strictly output the evaluation result in the following JSON format:
{{
    "score": <1 or 2 or 3 or 4 or 5>,
    "reasoning": "<Detailed reasoning>"
}}

Your evaluation (JSON only):"""

    success, response, error = call_llm_api(
        prompt=prompt,
        api_key=api_key,
        model=model,
        max_retry=max_retry,
        sleep_between_retry=sleep_between_retry,
        max_tokens=512,
    )

    if success:
        score = extract_score_from_response(response, valid_scores=[1, 2, 3, 4, 5])
        if score is not None:
            return True, score, response
        else:
            return False, 0.0, f"Parse error: {response}"

    return False, 0.0, error


def evaluate_implicit_text_sample(sample: Dict, **kwargs) -> Dict:
    """Evaluate a single implicit text sample."""
    success, score, raw = call_llm_judge_implicit_text(
        user_input=sample.get("User", ""),
        model_response=sample.get("model_response_text", ""),
        **kwargs
    )

    sample["text_eval_score"] = score
    sample["text_eval_status"] = "success" if success else "error"
    sample["text_eval_raw_response"] = raw
    if not success:
        sample["text_eval_reason"] = raw

    return sample


# ============================================================================
# Multi-round Evaluators
# ============================================================================

def evaluate_multi_round_audio_sample(
    sample: Dict,
    dataset_name: str,
    result_base_dir: Path,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Dict:
    """Evaluate all 4 rounds of a multi-round audio sample."""
    for r in range(1, 5):
        status_key = f"audio_eval_status_round{r}"
        if sample.get(status_key) == "success":
            continue

        label = sample.get(f"label{r}", "")
        audio_rel_path = sample.get(f"model{r}_response_audio", "")
        if not label or not audio_rel_path:
            continue

        audio_path = result_base_dir / dataset_name / audio_rel_path

        prompt = f"""You are an expert evaluator for speech audio characteristics.
Round: {r}
User Input: {sample.get(f'user{r}', '')}
Required characteristic (label): {label}

Question: Does the speech in this audio CLEARLY demonstrate the characteristic: "{label}"?

Please strictly output the evaluation result in the following JSON format:
{{
    "score": <1 or 5>,
    "reasoning": "<Detailed reasoning>"
}}

Scoring criteria:
- 5: Yes, the audio clearly demonstrates the "{label}" characteristic
- 1: No, the audio does not demonstrate the "{label}" characteristic

Your evaluation (JSON only):"""

        success, response, error = call_llm_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            audio_path=audio_path,
            max_retry=max_retry,
            sleep_between_retry=sleep_between_retry,
        )

        if success:
            score = extract_score_from_response(response, valid_scores=[1, 5])
            if score is not None:
                sample[f"audio_eval_score_round{r}"] = score
                sample[status_key] = "success"
            else:
                sample[f"audio_eval_score_round{r}"] = 0.0
                sample[status_key] = "error"
        else:
            sample[f"audio_eval_score_round{r}"] = 0.0
            sample[status_key] = "error"
            sample[f"audio_eval_reason_round{r}"] = error

        sample[f"audio_eval_raw_round{r}"] = response if success else error

    return sample


def evaluate_multi_round_text_sample(
    sample: Dict,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Dict:
    """Evaluate all 4 rounds of a multi-round text sample."""
    def build_context(sample: Dict, current_round: int) -> str:
        parts = []
        for r in range(1, current_round):
            user = sample.get(f"user{r}", "")
            model_resp = sample.get(f"model{r}", "")
            if user:
                parts.append(f"User (Round {r}): {user}")
            if model_resp:
                parts.append(f"Assistant (Round {r}): {model_resp}")
        return "\n".join(parts)

    for r in range(1, 5):
        status_key = f"text_eval_status_round{r}"
        if sample.get(status_key) == "success":
            continue

        user_input = sample.get(f"user{r}", "")
        model_response = sample.get(f"model{r}_response_text", "")
        if not user_input or not model_response:
            continue

        history = build_context(sample, r)
        history_str = f"Conversation History:\n{history}\n" if history else ""

        prompt = f"""You are an expert evaluator for conversational AI systems.
{history_str}
Current Round {r}:
User: {user_input}
Assistant: {model_response}

Rate the assistant's response on a scale of 1 to 5:
- 5: Perfect response. Coherent, helpful, and contextually accurate.
- 4: Good response. Mostly accurate with minor issues.
- 3: Average. Some relevance but lacks depth.
- 2: Poor. Noticeable mistakes or off-topic.
- 1: Fail. Completely irrelevant or nonsensical.

Please strictly output the evaluation result in the following JSON format:
{{
    "score": <1 or 2 or 3 or 4 or 5>,
    "reasoning": "<Detailed reasoning>"
}}

Your evaluation (JSON only):"""

        success, response, error = call_llm_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_retry=max_retry,
            sleep_between_retry=sleep_between_retry,
            max_tokens=512,
        )

        if success:
            score = extract_score_from_response(response, valid_scores=[1, 2, 3, 4, 5])
            if score is not None:
                sample[f"text_eval_score_round{r}"] = score
                sample[status_key] = "success"
            else:
                sample[f"text_eval_score_round{r}"] = 0.0
                sample[status_key] = "error"
        else:
            sample[f"text_eval_score_round{r}"] = 0.0
            sample[status_key] = "error"
            sample[f"text_eval_reason_round{r}"] = error

        sample[f"text_eval_raw_round{r}"] = response if success else error

    return sample


# ============================================================================
# Dataset Evaluation Functions
# ============================================================================

def append_result_to_jsonl(file_path: Path, data: Dict, lock: threading.Lock):
    """Thread-safe append to JSONL file."""
    with lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def evaluate_acoustic_dataset(
    result_json_path: Path,
    dataset_name: str,
    eval_type: str,
    result_base_dir: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
    concurrency: int = 4,
    overwrite: bool = False,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Evaluate an Acoustic dataset.

    Args:
        eval_type: One of 'explicit_generation', 'explicit_understanding',
                   'implicit_audio', 'implicit_text', 'multi_audio', 'multi_text'
    """
    if not result_json_path.exists():
        print(f"Result file not found: {result_json_path}")
        return [], {}

    with open(result_json_path, 'r', encoding='utf-8') as f:
        original_samples = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{dataset_name}_{eval_type}_stream.jsonl"
    final_path = output_dir / f"{dataset_name}_{eval_type}_results.json"

    # Determine success check key based on eval type
    status_keys = {
        'explicit_generation': 'eval_result',
        'explicit_understanding': 'eval_result',
        'implicit_audio': 'audio_eval_status',
        'implicit_text': 'text_eval_status',
        'multi_audio': None,  # Special handling
        'multi_text': None,   # Special handling
    }
    status_key = status_keys.get(eval_type)

    # Load existing progress
    success_ids = set()
    history_records = {}

    if jsonl_path.exists() and not overwrite:
        print(f"Loading existing progress for {dataset_name}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    history_records[item["id"]] = item

                    if eval_type in ['multi_audio', 'multi_text']:
                        # Check all 4 rounds
                        prefix = 'audio' if eval_type == 'multi_audio' else 'text'
                        is_complete = all(
                            item.get(f"{prefix}_eval_status_round{r}") == "success"
                            for r in range(1, 5)
                        )
                        if is_complete:
                            success_ids.add(item["id"])
                    else:
                        if status_key and item.get(status_key) in ["success", "correct", "incorrect"]:
                            success_ids.add(item["id"])
                except:
                    continue
        print(f"Found {len(success_ids)} completed samples")
    elif overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    # Prepare samples to evaluate
    samples_to_eval = []
    for s in original_samples:
        if s["id"] in success_ids:
            continue
        # Use history record if available (for retry)
        samples_to_eval.append(history_records.get(s["id"], s))

    if not samples_to_eval:
        print("All samples already evaluated")
    else:
        print(f"Evaluating {len(samples_to_eval)} samples...")
        file_lock = threading.Lock()

        # Select evaluation function
        eval_funcs = {
            'explicit_generation': lambda s: evaluate_explicit_generation_sample(
                s, dataset_name.replace("explicit_generation_", ""),
                result_base_dir, dataset_name,
                api_key=api_key, model=model,
                max_retry=max_retry, sleep_between_retry=sleep_between_retry
            ),
            'explicit_understanding': lambda s: evaluate_explicit_understanding_sample(
                s, dataset_name.replace("explicit_understanding_", ""),
                api_key=api_key, model=model,
                max_retry=max_retry, sleep_between_retry=sleep_between_retry
            ),
            'implicit_audio': lambda s: evaluate_implicit_audio_sample(
                s, dataset_name.replace("implicit_", "").replace("_generation", ""),
                result_base_dir, dataset_name,
                api_key=api_key, model=model,
                max_retry=max_retry, sleep_between_retry=sleep_between_retry
            ),
            'implicit_text': lambda s: evaluate_implicit_text_sample(
                s, api_key=api_key, model=model,
                max_retry=max_retry, sleep_between_retry=sleep_between_retry
            ),
            'multi_audio': lambda s: evaluate_multi_round_audio_sample(
                s, dataset_name, result_base_dir,
                api_key=api_key, model=model,
                max_retry=max_retry, sleep_between_retry=sleep_between_retry
            ),
            'multi_text': lambda s: evaluate_multi_round_text_sample(
                s, api_key=api_key, model=model,
                max_retry=max_retry, sleep_between_retry=sleep_between_retry
            ),
        }

        eval_func = eval_funcs.get(eval_type)
        if not eval_func:
            raise ValueError(f"Unknown eval_type: {eval_type}")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(eval_func, s): s for s in samples_to_eval}

            for fut in tqdm(as_completed(futures), total=len(samples_to_eval),
                           desc=f"Evaluating {dataset_name}"):
                res = fut.result()
                append_result_to_jsonl(jsonl_path, res, file_lock)
                history_records[res["id"]] = res

    # Merge final results
    final_samples = [history_records.get(s["id"], s) for s in original_samples]

    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    # Compute statistics
    stats = compute_acoustic_stats(final_samples, eval_type)

    return final_samples, stats


def compute_acoustic_stats(samples: List[Dict], eval_type: str) -> Dict[str, Any]:
    """Compute evaluation statistics based on eval type."""
    total = len(samples)

    if eval_type in ['explicit_generation', 'explicit_understanding']:
        correct = sum(1 for s in samples if s.get("eval_correct") == 1)
        errors = sum(1 for s in samples if s.get("eval_result") == "error")
        return {
            "total": total,
            "correct": correct,
            "errors": errors,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    elif eval_type == 'implicit_audio':
        success = [s for s in samples if s.get("audio_eval_status") == "success"]
        scores = [s.get("audio_eval_score", 0) for s in success]
        return {
            "total": total,
            "success": len(success),
            "errors": total - len(success),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }

    elif eval_type == 'implicit_text':
        success = [s for s in samples if s.get("text_eval_status") == "success"]
        scores = [s.get("text_eval_score", 0) for s in success]
        return {
            "total": total,
            "success": len(success),
            "errors": total - len(success),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
        }

    elif eval_type in ['multi_audio', 'multi_text']:
        prefix = 'audio' if eval_type == 'multi_audio' else 'text'
        round_stats = {}
        all_scores = []

        for r in range(1, 5):
            scores = [
                s.get(f"{prefix}_eval_score_round{r}", 0)
                for s in samples
                if s.get(f"{prefix}_eval_status_round{r}") == "success"
            ]
            all_scores.extend(scores)
            round_stats[f"round{r}"] = {
                "avg": sum(scores) / len(scores) if scores else 0.0,
                "success": len(scores),
            }

        return {
            "total": total,
            "overall_avg": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "round_stats": round_stats,
        }

    return {"total": total}


def print_acoustic_stats(stats: Dict, dataset_name: str, eval_type: str):
    """Print evaluation statistics."""
    print(f"\n[{dataset_name}] {eval_type} Results:")

    if "accuracy" in stats:
        print(f"  Accuracy: {stats['accuracy']*100:.2f}%")
        print(f"  Correct: {stats['correct']}/{stats['total']}")
        print(f"  Errors: {stats['errors']}")
    elif "avg_score" in stats:
        print(f"  Average Score: {stats['avg_score']:.2f}/5.0")
        print(f"  Success: {stats['success']}/{stats['total']}")
        print(f"  Errors: {stats['errors']}")
    elif "overall_avg" in stats:
        print(f"  Overall Average: {stats['overall_avg']:.2f}/5.0")
        for r in range(1, 5):
            rs = stats['round_stats'].get(f"round{r}", {})
            print(f"  Round {r}: {rs.get('avg', 0):.2f} ({rs.get('success', 0)} success)")
