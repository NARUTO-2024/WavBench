#!/usr/bin/env python3
"""
WavBench Colloquial Evaluator

Evaluates Basic and Pro dataset responses using task-specific prompts.
Uses LLM to judge response quality on a 1/3/5 scale.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..api.utils import call_llm_api, parse_json_response


# Task type to prompt file mapping
TASK_PROMPT_FILES = {
    "code": "code.txt",
    "creative": "creative.txt",
    "instruction": "instruction.txt",
    "logic": "logic.txt",
    "math": "math.txt",
    "qa": "qa.txt",
    "satety": "satety.txt",  # Note: typo in original filename
}


def load_prompt_template(task_type: str, prompt_dir: Path) -> str:
    """Load evaluation prompt template for task type."""
    filename = TASK_PROMPT_FILES.get(task_type)
    if not filename:
        raise ValueError(f"Unknown task type: {task_type}")

    prompt_path = prompt_dir / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def build_evaluation_prompt(
    template: str,
    spoken_instruction: str,
    spoken_reference: str,
    model_response: str,
) -> str:
    """Build complete evaluation prompt with sample data."""
    # Append the actual data to evaluate
    data_section = f"""
**Data to Evaluate:**

spoken_instruction: {spoken_instruction}

spoken_reference: {spoken_reference}

test_model_response: {model_response}

Please evaluate the test_model_response according to the criteria above.
"""
    return template + data_section


def evaluate_one_sample(
    sample: Dict,
    task_type: str,
    prompt_template: str,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
) -> Dict:
    """Evaluate a single sample."""
    # Build evaluation prompt
    prompt = build_evaluation_prompt(
        template=prompt_template,
        spoken_instruction=sample.get("spoken_instruction", ""),
        spoken_reference=sample.get("spoken_reference", ""),
        model_response=sample.get("response", "") or sample.get("model_response", ""),
    )

    # Call LLM API
    success, response_text, error = call_llm_api(
        prompt=prompt,
        api_key=api_key,
        model=model,
        max_retry=max_retry,
        sleep_between_retry=sleep_between_retry,
        max_tokens=512,
    )

    if success:
        # Parse JSON response
        result = parse_json_response(response_text)
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "")

        # Validate score
        if score in [1, 3, 5]:
            sample["eval_score"] = score
            sample["eval_status"] = "success"
            sample["eval_reasoning"] = reasoning
        else:
            sample["eval_score"] = 0
            sample["eval_status"] = "error"
            sample["eval_reasoning"] = f"Invalid score: {score}"
            sample["eval_raw_response"] = response_text
    else:
        sample["eval_score"] = 0
        sample["eval_status"] = "error"
        sample["eval_reasoning"] = error
        sample["eval_raw_response"] = ""

    return sample


def append_result_to_jsonl(file_path: Path, data: Dict, lock: threading.Lock):
    """Thread-safe append to JSONL file."""
    with lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def evaluate_colloquial_dataset(
    result_json_path: Path,
    task_type: str,
    dataset_name: str,
    output_dir: Path,
    prompt_dir: Path,
    api_key: str,
    model: str,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
    concurrency: int = 4,
    overwrite: bool = False,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Evaluate a Colloquial (Basic/Pro) dataset.

    Args:
        result_json_path: Path to inference results JSON file
        task_type: Task type (code, creative, etc.)
        dataset_name: Dataset name for output files
        output_dir: Output directory
        prompt_dir: Directory containing prompt templates
        api_key: API key
        model: Model name
        max_retry: Max retry attempts
        sleep_between_retry: Sleep between retries
        concurrency: Number of concurrent workers
        overwrite: Whether to overwrite existing results

    Returns:
        Tuple of (evaluated_samples, statistics)
    """
    if not result_json_path.exists():
        print(f"Result file not found: {result_json_path}")
        return [], {}

    # Load results
    with open(result_json_path, 'r', encoding='utf-8') as f:
        original_samples = json.load(f)

    # Load prompt template
    prompt_template = load_prompt_template(task_type, prompt_dir)

    # Setup output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output_path = output_dir / f"{dataset_name}_eval_stream.jsonl"
    final_output_path = output_dir / f"{dataset_name}_eval_results.json"

    # Load existing progress
    success_ids = set()
    history_records = []

    if jsonl_output_path.exists() and not overwrite:
        print(f"Loading existing progress...")
        with open(jsonl_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    history_records.append(item)
                    if item.get("eval_status") == "success":
                        success_ids.add(item.get("id"))
                except:
                    continue
        print(f"Found {len(success_ids)} completed samples")
    elif overwrite and jsonl_output_path.exists():
        jsonl_output_path.unlink()

    # Filter samples to evaluate
    samples_to_eval = [s for s in original_samples if s.get("id") not in success_ids]

    if not samples_to_eval:
        print("All samples already evaluated")
    else:
        print(f"Evaluating {len(samples_to_eval)} samples...")
        file_write_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    evaluate_one_sample, s, task_type, prompt_template,
                    api_key, model, max_retry, sleep_between_retry
                ): s for s in samples_to_eval
            }

            for fut in tqdm(as_completed(futures), total=len(samples_to_eval),
                           desc=f"Evaluating {dataset_name}"):
                res = fut.result()
                append_result_to_jsonl(jsonl_output_path, res, file_write_lock)
                history_records.append(res)

    # Merge results
    id_to_sample = {s["id"]: s for s in history_records}
    final_samples = [id_to_sample.get(s["id"], s) for s in original_samples]

    # Save final results
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    # Compute statistics
    stats = compute_colloquial_stats(final_samples)

    return final_samples, stats


def compute_colloquial_stats(samples: List[Dict]) -> Dict[str, Any]:
    """Compute evaluation statistics."""
    total = len(samples)
    success_samples = [s for s in samples if s.get("eval_status") == "success"]
    scores = [s.get("eval_score", 0) for s in success_samples]

    # Score distribution
    score_dist = defaultdict(int)
    for score in scores:
        score_dist[score] += 1

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_samples": total,
        "success_count": len(success_samples),
        "error_count": total - len(success_samples),
        "avg_score": avg_score,
        "score_distribution": dict(score_dist),
    }


def print_colloquial_stats(stats: Dict, dataset_name: str):
    """Print evaluation statistics."""
    print(f"\n[{dataset_name}] Results:")
    print(f"  Total: {stats['total_samples']}")
    print(f"  Success: {stats['success_count']}")
    print(f"  Errors: {stats['error_count']}")
    print(f"  Average Score: {stats['avg_score']:.2f}")
    print(f"  Score Distribution: {stats['score_distribution']}")
