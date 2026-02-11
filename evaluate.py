#!/usr/bin/env python3
"""
WavBench Evaluation Script

Unified evaluation script for all WavBench datasets.
Supports Colloquial (Basic/Pro) and Acoustic evaluations.
"""

import argparse
import json
from pathlib import Path

from src.evaluator import (
    evaluate_colloquial_dataset,
    print_colloquial_stats,
    evaluate_acoustic_dataset,
    print_acoustic_stats,
)


# Dataset type mappings
COLLOQUIAL_TASKS = ["code", "creative", "instruction", "logic", "math", "qa", "satety"]

ACOUSTIC_EVAL_TYPES = {
    # Explicit Generation datasets
    "explicit_generation_accent": "explicit_generation",
    "explicit_generation_emotion": "explicit_generation",
    "explicit_generation_gender": "explicit_generation",
    "explicit_generation_age": "explicit_generation",
    "explicit_generation_pitch": "explicit_generation",
    "explicit_generation_speed": "explicit_generation",
    "explicit_generation_volume": "explicit_generation",
    "explicit_generation_lang": "explicit_generation",
    "explicit_generation_music": "explicit_generation",
    "explicit_generation_audio": "explicit_generation",

    # Explicit Understanding datasets
    "explicit_understanding_music": "explicit_understanding",
    "explicit_understanding_emotion": "explicit_understanding",
    "explicit_understanding_gender": "explicit_understanding",
    "explicit_understanding_age": "explicit_understanding",
    "explicit_understanding_accent": "explicit_understanding",
    "explicit_understanding_lang": "explicit_understanding",
    "explicit_understanding_audio": "explicit_understanding",
    "explicit_understanding_pitch": "explicit_understanding",
    "explicit_understanding_speed": "explicit_understanding",
    "explicit_understanding_volume": "explicit_understanding",

    # Implicit datasets (need both audio and text evaluation)
    "implicit_age_generation": "implicit",
    "implicit_emotion_generation": "implicit",
    "implicit_pitch_generation": "implicit",
    "implicit_speed_generation": "implicit",
    "implicit_understanding": "implicit",

    # Multi-round datasets (need both audio and text evaluation)
    "multi_round_generation": "multi",
    "multi_round_understanding": "multi",
}


def evaluate_colloquial(args):
    """Evaluate Colloquial (Basic/Pro) datasets."""
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    prompt_dir = Path(args.prompt_dir)

    # Determine datasets to evaluate
    if args.dataset == "all":
        datasets = []
        for level in ["basic", "pro"]:
            for task in COLLOQUIAL_TASKS:
                datasets.append((level, task))
    else:
        # Parse dataset name like "basic_code" or "pro_creative"
        parts = args.dataset.split("_", 1)
        if len(parts) == 2 and parts[0] in ["basic", "pro"]:
            datasets = [(parts[0], parts[1])]
        else:
            print(f"Invalid dataset name: {args.dataset}")
            return

    all_stats = {}
    for level, task in datasets:
        dataset_name = f"{level}_{task}"
        # Path format: {result_dir}/{level}/{task}/{level}_{task}.json
        result_path = result_dir / level / task / f"{dataset_name}.json"

        if not result_path.exists():
            print(f"Skipping {dataset_name}: result file not found at {result_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*60}")

        samples, stats = evaluate_colloquial_dataset(
            result_json_path=result_path,
            task_type=task,
            dataset_name=dataset_name,
            output_dir=output_dir / level / task,
            prompt_dir=prompt_dir,
            api_key=args.api_key,
            model=args.model,
            max_retry=args.max_retry,
            sleep_between_retry=args.sleep_between_retry,
            concurrency=args.concurrency,
            overwrite=args.overwrite,
        )

        print_colloquial_stats(stats, dataset_name)
        all_stats[dataset_name] = stats

    # Print summary
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for name, stats in all_stats.items():
            print(f"{name:30s}: {stats['avg_score']:.2f} avg score")


def evaluate_acoustic(args):
    """Evaluate Acoustic datasets."""
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)

    # Determine datasets to evaluate
    if args.dataset == "all":
        datasets = list(ACOUSTIC_EVAL_TYPES.keys())
    else:
        datasets = [args.dataset]

    all_stats = {}
    for dataset_name in datasets:
        eval_category = ACOUSTIC_EVAL_TYPES.get(dataset_name)
        if not eval_category:
            print(f"Unknown acoustic dataset: {dataset_name}")
            continue

        # Determine result path
        # main.py outputs to: output/acoustic/{task}/{task}.json or output/acoustic/multi/{task}/{task}.json
        if dataset_name.startswith("multi_round"):
            result_path = result_dir / "acoustic" / "multi" / dataset_name / f"{dataset_name}.json"
        else:
            result_path = result_dir / "acoustic" / dataset_name / f"{dataset_name}.json"

        if not result_path.exists():
            print(f"Skipping {dataset_name}: result file not found at {result_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*60}")

        # Determine evaluation types based on category
        if eval_category == "explicit_generation":
            eval_types = ["explicit_generation"]
        elif eval_category == "explicit_understanding":
            eval_types = ["explicit_understanding"]
        elif eval_category == "implicit":
            eval_types = ["implicit_audio", "implicit_text"]
        elif eval_category == "multi":
            eval_types = ["multi_audio", "multi_text"]
        else:
            eval_types = []

        for eval_type in eval_types:
            print(f"\n--- {eval_type} ---")

            # Determine result base dir for audio files
            # main.py outputs audio to: output/acoustic/{task}/ or output/acoustic/multi/{task}/
            if dataset_name.startswith("multi_round"):
                result_base = result_dir / "acoustic" / "multi"
            else:
                result_base = result_dir / "acoustic"

            samples, stats = evaluate_acoustic_dataset(
                result_json_path=result_path,
                dataset_name=dataset_name,
                eval_type=eval_type,
                result_base_dir=result_base,
                output_dir=output_dir / dataset_name,
                api_key=args.api_key,
                model=args.model,
                max_retry=args.max_retry,
                sleep_between_retry=args.sleep_between_retry,
                concurrency=args.concurrency,
                overwrite=args.overwrite,
            )

            print_acoustic_stats(stats, dataset_name, eval_type)
            all_stats[f"{dataset_name}_{eval_type}"] = stats

    # Print summary
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for name, stats in all_stats.items():
            if "accuracy" in stats:
                print(f"{name:50s}: {stats['accuracy']*100:.2f}% accuracy")
            elif "avg_score" in stats:
                print(f"{name:50s}: {stats['avg_score']:.2f}/5.0 avg")
            elif "overall_avg" in stats:
                print(f"{name:50s}: {stats['overall_avg']:.2f}/5.0 overall")


def main():
    parser = argparse.ArgumentParser(description="WavBench Evaluation")

    # Common arguments
    parser.add_argument("--eval_type", type=str, required=True,
                        choices=["colloquial", "acoustic"],
                        help="Evaluation type: colloquial (Basic/Pro) or acoustic")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset to evaluate (e.g., basic_code, explicit_generation_emotion) or 'all'")
    parser.add_argument("--result_dir", type=str, default="./output",
                        help="Directory containing inference results")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")

    # API arguments
    parser.add_argument("--api_key", type=str,
                        default=None,
                        help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--model", type=str,
                        default="gemini-3-pro-preview",
                        help="Model name for evaluation")

    # Colloquial-specific arguments
    parser.add_argument("--prompt_dir", type=str, default="./src/prompt",
                        help="Directory containing prompt templates")

    # Execution arguments
    parser.add_argument("--max_retry", type=int, default=3,
                        help="Maximum retry attempts")
    parser.add_argument("--sleep_between_retry", type=int, default=2,
                        help="Sleep time between retries")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent workers")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing evaluation results")

    args = parser.parse_args()

    if args.eval_type == "colloquial":
        evaluate_colloquial(args)
    elif args.eval_type == "acoustic":
        evaluate_acoustic(args)


if __name__ == "__main__":
    main()
