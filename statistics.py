#!/usr/bin/env python3
"""
WavBench Evaluation Statistics Script

Aggregates evaluation results from eval_results/ directory.
Directly reads the JSON output from evaluate.py.

Score normalization:
- 1/3/5 scale -> multiply by 20 (max 100)
- 1-5 scale -> multiply by 20 (max 100)
- Accuracy (correct/incorrect) -> multiply by 100 (percentage)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ============================================================================
# Category Definitions
# ============================================================================

COLLOQUIAL_TASKS = ["code", "creative", "instruction", "logic", "math", "qa", "satety"]

TASK_DISPLAY_NAMES = {"satety": "safety"}

EXPLICIT_SUBTYPES = [
    "accent", "age", "emotion", "gender", "lang", "pitch", "speed", "volume",
    "audio", "music"
]

IMPLICIT_DATASETS = [
    "implicit_age_generation", "implicit_emotion_generation",
    "implicit_pitch_generation", "implicit_speed_generation",
    "implicit_understanding"
]

MULTI_ROUND_DATASETS = ["multi_round_generation", "multi_round_understanding"]


# ============================================================================
# Result Parsing
# ============================================================================

def parse_colloquial_json(file_path: Path) -> Tuple[float, int]:
    """Parse Colloquial evaluation results JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores = []
    for item in data:
        if item.get("eval_status") == "success":
            score = item.get("eval_score")
            if score is not None:
                scores.append(score)

    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


def parse_acoustic_json(file_path: Path, eval_type: str) -> Tuple[float, int]:
    """Parse Acoustic evaluation results JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if eval_type in ["explicit_generation", "explicit_understanding"]:
        # Accuracy-based evaluation
        correct = sum(1 for item in data if item.get("eval_correct") == 1)
        total = sum(1 for item in data if item.get("eval_result") in ["correct", "incorrect"])
        if total == 0:
            return 0.0, 0
        return correct / total, total

    elif eval_type == "implicit_audio":
        scores = [item.get("audio_eval_score", 0) for item in data
                  if item.get("audio_eval_status") == "success"]
        if not scores:
            return 0.0, 0
        return sum(scores) / len(scores), len(scores)

    elif eval_type == "implicit_text":
        scores = [item.get("text_eval_score", 0) for item in data
                  if item.get("text_eval_status") == "success"]
        if not scores:
            return 0.0, 0
        return sum(scores) / len(scores), len(scores)

    elif eval_type == "multi_audio":
        all_scores = []
        for item in data:
            for r in range(1, 5):
                if item.get(f"audio_eval_status_round{r}") == "success":
                    score = item.get(f"audio_eval_score_round{r}")
                    if score is not None:
                        all_scores.append(score)
        if not all_scores:
            return 0.0, 0
        return sum(all_scores) / len(all_scores), len(all_scores)

    elif eval_type == "multi_text":
        all_scores = []
        for item in data:
            for r in range(1, 5):
                if item.get(f"text_eval_status_round{r}") == "success":
                    score = item.get(f"text_eval_score_round{r}")
                    if score is not None:
                        all_scores.append(score)
        if not all_scores:
            return 0.0, 0
        return sum(all_scores) / len(all_scores), len(all_scores)

    return 0.0, 0


# ============================================================================
# Result Collection
# ============================================================================

def collect_results(eval_dir: Path) -> Dict:
    """Collect all evaluation results from eval_results directory."""
    results = {
        "basic": {},
        "pro": {},
        "explicit_understanding": {},
        "explicit_generation": {},
        "implicit_audio": {},
        "implicit_text": {},
        "multi_audio": {},
        "multi_text": {},
    }

    if not eval_dir.exists():
        print(f"Warning: Evaluation directory not found: {eval_dir}")
        return results

    # Collect Basic results
    for task in COLLOQUIAL_TASKS:
        result_file = eval_dir / "basic" / task / f"basic_{task}_eval_results.json"
        if result_file.exists():
            avg_score, count = parse_colloquial_json(result_file)
            if count > 0:
                results["basic"][task] = {
                    "raw_score": avg_score,
                    "normalized_score": avg_score * 20,
                    "count": count,
                }

    # Collect Pro results
    for task in COLLOQUIAL_TASKS:
        result_file = eval_dir / "pro" / task / f"pro_{task}_eval_results.json"
        if result_file.exists():
            avg_score, count = parse_colloquial_json(result_file)
            if count > 0:
                results["pro"][task] = {
                    "raw_score": avg_score,
                    "normalized_score": avg_score * 20,
                    "count": count,
                }

    # Collect Explicit Understanding results
    for subtype in EXPLICIT_SUBTYPES:
        dataset_name = f"explicit_understanding_{subtype}"
        result_file = eval_dir / dataset_name / f"{dataset_name}_explicit_understanding_results.json"
        if result_file.exists():
            accuracy, count = parse_acoustic_json(result_file, "explicit_understanding")
            if count > 0:
                results["explicit_understanding"][subtype] = {
                    "raw_score": accuracy,
                    "normalized_score": accuracy * 100,
                    "count": count,
                }

    # Collect Explicit Generation results
    for subtype in EXPLICIT_SUBTYPES:
        dataset_name = f"explicit_generation_{subtype}"
        result_file = eval_dir / dataset_name / f"{dataset_name}_explicit_generation_results.json"
        if result_file.exists():
            accuracy, count = parse_acoustic_json(result_file, "explicit_generation")
            if count > 0:
                results["explicit_generation"][subtype] = {
                    "raw_score": accuracy,
                    "normalized_score": accuracy * 100,
                    "count": count,
                }

    # Collect Implicit Audio results
    for dataset_name in IMPLICIT_DATASETS:
        result_file = eval_dir / dataset_name / f"{dataset_name}_implicit_audio_results.json"
        if result_file.exists():
            avg_score, count = parse_acoustic_json(result_file, "implicit_audio")
            if count > 0:
                results["implicit_audio"][dataset_name] = {
                    "raw_score": avg_score,
                    "normalized_score": avg_score * 20,
                    "count": count,
                }

    # Collect Implicit Text results
    for dataset_name in IMPLICIT_DATASETS:
        result_file = eval_dir / dataset_name / f"{dataset_name}_implicit_text_results.json"
        if result_file.exists():
            avg_score, count = parse_acoustic_json(result_file, "implicit_text")
            if count > 0:
                results["implicit_text"][dataset_name] = {
                    "raw_score": avg_score,
                    "normalized_score": avg_score * 20,
                    "count": count,
                }

    # Collect Multi-round Audio results
    for dataset_name in MULTI_ROUND_DATASETS:
        result_file = eval_dir / dataset_name / f"{dataset_name}_multi_audio_results.json"
        if result_file.exists():
            avg_score, count = parse_acoustic_json(result_file, "multi_audio")
            if count > 0:
                results["multi_audio"][dataset_name] = {
                    "raw_score": avg_score,
                    "normalized_score": avg_score * 20,
                    "count": count,
                }

    # Collect Multi-round Text results
    for dataset_name in MULTI_ROUND_DATASETS:
        result_file = eval_dir / dataset_name / f"{dataset_name}_multi_text_results.json"
        if result_file.exists():
            avg_score, count = parse_acoustic_json(result_file, "multi_text")
            if count > 0:
                results["multi_text"][dataset_name] = {
                    "raw_score": avg_score,
                    "normalized_score": avg_score * 20,
                    "count": count,
                }

    return results


# ============================================================================
# Output Formatting
# ============================================================================

def format_output(results: Dict) -> str:
    """Format results into readable text output."""
    lines = []
    lines.append("=" * 80)
    lines.append("WavBench Evaluation Statistics")
    lines.append("=" * 80)
    lines.append("")

    def get_display_name(name: str) -> str:
        return TASK_DISPLAY_NAMES.get(name, name)

    # Basic results
    if results["basic"]:
        lines.append("-" * 40)
        lines.append("Basic")
        lines.append("-" * 40)
        basic_scores = []
        for task in COLLOQUIAL_TASKS:
            if task in results["basic"]:
                display_name = get_display_name(task)
                score = results["basic"][task]["normalized_score"]
                count = results["basic"][task]["count"]
                lines.append(f"  {display_name:20s}: {score:6.2f} (n={count})")
                basic_scores.append(score)

        if basic_scores:
            overall = sum(basic_scores) / len(basic_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}")
        lines.append("")

    # Pro results
    if results["pro"]:
        lines.append("-" * 40)
        lines.append("Pro")
        lines.append("-" * 40)
        pro_scores = []
        for task in COLLOQUIAL_TASKS:
            if task in results["pro"]:
                display_name = get_display_name(task)
                score = results["pro"][task]["normalized_score"]
                count = results["pro"][task]["count"]
                lines.append(f"  {display_name:20s}: {score:6.2f} (n={count})")
                pro_scores.append(score)

        if pro_scores:
            overall = sum(pro_scores) / len(pro_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}")
        lines.append("")

    # Explicit Understanding results
    if results["explicit_understanding"]:
        lines.append("-" * 40)
        lines.append("Explicit Understanding")
        lines.append("-" * 40)
        eu_scores = []
        for subtype in EXPLICIT_SUBTYPES:
            if subtype in results["explicit_understanding"]:
                score = results["explicit_understanding"][subtype]["normalized_score"]
                count = results["explicit_understanding"][subtype]["count"]
                lines.append(f"  {subtype:20s}: {score:6.2f}% (n={count})")
                eu_scores.append(score)

        if eu_scores:
            overall = sum(eu_scores) / len(eu_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}%")
        lines.append("")

    # Explicit Generation results
    if results["explicit_generation"]:
        lines.append("-" * 40)
        lines.append("Explicit Generation")
        lines.append("-" * 40)
        eg_scores = []
        for subtype in EXPLICIT_SUBTYPES:
            if subtype in results["explicit_generation"]:
                score = results["explicit_generation"][subtype]["normalized_score"]
                count = results["explicit_generation"][subtype]["count"]
                lines.append(f"  {subtype:20s}: {score:6.2f}% (n={count})")
                eg_scores.append(score)

        if eg_scores:
            overall = sum(eg_scores) / len(eg_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}%")
        lines.append("")

    # Implicit Audio results
    if results["implicit_audio"]:
        lines.append("-" * 40)
        lines.append("Implicit Audio")
        lines.append("-" * 40)
        ia_scores = []
        for dataset_name in IMPLICIT_DATASETS:
            if dataset_name in results["implicit_audio"]:
                score = results["implicit_audio"][dataset_name]["normalized_score"]
                count = results["implicit_audio"][dataset_name]["count"]
                short_name = dataset_name.replace("implicit_", "").replace("_generation", "")
                lines.append(f"  {short_name:20s}: {score:6.2f} (n={count})")
                ia_scores.append(score)

        if ia_scores:
            overall = sum(ia_scores) / len(ia_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}")
        lines.append("")

    # Implicit Text results
    if results["implicit_text"]:
        lines.append("-" * 40)
        lines.append("Implicit Text")
        lines.append("-" * 40)
        it_scores = []
        for dataset_name in IMPLICIT_DATASETS:
            if dataset_name in results["implicit_text"]:
                score = results["implicit_text"][dataset_name]["normalized_score"]
                count = results["implicit_text"][dataset_name]["count"]
                short_name = dataset_name.replace("implicit_", "").replace("_generation", "")
                lines.append(f"  {short_name:20s}: {score:6.2f} (n={count})")
                it_scores.append(score)

        if it_scores:
            overall = sum(it_scores) / len(it_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}")
        lines.append("")

    # Multi-round Audio results
    if results["multi_audio"]:
        lines.append("-" * 40)
        lines.append("Multi-round Audio")
        lines.append("-" * 40)
        ma_scores = []
        for dataset_name in MULTI_ROUND_DATASETS:
            if dataset_name in results["multi_audio"]:
                score = results["multi_audio"][dataset_name]["normalized_score"]
                count = results["multi_audio"][dataset_name]["count"]
                short_name = dataset_name.replace("multi_round_", "")
                lines.append(f"  {short_name:20s}: {score:6.2f} (n={count // 4})")
                ma_scores.append(score)

        if ma_scores:
            overall = sum(ma_scores) / len(ma_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}")
        lines.append("")

    # Multi-round Text results
    if results["multi_text"]:
        lines.append("-" * 40)
        lines.append("Multi-round Text")
        lines.append("-" * 40)
        mt_scores = []
        for dataset_name in MULTI_ROUND_DATASETS:
            if dataset_name in results["multi_text"]:
                score = results["multi_text"][dataset_name]["normalized_score"]
                count = results["multi_text"][dataset_name]["count"]
                short_name = dataset_name.replace("multi_round_", "")
                lines.append(f"  {short_name:20s}: {score:6.2f} (n={count // 4})")
                mt_scores.append(score)

        if mt_scores:
            overall = sum(mt_scores) / len(mt_scores)
            lines.append(f"  {'overall':20s}: {overall:6.2f}")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WavBench Evaluation Statistics")
    parser.add_argument("--eval_dir", type=str, default="./eval_results",
                        help="Directory containing evaluation results")
    parser.add_argument("--output", type=str, default="./statistics.txt",
                        help="Output file path")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output)

    print(f"Collecting results from: {eval_dir}")
    results = collect_results(eval_dir)

    # Check if any results were found
    total_results = sum(len(v) for v in results.values())
    if total_results == 0:
        print("No evaluation results found!")
        return

    # Format and output
    output_text = format_output(results)
    print(output_text)

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nStatistics saved to: {output_path}")


if __name__ == "__main__":
    main()
