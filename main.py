#!/usr/bin/env python3
"""
WavBench: A Comprehensive Benchmark for Evaluating Voice Assistants

Unified inference script supporting:
- Colloquial datasets (Basic/Pro): Single-round text-only inference
- Acoustic datasets: Single-round and multi-round inference with audio output

Usage:
    # Colloquial inference
    python main.py --model step_audio2 --data basic_code

    # Acoustic inference (with audio output)
    python main.py --model step_audio2 --data acoustic_explicit_generation_emotion --audio_output

    # List all available datasets
    python main.py --list_datasets
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from loguru import logger

from src.models import model_cls_mapping


# ============================================================================
# Dataset Configuration
# ============================================================================

# Colloquial task types
COLLOQUIAL_TASKS = ["code", "creative", "instruction", "logic", "math", "qa", "satety"]

# Acoustic dataset types
ACOUSTIC_SINGLE_ROUND = [
    "explicit_generation_accent", "explicit_generation_age", "explicit_generation_audio",
    "explicit_generation_emotion", "explicit_generation_gender", "explicit_generation_lang",
    "explicit_generation_music", "explicit_generation_pitch", "explicit_generation_speed",
    "explicit_generation_volume",
    "explicit_understanding_accent", "explicit_understanding_age", "explicit_understanding_audio",
    "explicit_understanding_emotion", "explicit_understanding_gender", "explicit_understanding_lang",
    "explicit_understanding_music", "explicit_understanding_pitch", "explicit_understanding_speed",
    "explicit_understanding_volume",
    "implicit_age_generation", "implicit_emotion_generation", "implicit_pitch_generation",
    "implicit_speed_generation", "implicit_understanding",
]

ACOUSTIC_MULTI_ROUND = [
    "multi_round_generation", "multi_round_understanding",
]


# ============================================================================
# Data Loading
# ============================================================================

def load_colloquial_dataset(data_dir: Path, level: str, task: str) -> List[Dict]:
    """
    Load Colloquial (Basic/Pro) dataset.

    Args:
        data_dir: Base wavbench directory
        level: 'basic' or 'pro'
        task: Task name (code, creative, etc.)

    Returns:
        List of samples with audio_path resolved
    """
    if level == "basic":
        json_path = data_dir / "Basic" / "basic_class" / f"{task}.json"
    else:
        json_path = data_dir / "Pro" / "pro_class" / f"{task}.json"

    audio_base = data_dir / "Colloquial_audio"

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = item.copy()

        # Resolve audio path
        audio_rel = item.get("audio_path", "")
        if audio_rel.startswith("./audio/"):
            audio_rel = audio_rel[8:]
        elif audio_rel.startswith("./"):
            audio_rel = audio_rel[2:]

        audio_path = audio_base / audio_rel
        if not audio_path.exists():
            logger.warning(f"Audio not found: {audio_path}, skipping {item.get('id')}")
            continue

        sample["audio_abs_path"] = str(audio_path)
        sample["prompt"] = item.get("spoken_instruction", item.get("text", ""))
        samples.append(sample)

    return samples


def load_acoustic_single_round(data_dir: Path, dataset_name: str) -> List[Dict]:
    """
    Load Acoustic single-round dataset.

    Args:
        data_dir: Base wavbench directory
        dataset_name: Dataset name (e.g., explicit_generation_emotion)

    Returns:
        List of samples with audio_path resolved
    """
    json_path = data_dir / "Acoustic" / "json" / f"{dataset_name}.json"
    audio_base = data_dir / "Acoustic"

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = item.copy()

        # Find audio path field
        audio_rel = None
        for field in ["merge_wav", "user_input_wav", "audio", "wav_path", "audio_path"]:
            if field in item and item[field]:
                audio_rel = item[field]
                break

        if not audio_rel:
            logger.warning(f"No audio path found for {item.get('id')}")
            continue

        if audio_rel.startswith("./"):
            audio_rel = audio_rel[2:]

        audio_path = audio_base / audio_rel
        if not audio_path.exists():
            logger.warning(f"Audio not found: {audio_path}, skipping {item.get('id')}")
            continue

        sample["audio_abs_path"] = str(audio_path)
        samples.append(sample)

    return samples


def load_acoustic_multi_round(data_dir: Path, dataset_name: str) -> List[Dict]:
    """
    Load Acoustic multi-round dataset (4 rounds per sample).

    Args:
        data_dir: Base wavbench directory
        dataset_name: Dataset name (e.g., multi_round_generation)

    Returns:
        List of samples with audio_paths resolved
    """
    json_path = data_dir / "Acoustic" / "json" / f"{dataset_name}.json"
    audio_base = data_dir / "Acoustic"

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = item.copy()

        # Check all 4 rounds
        audio_paths = []
        all_exist = True

        for r in range(1, 5):
            wav_key = f"user{r}_wav"
            if wav_key not in item:
                all_exist = False
                break

            audio_rel = item[wav_key]
            if audio_rel.startswith("./"):
                audio_rel = audio_rel[2:]

            audio_path = audio_base / audio_rel
            if not audio_path.exists():
                logger.warning(f"Audio not found for round {r}: {audio_path}")
                all_exist = False
                break

            audio_paths.append(str(audio_path))

        if not all_exist:
            continue

        sample["audio_paths"] = audio_paths
        samples.append(sample)

    return samples


def discover_datasets(data_dir: Path) -> Dict[str, List[str]]:
    """
    Discover all available datasets.

    Returns:
        Dict with keys: 'basic', 'pro', 'acoustic_single', 'acoustic_multi'
    """
    datasets = {
        "basic": [],
        "pro": [],
        "acoustic_single": [],
        "acoustic_multi": [],
    }

    # Basic datasets
    basic_dir = data_dir / "Basic" / "basic_class"
    if basic_dir.exists():
        for f in sorted(basic_dir.glob("*.json")):
            datasets["basic"].append(f.stem)

    # Pro datasets
    pro_dir = data_dir / "Pro" / "pro_class"
    if pro_dir.exists():
        for f in sorted(pro_dir.glob("*.json")):
            datasets["pro"].append(f.stem)

    # Acoustic datasets
    acoustic_dir = data_dir / "Acoustic" / "json"
    if acoustic_dir.exists():
        for f in sorted(acoustic_dir.glob("*.json")):
            name = f.stem
            if name.startswith("multi_round_"):
                datasets["acoustic_multi"].append(name)
            else:
                datasets["acoustic_single"].append(name)

    return datasets


# ============================================================================
# Checkpoint Support
# ============================================================================

def get_processed_ids(output_json_path: Path, is_multi_round: bool = False) -> set:
    """Get already processed sample IDs from output file."""
    processed_ids = set()
    if not output_json_path.exists():
        return processed_ids

    try:
        with open(output_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "id" not in item:
                        continue
                    if is_multi_round:
                        # Check all 4 rounds are complete
                        if all(f"model{i}_response_text" in item for i in range(1, 5)):
                            processed_ids.add(item["id"])
                    else:
                        if "model_response_text" in item:
                            processed_ids.add(item["id"])
    except (json.JSONDecodeError, IOError):
        pass

    return processed_ids


def save_result(output_json_path: Path, result: Dict):
    """Save or update a single result to JSON file."""
    existing = []
    if output_json_path.exists():
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    # Update or append
    found = False
    for i, item in enumerate(existing):
        if item.get("id") == result["id"]:
            existing[i] = result
            found = True
            break

    if not found:
        existing.append(result)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ============================================================================
# Processing Functions
# ============================================================================

def process_colloquial_dataset(
    model,
    samples: List[Dict],
    output_json_path: Path,
    output_audio_dir: Optional[Path] = None,
    with_audio_output: bool = False,
    max_new_tokens: int = 2048,
):
    """
    Process Colloquial (Basic/Pro) dataset with optional audio output.

    Args:
        model: VoiceAssistant model instance
        samples: List of samples to process
        output_json_path: Path to save results
        output_audio_dir: Directory to save audio outputs
        with_audio_output: Whether to generate audio output
        max_new_tokens: Maximum tokens to generate
    """
    processed_ids = get_processed_ids(output_json_path, is_multi_round=False)
    pending = [s for s in samples if s.get("id") not in processed_ids]

    logger.info(f"Total: {len(samples)}, Processed: {len(processed_ids)}, Pending: {len(pending)}")

    if with_audio_output and output_audio_dir:
        output_audio_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(pending, desc="Processing"):
        sample_id = sample.get("id")
        audio_path = sample.get("audio_abs_path")

        try:
            response_text, audio_bytes = model.generate_from_file(
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                with_audio_output=with_audio_output
            )

            result = sample.copy()
            result["model_response_text"] = response_text

            # Save audio if generated
            if with_audio_output and audio_bytes and output_audio_dir:
                audio_filename = f"{sample_id}.wav"
                audio_save_path = output_audio_dir / audio_filename
                with open(audio_save_path, 'wb') as f:
                    f.write(audio_bytes)
                # model_response_audio: filename only (for evaluator compatibility)
                result["model_response_audio"] = audio_filename
                # model_response_audio_path: full absolute path to output audio
                result["model_response_audio_path"] = str(audio_save_path)

            save_result(output_json_path, result)
            logger.debug(f"Processed {sample_id}: {response_text[:50]}...")

        except Exception as e:
            logger.error(f"Error processing {sample_id}: {e}")
            continue


def process_acoustic_single_round(
    model,
    samples: List[Dict],
    output_json_path: Path,
    output_audio_dir: Optional[Path] = None,
    with_audio_output: bool = True,
    max_new_tokens: int = 2048,
):
    """
    Process Acoustic single-round dataset with optional audio output.

    Args:
        model: VoiceAssistant model instance
        samples: List of samples to process
        output_json_path: Path to save results
        output_audio_dir: Directory to save audio outputs
        with_audio_output: Whether to generate audio output
        max_new_tokens: Maximum tokens to generate
    """
    processed_ids = get_processed_ids(output_json_path, is_multi_round=False)
    pending = [s for s in samples if s.get("id") not in processed_ids]

    logger.info(f"Total: {len(samples)}, Processed: {len(processed_ids)}, Pending: {len(pending)}")

    if with_audio_output and output_audio_dir:
        output_audio_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(pending, desc="Processing"):
        sample_id = sample.get("id")
        audio_path = sample.get("audio_abs_path")

        try:
            response_text, audio_bytes = model.generate_from_file(
                audio_path=audio_path,
                max_new_tokens=max_new_tokens,
                with_audio_output=with_audio_output
            )

            result = sample.copy()
            result["model_response_text"] = response_text

            # Save audio if generated
            if with_audio_output and audio_bytes and output_audio_dir:
                audio_filename = f"{sample_id}.wav"
                audio_save_path = output_audio_dir / audio_filename
                with open(audio_save_path, 'wb') as f:
                    f.write(audio_bytes)
                # model_response_audio: filename only (for evaluator compatibility)
                result["model_response_audio"] = audio_filename
                # model_response_audio_path: full absolute path to output audio
                result["model_response_audio_path"] = str(audio_save_path)

            save_result(output_json_path, result)
            logger.debug(f"Processed {sample_id}: {response_text[:50]}...")

        except Exception as e:
            logger.error(f"Error processing {sample_id}: {e}")
            continue


def process_acoustic_multi_round(
    model,
    samples: List[Dict],
    output_json_path: Path,
    output_audio_dir: Path,
    with_audio_output: bool = True,
    max_new_tokens: int = 2048,
):
    """
    Process Acoustic multi-round dataset (4 rounds per sample).

    Args:
        model: VoiceAssistant model instance
        samples: List of samples to process
        output_json_path: Path to save results
        output_audio_dir: Directory to save audio outputs
        with_audio_output: Whether to generate audio output
        max_new_tokens: Maximum tokens to generate
    """
    processed_ids = get_processed_ids(output_json_path, is_multi_round=True)
    pending = [s for s in samples if s.get("id") not in processed_ids]

    logger.info(f"Total: {len(samples)}, Processed: {len(processed_ids)}, Pending: {len(pending)}")

    output_audio_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(pending, desc="Processing multi-round"):
        sample_id = sample.get("id")
        audio_paths = sample.get("audio_paths", [])

        if len(audio_paths) != 4:
            logger.warning(f"Sample {sample_id} does not have 4 audio paths, skipping")
            continue

        try:
            # Create sample output directory for audio files
            sample_audio_dir = output_audio_dir / sample_id
            sample_audio_dir.mkdir(parents=True, exist_ok=True)

            # Multi-round inference
            round_results = model.generate_multi_round(
                audio_paths=audio_paths,
                with_audio_output=with_audio_output,
                max_new_tokens=max_new_tokens
            )

            result = sample.copy()

            # Save each round's results
            for i, (text, audio_bytes) in enumerate(round_results, start=1):
                result[f"model{i}_response_text"] = text

                if with_audio_output and audio_bytes:
                    audio_filename = f"{sample_id}/model{i}.wav"
                    audio_save_path = sample_audio_dir / f"model{i}.wav"
                    with open(audio_save_path, 'wb') as f:
                        f.write(audio_bytes)
                    # model{i}_response_audio: relative path (for evaluator compatibility)
                    result[f"model{i}_response_audio"] = audio_filename
                    # model{i}_response_audio_path: full absolute path to output audio
                    result[f"model{i}_response_audio_path"] = str(audio_save_path)

            save_result(output_json_path, result)
            logger.debug(f"Processed multi-round {sample_id}")

        except Exception as e:
            logger.error(f"Error processing multi-round {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WavBench Unified Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available datasets
    python main.py --list_datasets

    # Run inference on Basic code dataset
    python main.py --model step_audio2 --data basic_code

    # Run inference on Acoustic dataset with audio output
    python main.py --model step_audio2 --data acoustic_explicit_generation_emotion --audio_output

    # Run inference on multi-round dataset
    python main.py --model step_audio2 --data acoustic_multi_round_generation --audio_output
        """
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="step_audio2",
        choices=list(model_cls_mapping.keys()),
        help="Model to use for inference"
    )

    # Data selection
    parser.add_argument(
        "--data", type=str, default=None,
        help="Dataset name (e.g., basic_code, pro_math, acoustic_explicit_generation_emotion)"
    )

    # Paths
    parser.add_argument(
        "--data_dir", type=str, default="./wavbench",
        help="Base directory for WavBench data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for results"
    )

    # Model paths (for step_audio2)
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to model weights (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--token2wav_path", type=str, default=None,
        help="Path to token2wav weights (optional)"
    )

    # Generation options
    parser.add_argument(
        "--audio_output", action="store_true",
        help="Generate audio output (for acoustic datasets)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Maximum tokens to generate"
    )

    # Utility options
    parser.add_argument(
        "--list_datasets", action="store_true",
        help="List all available datasets and exit"
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # List datasets mode
    if args.list_datasets:
        print("\n" + "=" * 60)
        print("Available Datasets")
        print("=" * 60)

        datasets = discover_datasets(data_dir)

        print("\n[Basic (Colloquial)]")
        for name in datasets["basic"]:
            print(f"  - basic_{name}")

        print("\n[Pro (Colloquial)]")
        for name in datasets["pro"]:
            print(f"  - pro_{name}")

        print("\n[Acoustic Single-Round]")
        for name in datasets["acoustic_single"]:
            print(f"  - acoustic_{name}")

        print("\n[Acoustic Multi-Round]")
        for name in datasets["acoustic_multi"]:
            print(f"  - acoustic_{name}")

        print("\n" + "=" * 60)
        return

    # Validate data argument
    if not args.data:
        parser.error("--data is required. Use --list_datasets to see available datasets.")

    # Parse dataset name
    data_name = args.data
    if data_name.startswith("basic_"):
        dataset_type = "basic"
        task_name = data_name[6:]
    elif data_name.startswith("pro_"):
        dataset_type = "pro"
        task_name = data_name[4:]
    elif data_name.startswith("acoustic_"):
        acoustic_name = data_name[9:]
        if acoustic_name.startswith("multi_round_"):
            dataset_type = "acoustic_multi"
            task_name = acoustic_name
        else:
            dataset_type = "acoustic_single"
            task_name = acoustic_name
    else:
        parser.error(f"Unknown dataset format: {data_name}")

    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    model_cls = model_cls_mapping[args.model]

    model_kwargs = {}
    if args.model_path:
        model_kwargs["model_path"] = args.model_path
    if args.token2wav_path:
        model_kwargs["token2wav_path"] = args.token2wav_path

    model = model_cls(**model_kwargs)

    # Set appropriate system prompt based on dataset type
    if dataset_type in ["basic", "pro"]:
        model.set_system_prompt(
            "You are an expert in audio analysis. Please analyze the audio content and answer accurately."
        )
    else:
        model.set_system_prompt("You are a helpful assistant.")

    # Process based on dataset type
    if dataset_type == "basic":
        logger.info(f"Processing Basic dataset: {task_name}")
        samples = load_colloquial_dataset(data_dir, "basic", task_name)

        # Output: {output_dir}/basic/{task}/basic_{task}.json
        output_path = output_dir / "basic" / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_json = output_path / f"basic_{task_name}.json"

        process_colloquial_dataset(
            model, samples, output_json,
            output_audio_dir=output_path if args.audio_output else None,
            with_audio_output=args.audio_output,
            max_new_tokens=args.max_tokens
        )

    elif dataset_type == "pro":
        logger.info(f"Processing Pro dataset: {task_name}")
        samples = load_colloquial_dataset(data_dir, "pro", task_name)

        # Output: {output_dir}/pro/{task}/pro_{task}.json
        output_path = output_dir / "pro" / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_json = output_path / f"pro_{task_name}.json"

        process_colloquial_dataset(
            model, samples, output_json,
            output_audio_dir=output_path if args.audio_output else None,
            with_audio_output=args.audio_output,
            max_new_tokens=args.max_tokens
        )

    elif dataset_type == "acoustic_single":
        logger.info(f"Processing Acoustic single-round dataset: {task_name}")
        samples = load_acoustic_single_round(data_dir, task_name)

        output_path = output_dir / "acoustic" / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_json = output_path / f"{task_name}.json"

        process_acoustic_single_round(
            model, samples, output_json,
            output_audio_dir=output_path if args.audio_output else None,
            with_audio_output=args.audio_output,
            max_new_tokens=args.max_tokens
        )

    elif dataset_type == "acoustic_multi":
        logger.info(f"Processing Acoustic multi-round dataset: {task_name}")
        samples = load_acoustic_multi_round(data_dir, task_name)

        output_path = output_dir / "acoustic" / "multi" / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        output_json = output_path / f"{task_name}.json"

        process_acoustic_multi_round(
            model, samples, output_json,
            output_audio_dir=output_path,
            with_audio_output=args.audio_output,
            max_new_tokens=args.max_tokens
        )

    logger.info(f"Results saved to: {output_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()