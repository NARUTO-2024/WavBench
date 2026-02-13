# WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models

[**📖 Paper**](https://arxiv.org/abs/2602.12135) | [**🏠 Website**](https://naruto-2024.github.io/wavbench.github.io/) | [**🤗 Dataset (HuggingFace)**](https://huggingface.co/datasets/WavBench/WavBench)

## Overview of WavBench

<div align="center">
  <img src="assets/colloquial_expression.png" width="80%"/>
  <br>
  <em>Figure 1: Examples of Colloquial Expression in WavBench, covering diverse cognitive domains across Basic and Pro subsets.</em>
</div>
<br>

<div align="center">
  <img src="assets/acoustic_interaction.png" width="80%"/>
  <br>
  <em>Figure 2: Examples of Acoustic Interaction in WavBench, demonstrating Explicit Understanding, Explicit Generation, and Implicit Dialogue.</em>
</div>

## News
* **`2026.02.11`** Released the **WavBench** paper, code, and dataset.
* **`2026.02.11`** Released the leaderboard evaluating 5 state-of-the-art E2E spoken dialogue models.

## Table of Contents
- [**Leaderboard**](#leaderboard)
- [**Setup**](#setup)
- [**Dataset**](#dataset)
- [**Evaluation**](#evaluation)
- [**Citation**](#citation)

## Leaderboard

Below is the overall evaluation of WavBench across five panels: **Colloquial Expression** (Pro & Basic) and **Acoustic Interaction** (Explicit Understanding, Explicit Generation, and Implicit).

| Metrics / Tasks | Qwen3-Omni | Kimi-Audio | Mimo-Audio | Step-Audio-2 | GPT-4o Audio |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Panel A: Colloquial (Pro)** | | | | | |
| Code | 39.75 | 30.29 | 28.96 | 31.20 | **53.60** |
| Creativity | 48.39 | 31.78 | 42.86 | 35.00 | **63.00** |
| Instruction | 43.01 | 29.86 | 36.44 | 29.40 | **57.80** |
| Logic | 33.21 | 26.03 | 27.57 | 26.20 | **42.60** |
| Math | 38.55 | 27.30 | 25.68 | 22.40 | **50.20** |
| QA | 50.93 | 42.54 | 41.28 | 40.80 | **72.80** |
| Safety | 60.00 | 56.19 | 56.19 | 52.40 | **67.60** |
| **Avg (Pro)** | 39.53 | 30.79 | 32.02 | 30.40 | **58.23** |
| | | | | | |
| **Panel B: Colloquial (Basic)** | | | | | |
| Code | 53.10 | 40.69 | 42.07 | 37.20 | **58.00** |
| Creativity | 57.44 | 41.57 | 45.29 | 47.20 | **71.20** |
| Instruction | 57.29 | 44.41 | 33.56 | 36.60 | **66.80** |
| Logic | 52.35 | 50.74 | 49.91 | 48.80 | **67.00** |
| Math | 51.05 | 41.27 | 38.73 | 30.20 | **62.40** |
| QA | 57.54 | 49.07 | 49.12 | 48.60 | **75.60** |
| Safety | 59.67 | 58.83 | 62.83 | 60.20 | **81.00** |
| **Avg (Basic)** | 55.80 | 49.23 | 49.57 | 48.50 | **68.80** |
| | | | | | |
| **Panel C: Explicit Understanding** | | | | | |
| Accent | **37.50** | 11.00 | 27.00 | 20.67 | 15.67 |
| Age | 64.33 | 53.67 | 53.00 | **67.67** | 20.33 |
| Emotion | **92.86** | 77.33 | 77.33 | 75.43 | 85.90 |
| Gender | 21.00 | 44.50 | 20.00 | **68.00** | 61.50 |
| Language | 83.50 | 91.00 | 53.50 | 96.50 | **97.00** |
| Pitch | 32.44 | 23.11 | 24.00 | **34.22** | 23.56 |
| Speed | 46.67 | **54.67** | 48.89 | 44.00 | 48.00 |
| Volume | 33.78 | 38.22 | 31.11 | **50.67** | 41.78 |
| Audio Event | 61.73 | **67.90** | 19.75 | 39.51 | 59.26 |
| Music | 22.22 | 66.67 | 55.56 | **77.78** | 33.33 |
| **Avg (Understand)** | 49.60 | 52.80 | 41.02 | **57.36** | 48.70 |
| | | | | | |
| **Panel D: Explicit Generation** | | | | | |
| Accent | 37.50 | 3.52 | 23.44 | 22.07 | **74.22** |
| Age | 64.65 | 46.88 | 51.95 | 31.64 | **78.12** |
| Emotion | 90.04 | 50.29 | 57.13 | 66.50 | **95.51** |
| Gender | 72.27 | 45.31 | 67.58 | 59.77 | **98.83** |
| Language | 89.84 | 74.80 | 51.56 | **91.41** | 87.89 |
| Pitch | 76.56 | 47.27 | 80.27 | 55.66 | **85.74** |
| Speed | 43.75 | 47.27 | 51.56 | **69.14** | 66.60 |
| Volume | 56.25 | 64.06 | 59.96 | 57.03 | **82.42** |
| Audio | 27.03 | 10.81 | 9.46 | 32.43 | **45.95** |
| Music | 62.50 | 20.83 | 16.67 | **70.83** | 77.08 |
| **Avg (Generation)** | 62.03 | 41.10 | 46.93 | 55.65 | **79.23** |
| | | | | | |
| **Panel E: Implicit** | | | | | |
| Single-Turn (Text) | 1.85 | 1.84 | 2.23 | 1.12 | **2.43** |
| Single-Turn (Audio) | 3.17 | 3.21 | 2.47 | **3.50** | 2.96 |
| Multi-Turn (Text) | **4.88** | 4.57 | 4.61 | 4.38 | 4.48 |
| Multi-Turn (Audio) | **1.25** | 1.08 | 1.04 | 1.21 | 1.23 |
| **Avg (Implicit)** | **2.78** | 2.67 | 2.59 | 2.55 | **2.78** |

## Setup

```shell
conda create -n wavbench python=3.10
conda activate wavbench
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

git clone https://github.com/NARUTO-2024/WavBench.git
cd WavBench
```

## Dataset

The data used in this project is available at [WavBench Dataset](https://huggingface.co/datasets/WavBench/WavBench) hosted on Hugging Face.

You can load the dataset directly using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the dataset directly from Hugging Face
ds = load_dataset("WavBench/WavBench")
```

Alternatively, you can download the dataset to your local directory and use it directly.

### 1. Colloquial Expression
This category is divided into **Basic** and **Pro** subsets. Each subset contains tasks across 7 diverse cognitive domains:

| Domain | Description |
| :--- | :--- |
| **Code** | Evaluate the model's ability to explain code logic conversationally. |
| **Creative** | Evaluate creative writing without rigid formatting constraints. |
| **Instruction** | Evaluate adherence to spoken instructions. |
| **Logic** | Evaluate logical reasoning in a spoken context. |
| **Math** | Evaluate the verbalization of mathematical reasoning. |
| **QA** | Evaluate general knowledge answering capabilities. |
| **Safety** | Evaluate safety mechanisms in spoken interaction. |

### 2. Acoustic Interaction
This category evaluates the model's paralinguistic capabilities across three dimensions: **Explicit Understanding**, **Explicit Generation**, and **Implicit**.

| Category | Sub-tasks / Attributes |
| :--- | :--- |
| **Explicit Understanding** | **10 Attributes:** Accent, Age, Emotion, Gender, Language, Pitch, Speed, Volume, Audio, Music. |
| **Explicit Generation** | **10 Attributes:** Accent, Age, Emotion, Gender, Language, Pitch, Speed, Volume, Audio, Music. |
| **Implicit** | Single-turn Audio, Single-turn Text, Multi-turn Audio, Multi-turn Text. |

## Evaluation


### Step 1: Run Inference
`main.py` is the unified entry point for all dataset types.

```bash
# Colloquial Inference (Basic) - With audio output
python main.py --model step_audio2 --data basic_code --audio_output

# Colloquial Inference (Pro) - With audio output
python main.py --model step_audio2 --data pro_math --audio_output

# Acoustic Single-turn Inference (With audio output)
python main.py --model step_audio2 --data acoustic_explicit_generation_emotion --audio_output

# Acoustic Multi-round Inference (With audio output)
python main.py --model step_audio2 --data acoustic_multi_round_generation --audio_output

# [Optional] Run with custom data directory
python main.py --model step_audio2 --data basic_code --data_dir /path/to/your/wavbench
```

**Supported Arguments:**
* `--model`: Model name (e.g., `step_audio2`).
* `--data`: Dataset name (e.g., `basic_code`, `pro_math`, `acoustic_explicit_generation_emotion`).
* `--data_dir`: Optional. Base directory for WavBench data (Default: ./wavbench). Use this argument if you have downloaded the dataset to a specific location other than the default.
* `--audio_output`: **Important Flag**. If set, the model generates audio files in addition to text.
    * **Required** for all **Acoustic** tasks (as evaluation relies on audio).
    * **Optional** for **Colloquial** tasks (useful if you want to check the TTS quality manually).

### Step 2: Automatic Evaluation
`evaluate.py` uses LLMs (Gemini) to judge the responses based on the specific criteria of each subset.

```bash
# Option 1: Set API key via environment variable
export GOOGLE_API_KEY="your-api-key"

# Evaluate ALL Colloquial datasets
python evaluate.py --eval_type colloquial --dataset all

# Evaluate a SPECIFIC Colloquial dataset
python evaluate.py --eval_type colloquial --dataset basic_code

# Evaluate ALL Acoustic datasets
python evaluate.py --eval_type acoustic --dataset all

# Evaluate a SPECIFIC Acoustic dataset
python evaluate.py --eval_type acoustic --dataset explicit_generation_emotion
```

**Supported Arguments:**
* `--eval_type`: Choose between `colloquial` or `acoustic`.
* `--dataset`: Specific dataset name (e.g., `basic_code`) or use `all` to run the entire suite.

<details>
<summary><strong>👇 Available Dataset Options (for <code>--data</code> / <code>--dataset</code>)</strong></summary>

| Category | Available Values |
| :--- | :--- |
| **Basic** | `basic_code`, `basic_creative`, `basic_instruction`, `basic_logic`, `basic_math`, `basic_qa`, `basic_satety` |
| **Pro** | `pro_code`, `pro_creative`, `pro_instruction`, `pro_logic`, `pro_math`, `pro_qa`, `pro_satety` |
| **Explicit Generation** | `acoustic_explicit_generation_accent`, `acoustic_explicit_generation_age`, `acoustic_explicit_generation_audio`, `acoustic_explicit_generation_emotion`, `acoustic_explicit_generation_gender`, `acoustic_explicit_generation_lang`, `acoustic_explicit_generation_music`, `acoustic_explicit_generation_pitch`, `acoustic_explicit_generation_speed`, `acoustic_explicit_generation_volume` |
| **Explicit Understanding** | `acoustic_explicit_understanding_accent`, `acoustic_explicit_understanding_age`, `acoustic_explicit_understanding_audio`, `acoustic_explicit_understanding_emotion`, `acoustic_explicit_understanding_gender`, `acoustic_explicit_understanding_lang`, `acoustic_explicit_understanding_music`, `acoustic_explicit_understanding_pitch`, `acoustic_explicit_understanding_speed`, `acoustic_explicit_understanding_volume` |
| **Implicit** | `acoustic_implicit_age_generation`, `acoustic_implicit_emotion_generation`, `acoustic_implicit_pitch_generation`, `acoustic_implicit_speed_generation`, `acoustic_implicit_understanding` |
| **Multi-round** | `acoustic_multi_round_generation`, `acoustic_multi_round_understanding` |

</details>

### Step 3: Get Statistics
`statistics.py` aggregates the evaluation results into a final report.

```bash
# Basic usage: Output to TXT file
python statistics.py --eval_dir ./eval_results --output ./statistics.txt

# Advanced usage: Output to TXT and CSV format simultaneously
python statistics.py --eval_dir ./eval_results --output ./statistics.txt --csv
```

## Citation
If you use WavBench in your research, please cite the following paper:

```bibtex
@misc{li2026wavbenchbenchmarkingreasoningcolloquialism,
      title={WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models}, 
      author={Yangzhuo Li and Shengpeng Ji and Yifu Chen and Tianle Liang and Haorong Ying and Yule Wang and Junbo Li and Jun Fang and Zhou Zhao},
      year={2026},
      eprint={2602.12135},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.12135}, 
}
```
