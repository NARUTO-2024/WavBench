# WavBench

This repo contains the code and data of:
[**WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models**](https://wavbench.github.io/)

<div align="center">

[[**Paper**]](https://wavbench.github.io/) &nbsp; [[**Website**]](https://wavbench.github.io/) &nbsp; [[**Dataset**]](https://huggingface.co/datasets/WavBench/WavBench)

</div>

## Overview

<div align="center">
  <img src="colloquial_expression.jpg" width="100%"/>
  <br>
  <em>Figure 1: Examples of Colloquial Expression in WavBench, covering diverse cognitive domains across Basic and Pro subsets.</em>
</div>
<br>

<div align="center">
  <img src="acoustic_interaction.jpg" width="100%"/>
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
| **Panel E: Implicit Interaction** | | | | | |
| Single-Turn (Text) | 1.85 | 1.84 | 2.23 | 1.12 | **2.43** |
| Single-Turn (Audio) | 3.17 | 3.21 | 2.47 | **3.50** | 2.96 |
| Multi-Turn (Text) | **4.88** | 4.57 | 4.61 | 4.38 | 4.48 |
| Multi-Turn (Audio) | **1.25** | 1.08 | 1.04 | 1.21 | 1.23 |
| **Avg (Implicit)** | **2.78** | 2.67 | 2.59 | 2.55 | **2.78** |

## Setup

```shell
conda create -n wavbench python=3.8+
conda activate wavbench
pip install -r requirements.txt
