# Vc-VLIT: Vision-Centric Instruction Tuning and Data for Visual Reasoning in Small Vision-Language Models

Heethanjan Kanagalingam, Basim Azam, Sarah Monazam Erfani, Naveed Akhtar
The University of Melbourne

This repository contains the code and data pipeline for **Vc-VLIT** (Vision-centric Vision-Language Instruction Tuning) and the **SIVA** (Subtitle-Image Visual Alignment) dataset, using LLaVA-OneVision-Qwen2-0.5B as the base model.

## Overview

Small VLMs often fail at visual reasoning because of weak visual perception. Standard instruction tuning sends gradients mainly through the language model, leaving the vision encoder and projector under-optimized.

- **Vc-VLIT** trains only the vision encoder and projection layers and keeps the language model frozen. It combines:
  - a geometric (hyperspherical, geodesic-distance) bidirectional InfoNCE contrastive loss, and
  - a single-token frame-index prediction loss (`L_total = L_contrastive + L_LM`).
- **SIVA** is a fully automated dataset of 184,783 subtitle-image pairs and 739,132 same-domain hard distractor images (923,915 images total) across 18 subject areas. It is built from YouTube subtitles without proprietary models or human annotation.

### Main results (LLaVA-OneVision-0.5B)

| Benchmark | Accuracy / score (%) | Relative gain |
|---|---|---|
| MMMU (val) | 32.11 | +6.25% |
| MMMU-Pro (standard) | 16.59 | +6.23% |
| MathVision (test-mini) | 16.78 | +10.69% |
| MathVista (test-mini, solution) | 29.8 | +8.76% |

See the paper for the full results, ablations, and the Qwen2.5-VL-3B generalization experiments.

## Repository structure

```
Dataset/
  pipeline_scripts/   SIVA data-collection pipeline (steps 1-7)
  combined_csv/       Per-subject-area CSVs (video URLs / subtitle metadata)
llava/                Model, training and evaluation code (LLaVA-OneVision based)
scripts/              DeepSpeed configs (zero2.json, zero3.json)
train_scripts/        Vc-VLIT training launch scripts
requirements.txt
pyproject.toml
```

## Installation

```bash
git clone https://github.com/heethanjan/VCVLIT.git
cd VCVLIT
conda create -n vcvlit python=3.10 -y
conda activate vcvlit
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

## SIVA data pipeline

The scripts in `Dataset/pipeline_scripts/` run in order:

| Step | Script | Purpose |
|---|---|---|
| 1 | `1_sample_download.py` | Retrieve videos and subtitles for a subject area (YouTube Data API v3 + yt-dlp) |
| 2 | `2_sample_llm_filter.py` | LLM filter (Meta-Llama-3.1-8B-Instruct) keeps visually grounded subtitles |
| 3 | `3_sample_vlm_filter.py` | VLM verification (InternVL3-14B) of subtitle-frame alignment |
| 4 | `4_sample_distractor.py` | Same-domain distractor selection via cosine similarity (InternViT-300M, threshold 0.85, up to 5 resampling attempts) |
| 5 | `5_sample_consolidate.py` | Consolidate the filtered samples |
| 6-7 | `6_sample_gen_bal.py`, `7_sample_gen_dis.py` | Generate the balanced training data and the reasoning descriptions |

You need a YouTube Data API key and access to the LLM/VLM checkpoints above. Check the paths and arguments at the top of each script.

## Training

Vc-VLIT fine-tunes only the vision tower and projector for one epoch on SIVA. The language model stays frozen.

Settings used in the paper for LLaVA-OneVision-Qwen2-0.5B-OV:

- vision tower: `google/siglip-so400m-patch14-384`
- vision encoder LR `2e-7`, projector LR `1e-6`, contrastive projection MLP LR `1e-4`
- temperature `tau = 0.07`
- frozen text embedding model: `bge-large-en-v1.5`
- projection MLPs are first initialized with `L_contrastive` only

Launch scripts in `train_scripts/`:

| Script | Trains |
|---|---|
| `train_youtube_vr3.sh` | Vision tower + projector (full Vc-VLIT) |
| `train_youtube_vr3_projector_only.sh` | Projector only (ablation) |
| `train_youtube_vr3_vision_tower_only.sh` | Vision tower only (ablation) |

```bash
bash train_scripts/train_youtube_vr3.sh
```

**Note:** the scripts contain cluster-specific absolute paths (`HF_HOME`, `DATA_PATH`, `IMAGE_FOLDER`, `PRETRAINED_*_PROJECTOR`, `module load ...`). Edit these for your environment before running.

## Acknowledgements

This code builds on [LLaVA-NeXT / LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT). See `LICENSE` for license terms.

## Citation

If you use this work, please cite the paper (Vc-VLIT / SIVA, WACV submission).
