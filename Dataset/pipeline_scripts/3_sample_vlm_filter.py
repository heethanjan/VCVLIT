import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
import json
import os
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name, path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map

# Usage: python 3_sample_vlm_filter.py --base-dir <videos/DATASET>
# Each video dir needs selected_frames.json (from 2_sample_llm_filter.py);
# writes matched_subtitles.json and not_matched_subtitles.json.
parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True, help="folder containing one sub-folder per video")
parser.add_argument("--model", default="OpenGVLab/InternVL3-14B")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

print(f"Loading {args.model} model...")
path = args.model
world_size = torch.cuda.device_count()
print(f"Detected {world_size} GPU(s)")

# Use split_model for multi-GPU, device_map="auto" for single GPU
if world_size > 1:
    print("Using multi-GPU device mapping...")
    device_map = split_model('InternVL3-14B', path)
else:
    print("Using single GPU with device_map='auto'...")
    device_map = "auto"

model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
print("Model loaded successfully!")

def check_subtitle_frame_match(entry, model, tokenizer, generation_config, input_size=448, max_num=12, device='cuda'):
    """Check if a frame matches its subtitle using the VLM model."""
    frame_path = entry['frame_path']
    subtitle_text = entry['text']
    prev_text = entry.get('context', {}).get('prev_text', '')
    next_text = entry.get('context', {}).get('next_text', '')

    pixel_values = load_image(frame_path, input_size=input_size, max_num=max_num).to(torch.bfloat16).to(device)

    prompt = (
        f"<image>\n"
        f"Does the visual content in the image match the subtitle description?\n"
        f"Subtitle: \"{subtitle_text}\"\n"
        f"Previous context: \"{prev_text}\"\n"
        f"Next context: \"{next_text}\"\n"
        f"Answer with Yes or No and briefly explain."
    )

    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    response_text = response.strip().lower()
    print(response_text)

    return 'yes' in response_text, response_text


def process_video_folder(video_dir, model, tokenizer, generation_config, device='cuda'):
    subtitle_json_path = Path(video_dir) / "selected_frames.json"
    matched_path = subtitle_json_path.parent / "matched_subtitles.json"
    not_matched_path = subtitle_json_path.parent / "not_matched_subtitles.json"

    # Skip if both output files already exist
    if matched_path.exists() and not_matched_path.exists():
        print(f"[SKIP] Already processed → {video_dir}")
        return "skipped"

    if not subtitle_json_path.exists():
        print(f"[SKIP] No selected_frames.json in {video_dir}")
        return "skipped"

    with open(subtitle_json_path, "r") as f:
        subtitle_data = json.load(f)

    matched, not_matched = {}, {}

    for key, entry in subtitle_data.items():
        print(f"Checking {video_dir} → subtitle {key}...")
        try:
            match, explanation = check_subtitle_frame_match(entry, model, tokenizer, generation_config, device=device)
            if match:
                matched[key] = {**entry, "model_explanation": explanation}
            else:
                not_matched[key] = {**entry, "model_explanation": explanation}
        except Exception as e:
            print(f"Error processing {key} in {video_dir}: {e}")
            not_matched[key] = {**entry, "error": str(e)}

    # Save results
    with open(matched_path, "w") as f:
        json.dump(matched, f, indent=2)

    with open(not_matched_path, "w") as f:
        json.dump(not_matched, f, indent=2)

    print(f"✅ Saved matched → {matched_path}")
    print(f"✅ Saved not matched → {not_matched_path}")
    return "processed"


def main():
    BASE_DIR = args.base_dir
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    processed_count, skipped_count = 0, 0

    for vid in sorted(os.listdir(BASE_DIR)):
        video_dir = os.path.join(BASE_DIR, vid)
        if os.path.isdir(video_dir):
            result = process_video_folder(video_dir, model, tokenizer, generation_config, device=device)
            if result == "processed":
                processed_count += 1
            else:
                skipped_count += 1

    print(f"\nSummary: {processed_count} videos processed, {skipped_count} skipped.")


if __name__ == "__main__":
    main()
