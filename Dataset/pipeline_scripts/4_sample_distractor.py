import argparse
import os
import json
import cv2
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

# Usage: python 4_sample_distractor.py --base-dir <videos/DATASET>
# Each video dir needs matched_subtitles.json (from 3_sample_vlm_filter.py).
parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True, help="folder containing one sub-folder per video")
parser.add_argument("--distractor-count", type=int, default=1, help="distractor frames per subtitle")
parser.add_argument("--similarity-threshold", type=float, default=0.86, help="InternViT feature similarity limit")
parser.add_argument("--max-resample-attempts", type=int, default=10, help="max tries for a dissimilar distractor")
args = parser.parse_args()

# === Parameters ===
DISTRACTOR_COUNT = args.distractor_count
SIMILARITY_THRESHOLD = args.similarity_threshold
MAX_RESAMPLE_ATTEMPTS = args.max_resample_attempts

# Global model (loaded once)
internvit_model = None
internvit_processor = None

# === Helpers ===
def time_to_name(t_seconds: float) -> str:
    ms = int(round(t_seconds * 1000))
    return f"{ms:09d}ms"

def load_selected_frames(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Can't open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps
    cap.release()
    return fps, frames, duration

def extract_frame(video_path, t_seconds, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = min(max(0, int(round(t_seconds * fps))), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(out_path, frame)
    cap.release()
    return ret

def load_internvit_model():
    global internvit_model, internvit_processor
    if internvit_model is None:
        try:
            model_name = "OpenGVLab/InternViT-300M-448px-V2_5"
            internvit_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            internvit_model = internvit_model.to(device).eval()
            internvit_processor = CLIPImageProcessor.from_pretrained(model_name)
            print("✓ InternViT-300M-448px model loaded for feature extraction.")
        except Exception as e:
            print(f"✗ Failed to load InternViT: {e}. Falling back to pixel-based similarity.")
            internvit_model = None

def get_image_features(img_path, model, processor):
    if model is None:
        raise ValueError("Model not loaded.")
    
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    
    with torch.no_grad():
        outputs = model(pixel_values)
    
    features = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy()
    return features.flatten()

def pixel_cosine_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"Warning: Could not load images {img1_path} or {img2_path}")
        return 0.0
    
    img1 = cv2.resize(img1, (224, 224)).flatten().astype(np.float32)
    img2 = cv2.resize(img2, (224, 224)).flatten().astype(np.float32)
    
    dot = np.dot(img1, img2)
    norm1 = np.linalg.norm(img1)
    norm2 = np.linalg.norm(img2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def cosine_similarity(img1_path, img2_path):
    if internvit_model is None:
        print("Using pixel-based similarity (model not loaded).")
        return pixel_cosine_similarity(img1_path, img2_path)
    
    try:
        feat1 = get_image_features(img1_path, internvit_model, internvit_processor)
        feat2 = get_image_features(img2_path, internvit_model, internvit_processor)
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    except Exception as e:
        print(f"Feature extraction error: {e}. Using pixel-based fallback similarity.")
        return pixel_cosine_similarity(img1_path, img2_path)

def sample_distractor_videos_and_times(valid_dirs, current_dir, count=DISTRACTOR_COUNT):
    """Sample one timestamp from each of 4 different videos."""
    # Exclude the current video's directory
    available_dirs = [d for d in valid_dirs if d != current_dir]
    if len(available_dirs) < count:
        print(f"Warning: Only {len(available_dirs)} other videos available, need {count}. Using available videos.")
        selected_dirs = available_dirs
    else:
        selected_dirs = np.random.choice(available_dirs, size=count, replace=False)
    
    distractor_info = []
    for d_dir in selected_dirs:
        video_path = Path(d_dir) / "video.mp4"
        _, _, duration = get_video_info(str(video_path))
        t = np.random.uniform(0, duration)  # Sample a random timestamp
        distractor_info.append({
            "video_path": str(video_path),
            "sample_time": t
        })
    
    # If we have fewer than 4 videos, pad with random samples from available videos
    while len(distractor_info) < count:
        d_dir = np.random.choice(available_dirs)
        video_path = Path(d_dir) / "video.mp4"
        _, _, duration = get_video_info(str(video_path))
        t = np.random.uniform(0, duration)
        distractor_info.append({
            "video_path": str(video_path),
            "sample_time": t
        })
    
    return distractor_info[:count]  # Ensure exactly 4 distractors

# === Process a single video folder ===
def process_video_folder(video_dir, valid_dirs):
    matched_json = Path(video_dir) / "matched_subtitles.json"
    output_json = Path(video_dir) / "selected_with_distractors.json"
    video_path = Path(video_dir) / "video.mp4"
    frame_dir = Path(video_dir) / "frames"
    frame_dir.mkdir(exist_ok=True)

    if not matched_json.exists():
        print(f"[SKIP] No matched_subtitles.json → {video_dir}")
        return

    if output_json.exists():
        print(f"[SKIP] selected_with_distractors.json already exists → {video_dir}")
        return

    print(f"Processing: {video_dir}")
    selected_frames = load_selected_frames(matched_json)
    fps, total_frames, duration = get_video_info(video_path)

    updated = {}
    for sid, sdata in selected_frames.items():
        s_time = sdata["sample_time"]

        # Main frame (always from current video)
        main_name = f"frame_{sid}_{time_to_name(s_time)}.jpg"
        main_path = frame_dir / main_name
        if not main_path.exists():
            extract_frame(video_path, s_time, str(main_path))
        sdata["frame_path"] = str(main_path)

        # Distractors (from 4 different videos)
        distractor_info = sample_distractor_videos_and_times(valid_dirs, video_dir, DISTRACTOR_COUNT)
        distractors = []
        distractor_paths = []
        for j, info in enumerate(distractor_info):
            d_video_path = info["video_path"]
            current_t = info["sample_time"]
            attempts = 0
            dpath = None
            while attempts < MAX_RESAMPLE_ATTEMPTS:
                dname = f"distractor_{sid}_{j}_{time_to_name(current_t)}.jpg"
                dpath = frame_dir / dname
                if extract_frame(d_video_path, current_t, str(dpath)):
                    # Check similarity with main frame
                    sim_main = cosine_similarity(main_path, dpath)
                    if sim_main >= SIMILARITY_THRESHOLD:
                        print(f"✗ Distractor too similar to main (sim={sim_main:.2f}) for {sid}_{j}, resampling...")
                        current_t = np.random.uniform(0, get_video_info(d_video_path)[2])
                        attempts += 1
                        continue

                    # Check similarity with previously selected distractors
                    too_similar = False
                    for prev_dpath in distractor_paths:
                        sim_distractor = cosine_similarity(dpath, prev_dpath)
                        if sim_distractor >= SIMILARITY_THRESHOLD:
                            print(f"✗ Distractor too similar to another distractor (sim={sim_distractor:.2f}) for {sid}_{j}, resampling...")
                            too_similar = True
                            break
                    if too_similar:
                        current_t = np.random.uniform(0, get_video_info(d_video_path)[2])
                        attempts += 1
                        continue

                    # Frame is sufficiently different
                    distractors.append({
                        "sample_time": current_t,
                        "frame_path": str(dpath),
                        "source_video": d_video_path
                    })
                    distractor_paths.append(dpath)
                    break
                else:
                    # Extraction failed, resample
                    current_t = np.random.uniform(0, get_video_info(d_video_path)[2])
                    attempts += 1

            if attempts >= MAX_RESAMPLE_ATTEMPTS:
                print(f"⚠️ Could not find dissimilar distractor for {sid}_{j} after {attempts} attempts.")
                if dpath and os.path.exists(dpath):
                    distractors.append({
                        "sample_time": current_t,
                        "frame_path": str(dpath),
                        "source_video": d_video_path
                    })
                    distractor_paths.append(dpath)

        sdata["distractor_frames"] = distractors
        updated[sid] = sdata

        print(f"✓ Subtitle {sid}: {len(distractors)} distractors added")

    # Save updated JSON
    save_json(updated, output_json)
    print(f"💾 Saved with distractors → {output_json}\n")

# === Main: traverse all subfolders ===
def main():
    load_internvit_model()

    BASE_DIR = args.base_dir
    valid_dirs = []

    for sub in sorted(os.listdir(BASE_DIR)):
        video_dir = os.path.join(BASE_DIR, sub)
        if os.path.isdir(video_dir):
            matched_json = Path(video_dir) / "matched_subtitles.json"
            if matched_json.exists():
                valid_dirs.append(video_dir)

    processed_count = len(valid_dirs)
    skipped_count = len(os.listdir(BASE_DIR)) - processed_count

    if not valid_dirs:
        print("No valid folders found.")
        return

    for video_dir in valid_dirs:
        process_video_folder(video_dir, valid_dirs)

    print(f"\nSummary: {processed_count} folders processed, {skipped_count} skipped.")

if __name__ == "__main__":
    main()
