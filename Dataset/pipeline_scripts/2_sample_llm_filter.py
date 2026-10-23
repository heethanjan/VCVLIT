import argparse
import json
import os
import transformers
import torch
from tqdm import tqdm
from transformers import logging
logging.set_verbosity_error()

# Usage:
#   python 2_sample_llm_filter.py --base-dir <videos/DATASET> --prompt-file llm_prompts/math.txt
# Each video dir needs subtitle_frame_map_middle.json; writes selected_frames.json.
parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True, help="folder containing one sub-folder per video")
parser.add_argument("--prompt-file", required=True, help="text file with the system prompt (see llm_prompts/)")
parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--output-name", default="selected_frames.json")
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

BASE_DIR = args.base_dir
OUTPUT_NAME = args.output_name
model_id = args.model_id
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda",
)

SYSTEM_PROMPT = open(args.prompt_file, encoding="utf-8").read()



def make_user_message(prev_text, cur_text, next_text):
    return (
        f"Evaluate the following subtitle and decide if it is visually grounded:\n\n"
        f"CURRENT subtitle: \"{cur_text}\"\n"
        "Return ONLY the JSON object following the system prompt."
    )

# =========================
# Processing one video folder
# =========================
def process_video_folder(video_dir):
    input_path = os.path.join(video_dir, "subtitle_frame_map_middle.json")
    output_path = os.path.join(video_dir, OUTPUT_NAME)

    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"[SKIP] Selected frames already exist in {video_dir}")
        return "skipped"

    if not os.path.exists(input_path):
        print(f"[SKIP] No subtitle_frame_map_middle.json in {video_dir}")
        return "skipped"

    with open(input_path, "r", encoding="utf-8") as f:
        subtitle_data = json.load(f)

    ordered = sorted(((int(k), v) for k, v in subtitle_data.items()), key=lambda x: x[0])
    selected_data = {}

    for i, (idx_int, item) in tqdm(enumerate(ordered), total=len(ordered), desc=f"Analyzing {os.path.basename(video_dir)}"):
        try:
            prev_text = ordered[i-1][1]["text"] if i > 0 else ""
            cur_text  = item["text"]
            next_text = ordered[i+1][1]["text"] if i < len(ordered) - 1 else ""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(prev_text, cur_text, next_text)}
            ]

            outputs = pipeline(messages, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.00001)

            assistant_msg = outputs[0]["generated_text"][-1]["content"]

            # Parse JSON
            analysis = None
            try:
                analysis = json.loads(assistant_msg)
            except json.JSONDecodeError:
                start = assistant_msg.find("{")
                end = assistant_msg.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        analysis = json.loads(assistant_msg[start:end+1])
                    except Exception:
                        analysis = None

            if analysis is None or "decision" not in analysis:
                print(f"[WARN] JSON parse failed at index {idx_int}. Raw:\n{assistant_msg[:300]}...")
                continue

            if str(analysis.get("decision", "")).strip().lower() == "yes":
                selected_data[str(idx_int)] = {
                    **item,
                    "analysis": analysis,
                    "context": {
                        "prev_text": prev_text,
                        "next_text": next_text
                    }
                }

        except Exception as e:
            print(f"[ERROR] at index {idx_int}: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(selected_data)} selected frames → {output_path}")
    return "processed"

# =========================
# Main: loop over all videos
# =========================
def main():
    processed_count = 0
    skipped_count = 0

    for vid in sorted(os.listdir(BASE_DIR)):
        video_dir = os.path.join(BASE_DIR, vid)
        if os.path.isdir(video_dir):
            result = process_video_folder(video_dir)
            if result == "processed":
                processed_count += 1
            else:
                skipped_count += 1

    print(f"\nSummary: {processed_count} videos processed, {skipped_count} videos skipped.")

if __name__ == "__main__":
    main()
