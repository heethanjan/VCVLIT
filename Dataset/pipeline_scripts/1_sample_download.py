import argparse
import yt_dlp
import pysrt
import cv2
import os
import json
import pandas as pd
from pathlib import Path
import glob

# Usage:
#   python 1_sample_download.py --csv-dir <folder of csvs> --output-base <videos dir> --cookies cookies.txt
parser = argparse.ArgumentParser()
parser.add_argument("--csv-dir", required=True, help="folder of CSVs with id,title,channel,url")
parser.add_argument("--output-base", required=True, help="videos are written to <output-base>/<csv name>/<id>/")
parser.add_argument("--cookies", default=None, help="YouTube cookies.txt")
parser.add_argument("--ffmpeg", default=os.path.expanduser('~/ffmpeg-7.0.2-amd64-static/ffmpeg'))
parser.add_argument("--lang", default="en")
parser.add_argument("--mid-offset", type=float, default=0.0)
args = parser.parse_args()

CSV_DIRECTORY = args.csv_dir
OUTPUT_BASE = args.output_base
SUBTITLE_LANG = args.lang
MID_OFFSET_S = args.mid_offset
FFMPEG_PATH = args.ffmpeg
COOKIES = args.cookies

# =========================
# Utilities
# =========================
def ensure_dirs(path):
    os.makedirs(path, exist_ok=True)
    return path

def time_to_name(t_seconds: float) -> str:
    ms = int(round(t_seconds * 1000))
    return f"{ms:09d}ms"

def download_video_and_subtitles(video_url, output_dir, subtitle_lang):
    video_path = os.path.join(output_dir, "video.mp4")
    subtitle_path = os.path.join(output_dir, f"video.{subtitle_lang}.srt")

    if os.path.exists(video_path) and os.path.exists(subtitle_path):
        print("✔ Video + subtitles exist. Skipping download.")
        return video_path, subtitle_path

    print(f"⬇ Downloading {video_url} ...")
    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "video.%(ext)s"),
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": [subtitle_lang],
        "subtitlesformat": "srt",
        "skip_download": False,
        "format": "bv*[vcodec^=avc1][height=480]+ba/best[ext=mp4][vcodec^=avc1][height=480]/bv*[vcodec^=avc1][height=720]+ba/best[ext=mp4][vcodec^=avc1][height=720]/bv*[vcodec^=avc1][height=360]+ba/best[ext=mp4][vcodec^=avc1][height=360]/best",
        "merge_output_format": "mp4",
    }
    if COOKIES:
        ydl_opts["cookiefile"] = COOKIES
    if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
        ydl_opts["ffmpeg_location"] = FFMPEG_PATH

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        merged_ext = info.get("ext", "mp4")
        video_file = os.path.join(output_dir, f"video.{merged_ext}")
        if merged_ext.lower() != "mp4":
            target_path = os.path.join(output_dir, "video.mp4")
            if os.path.exists(video_file):
                os.replace(video_file, target_path)
            video_file = target_path

    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Expected video not found: {video_file}")

    candidate_srt = os.path.join(output_dir, f"video.{subtitle_lang}.srt")
    if os.path.exists(candidate_srt):
        subtitle_file = candidate_srt
    else:
        all_srts = sorted(Path(output_dir).glob("video.*.srt"))
        if not all_srts:
            raise FileNotFoundError("No subtitles found.")
        subtitle_file = str(all_srts[0])

    return video_file, subtitle_file

def parse_subtitles(subtitle_file):
    subs = pysrt.open(subtitle_file, encoding="utf-8")
    def to_seconds(t): return t.ordinal / 1000.0
    return [
        {
            "index": idx,
            "text": sub.text.replace("\n", " ").strip(),
            "start_time": to_seconds(sub.start),
            "end_time": to_seconds(sub.end),
        }
        for idx, sub in enumerate(subs)
    ]

class VideoReader:
    def __init__(self, video_file):
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open {video_file}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.frame_count / self.fps if self.frame_count > 0 else None

    def extract_frame_at(self, t_seconds, out_path):
        if self.duration is not None:
            eps = 0.5 / self.fps
            t_seconds = min(max(0.0, t_seconds), max(0.0, self.duration - eps))
        frame_number = max(0, int(round(t_seconds * self.fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(out_path, frame)
            return frame
        return None

    def release(self):
        if self.cap: 
            self.cap.release()

def match_subtitles_to_frames(video_file, subtitle_data, frame_output_dir):
    subtitle_frame_map = {}
    vr = VideoReader(video_file)
    try:
        for sub in subtitle_data:
            mid_time = ((sub["start_time"] + sub["end_time"]) / 2.0) + MID_OFFSET_S
            fname = f"frame_{sub['index']:04d}_{time_to_name(mid_time)}.jpg"
            frame_path = os.path.join(frame_output_dir, fname)
            vr.extract_frame_at(mid_time, frame_path)
            subtitle_frame_map[sub["index"]] = {
                "text": sub["text"],
                "start_time": sub["start_time"],
                "end_time": sub["end_time"],
                "sample_time": mid_time,
                "frame_path": frame_path,
            }
    finally:
        vr.release()
    return subtitle_frame_map

# =========================
# Main loop for all CSVs
# =========================
def main():
    # Get all CSV files from the directory
    csv_files = glob.glob(os.path.join(CSV_DIRECTORY, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {CSV_DIRECTORY}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    print()
    
    total_videos = 0
    successful_videos = 0
    skipped_videos = 0
    failed_videos = 0
    
    for csv_file in csv_files:
        csv_name = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n{'='*60}")
        print(f"Processing CSV: {csv_name}")
        print(f"{'='*60}")
        
        # Create output folder for this CSV
        csv_output_dir = ensure_dirs(os.path.join(OUTPUT_BASE, csv_name))
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'url' not in df.columns:
                print(f"⚠ Skipping {csv_name}: 'url' column not found")
                continue
            
            for idx, row in df.iterrows():
                total_videos += 1
                
                # Get video info
                vid_id = row.get("id", f"video_{idx}")
                title = row.get("title", "")
                channel = row.get("channel", "")
                url = row.get("url", "")
                
                if not url or pd.isna(url):
                    print(f"⚠ Row {idx}: Skipping empty URL")
                    failed_videos += 1
                    continue
                
                print(f"\n=== [{total_videos}] Processing {vid_id} | {title} ===")
                
                # Create folder structure for this video
                out_dir = ensure_dirs(os.path.join(csv_output_dir, str(vid_id)))
                frame_dir = ensure_dirs(os.path.join(out_dir, "frames"))
                map_path = os.path.join(out_dir, "subtitle_frame_map_middle.json")
                
                # Check if already fully processed
                if os.path.exists(map_path):
                    print(f"✔ Already processed (JSON exists). Skipping.")
                    skipped_videos += 1
                    successful_videos += 1
                    continue
                
                try:
                    # Download video and subtitles
                    video_file, subtitle_file = download_video_and_subtitles(url, out_dir, SUBTITLE_LANG)
                    
                    # Parse subtitles
                    subtitle_data = parse_subtitles(subtitle_file)
                    
                    # Extract frames
                    mapping = match_subtitles_to_frames(video_file, subtitle_data, frame_dir)
                    
                    # Keep only mappings where a non-empty frame file exists
                    non_empty_mapping = {
                        k: v for k, v in mapping.items()
                        if os.path.exists(v["frame_path"]) and os.path.getsize(v["frame_path"]) > 0
                    }
                    
                    # Save mapping to JSON
                    with open(map_path, "w", encoding="utf-8") as f:
                        json.dump(non_empty_mapping, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ Saved subtitle-frame map → {map_path}")
                    print(f"📊 Non-empty frames extracted: {len(non_empty_mapping)} / {len(subtitle_data)}")
                    
                    successful_videos += 1
                    
                except Exception as e:
                    print(f"❌ [ERROR] {vid_id}: {e}")
                    failed_videos += 1
        
        except Exception as e:
            print(f"❌ Error processing {csv_name}: {str(e)}")
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"Successfully processed: {successful_videos}")
    print(f"Already completed (skipped): {skipped_videos}")
    print(f"Failed: {failed_videos}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
