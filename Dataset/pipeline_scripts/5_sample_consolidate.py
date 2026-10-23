import os
import json
import shutil
from pathlib import Path

def get_new_image_name(original_path, parent_folder, video_folder):
    """Generate new image name with parent and video folder prefixes."""
    original_name = os.path.basename(original_path)
    return f"{parent_folder}_{video_folder}_{original_name}"

def update_json_paths(data, parent_folder, video_folder):
    """Update frame_path and distractor_frames paths in JSON data to be relative to FData folder.
    
    Keys are prefixed with parent_folder_video_folder_ to avoid overwriting (e.g., 'Accounting__1s3Eq7dOXg_0').
    Paths in the output JSON are relative to FData, e.g., 'Images/Accounting__1s3Eq7dOXg_frame_0_000003219ms.jpg'.
    """
    updated_data = {}
    for key, entry in data.items():
        updated_entry = entry.copy()
        # Update main frame_path to relative path
        if "frame_path" in entry:
            original_name = os.path.basename(entry["frame_path"])
            new_name = f"{parent_folder}_{video_folder}_{original_name}"
            updated_entry["frame_path"] = os.path.join("Images", new_name).replace(os.sep, "/")
        # Update distractor_frames paths to relative paths
        if "distractor_frames" in entry:
            updated_distractors = []
            for distractor in entry["distractor_frames"]:
                distractor_copy = distractor.copy()
                if "frame_path" in distractor:
                    original_name = os.path.basename(distractor["frame_path"])
                    new_name = f"{parent_folder}_{video_folder}_{original_name}"
                    distractor_copy["frame_path"] = os.path.join("Images", new_name).replace(os.sep, "/")
                updated_distractors.append(distractor_copy)
            updated_entry["distractor_frames"] = updated_distractors
        # Prefix key to ensure uniqueness
        updated_data[f"{parent_folder}_{video_folder}_{key}"] = updated_entry
    return updated_data

def get_referenced_frames(json_data):
    """Extract all referenced frame paths from frame_path and distractor_frames."""
    frame_paths = set()
    for entry in json_data.values():
        if "frame_path" in entry:
            frame_paths.add(entry["frame_path"])
        if "distractor_frames" in entry:
            for distractor in entry["distractor_frames"]:
                if "frame_path" in distractor:
                    frame_paths.add(distractor["frame_path"])
    return frame_paths

def consolidate_referenced_frames_and_json(base_dirs, target_images_dir, target_json_path):
    """Copy only referenced frames from selected_with_distractors.json and consolidate JSON files."""
    # Convert to Path objects for consistent handling
    target_images_path = Path(target_images_dir)
    target_json_path = Path(target_json_path)
    
    # Create target images directory if it doesn't exist
    target_images_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize consolidated JSON data
    consolidated_json = {}
    processed_dirs = 0
    copied_files = 0
    
    for base_dir in base_dirs:
        # Extract parent folder name
        parent_folder = os.path.basename(base_dir.rstrip(os.sep))
        if not os.path.exists(base_dir):
            print(f"[SKIP] Directory does not exist: {base_dir}")
            continue
        
        # Traverse all subfolders
        for vid in sorted(os.listdir(base_dir)):
            video_dir = os.path.join(base_dir, vid)
            if not os.path.isdir(video_dir):
                continue
                
            json_path = Path(video_dir) / "selected_with_distractors.json"
            
            # Check if JSON exists
            if not json_path.exists():
                print(f"[SKIP] No selected_with_distractors.json in {video_dir}")
                continue
                
            # Load JSON
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to read {json_path}: {e}")
                continue
                
            # Get referenced frame paths
            frame_paths = get_referenced_frames(json_data)
            
            # Copy only referenced frame files
            for frame_path in frame_paths:
                frame_file = Path(frame_path)
                if frame_file.exists():
                    new_name = get_new_image_name(frame_file, parent_folder, vid)
                    target_path = target_images_path / new_name
                    try:
                        shutil.copy2(frame_file, target_path)
                        copied_files += 1
                        # print(f"Copied: {frame_file} -> {target_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to copy {frame_file}: {e}")
                else:
                    print(f"[WARNING] Frame file not found: {frame_file}")
            
            # Update JSON paths and add to consolidated data
            updated_json = update_json_paths(json_data, parent_folder, vid)
            consolidated_json.update(updated_json)
            
            processed_dirs += 1
            print(f"✓ Processed {video_dir}")
    
    # Write consolidated JSON with relative paths
    try:
        with open(target_json_path, "w", encoding="utf-8") as f:
            json.dump(consolidated_json, f, indent=2)
        print(f"✓ Consolidated JSON saved to {target_json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {target_json_path}: {e}")
    
    print(f"\nSummary: {processed_dirs} directories processed, {copied_files} files copied.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dirs", nargs="+", required=True, help="video dataset folders (each a 4_sample_distractor.py output)")
    parser.add_argument("--images-dir", required=True, help="target folder for all referenced frames")
    parser.add_argument("--json-path", required=True, help="target Image_details.json")
    a = parser.parse_args()
    consolidate_referenced_frames_and_json(a.base_dirs, a.images_dir, a.json_path)
