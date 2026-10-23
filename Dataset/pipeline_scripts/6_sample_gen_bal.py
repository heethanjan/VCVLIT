import os
import json
import random

# Define variables directly in the script
INPUT_FILE = "/data/scratch/projects/punim2198/heetha_scratch/data/Image_details.json"
OUTPUT_FILE = "/data/scratch/projects/punim2198/heetha_scratch/data/youtube_VR_one_disc_train_2.JSON"
IMAGES_DIR = "/data/scratch/projects/punim2198/heetha_scratch/data/Images"

def create_human_prompt(subtitle, prev_text, next_text):
    """
    Create the human prompt for the conversation.
    """
    return (
        "<image>\n<image>\n\n"
        "Select the image that best matches the subtitle below. When selecting, give the highest consideration to the CURRENT SUBTITLE, and then use the PREVIOUS and NEXT contexts as supporting clues.\n\n"
        "CURRENT SUBTITLE: \"{subtitle}\"\n"
        "PREVIOUS CONTEXT: \"{prev_text}\"\n"
        "NEXT CONTEXT: \"{next_text}\"\n\n"
        "Return only  <0 if Image 1 is correct, 1 if Image 2 is correct>\n dont include any other text other than 0 or 1"
        
    ).format(subtitle=subtitle, prev_text=prev_text, next_text=next_text)


def process_data(input_file, output_file, images_dir):
    """
    Process the input JSON file and generate the output JSON file in the required format.
    """
    # Load the input JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    output = []
    balance_counter = 0  # To alternate between 0 and 1 logits

    for key, details in data.items():
        # Extract the positive frame and distractor frames
        positive_frame = details.get("frame_path")
        distractor_frames = details.get("distractor_frames", [])

        if not positive_frame or not distractor_frames:
            continue

        # Randomly select a distractor frame
        distractor = random.choice(distractor_frames)
        distractor_frame = distractor.get("frame_path")

        # Alternate the order of positive and distractor frames
        if balance_counter % 2 == 0:
            images = [positive_frame, distractor_frame]
            selected_frame = 0
        else:
            images = [distractor_frame, positive_frame]
            selected_frame = 1

        # Extract subtitle and context
        subtitle = details.get("text", "")
        prev_text = details.get("context", {}).get("prev_text", "")
        next_text = details.get("context", {}).get("next_text", "")

        # Create the human and GPT conversation
        human_prompt = create_human_prompt(subtitle, prev_text, next_text)
        gpt_response = str(selected_frame)

        # Add the entry to the output
        output.append({
            "image": [
                os.path.join("Images", os.path.basename(images[0])),
                os.path.join("Images", os.path.basename(images[1]))
            ],
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": gpt_response}
            ]
        })

        balance_counter += 1

    # Save the output JSON file
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Processed {len(output)} entries. Output saved to {output_file}")

if __name__ == "__main__":
    process_data(INPUT_FILE, OUTPUT_FILE, IMAGES_DIR)