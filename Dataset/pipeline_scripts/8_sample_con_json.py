import json

def transform_dataset(input_path, output_path):
    """
    Transform the enhanced dataset to match image order with renamed keys.
    - desc_correct_img and desc_disc_img → desc_img_0 and desc_img_1
    - Order based on the conversation answer value (0 or 1)
    """
    # Load the enhanced dataset
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    transformed_dataset = []
    
    for item in dataset:
        try:
            # Get the answer (0 or 1) from conversations
            answer = item["conversations"][1]["value"].strip()
            
            # Get the original descriptions
            desc_correct = item["descriptions"].get("desc_correct_img", "")
            desc_distractor = item["descriptions"].get("desc_disc_img", "")
            
            # Create new descriptions dictionary based on answer
            if answer == "0":
                # Image 0 is correct, Image 1 is distractor
                new_descriptions = {
                    "desc_img_0": desc_correct,
                    "desc_img_1": desc_distractor
                }
            elif answer == "1":
                # Image 1 is correct, Image 0 is distractor
                new_descriptions = {
                    "desc_img_0": desc_distractor,
                    "desc_img_1": desc_correct
                }
            else:
                print(f"⚠️  Unknown answer value: {answer}, skipping item")
                continue
            
            # Create transformed item
            transformed_item = {
                "image": item["image"],
                "conversations": item["conversations"],
                "descriptions": new_descriptions
            }
            
            transformed_dataset.append(transformed_item)
            
        except Exception as e:
            print(f"❌ Error processing item: {e}")
            continue
    
    # Save transformed dataset
    with open(output_path, 'w') as f:
        json.dump(transformed_dataset, f, indent=2)
    
    print(f"✅ Transformation complete!")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Total items: {len(dataset)}")
    print(f"   Transformed items: {len(transformed_dataset)}")
    
    return transformed_dataset


if __name__ == "__main__":
    input_path = "/data/scratch/projects/punim2198/heetha_scratch/data/youtube_VR_one_disc_train_enhanced_2.json"
    output_path = "/data/scratch/projects/punim2198/heetha_scratch/data/youtube_VR_one_disc_train_transformed_2.json"
    
    transform_dataset(input_path, output_path)