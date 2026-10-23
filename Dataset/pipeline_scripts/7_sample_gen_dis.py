import json
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class ImageDescriptionGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", batch_size=4):
        """Initialize the Qwen2.5-VL model for image description generation"""
        print(f"Loading model: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # ✅ Fix padding side for decoder-only models
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        self.model.eval()
        self.batch_size = batch_size
        print(f"✅ Model loaded successfully with batch_size={batch_size}")
    
    def _parse_prompt_from_conversation(self, conversation_value):
        """Extract subtitle, previous context, and next context from conversation"""
        subtitle_match = re.search(r'CURRENT SUBTITLE:\s*"([^"]+)"', conversation_value)
        subtitle = subtitle_match.group(1) if subtitle_match else ""
        
        prev_match = re.search(r'PREVIOUS CONTEXT:\s*"([^"]+)"', conversation_value)
        prev_text = prev_match.group(1) if prev_match else ""
        
        next_match = re.search(r'NEXT CONTEXT:\s*"([^"]+)"', conversation_value)
        next_text = next_match.group(1) if next_match else ""
        
        return subtitle, prev_text, next_text
    
    def _create_correct_image_prompt(self, subtitle, prev_text, next_text):
        """Create prompt for describing the correct image"""
        return (
            "Please analyze the image and give the description and why it matches the subtitle below. "
            "Give highest consideration to the CURRENT SUBTITLE, then use PREVIOUS and NEXT contexts as supporting clues.\n\n"
            f"CURRENT SUBTITLE: \"{subtitle}\"\n"
            f"PREVIOUS CONTEXT: \"{prev_text}\"\n"
            f"NEXT CONTEXT: \"{next_text}\"\n\n"
            "For the image:\n"
            "1. Describe what you see in detail (objects, text, scenes, actions)\n"
            "2. Explain how well it matches the current subtitle\n"
            "3. Consider the previous and next context for additional support\n\n"
            "Return your answer in JSON format:\n"
            "{\n"
            '  "desc_correct_img": "Detailed description of the image and why it matches the subtitle"\n'
            "}\n"
        )
    
    def _create_distractor_image_prompt(self, subtitle, prev_text, next_text):
        """Create prompt for describing the distractor image"""
        return (
            "Please analyze the image and give the description and why it doesn't match the subtitle below. "
            "Give highest consideration to the CURRENT SUBTITLE, then use PREVIOUS and NEXT contexts as supporting clues.\n\n"
            f"CURRENT SUBTITLE: \"{subtitle}\"\n"
            f"PREVIOUS CONTEXT: \"{prev_text}\"\n"
            f"NEXT CONTEXT: \"{next_text}\"\n\n"
            "For the image:\n"
            "1. Describe what you see in detail (objects, text, scenes, actions)\n"
            "2. Explain why it doesn't match the current subtitle\n"
            "3. Consider the previous and next context for additional support\n\n"
            "Return your answer in JSON format:\n"
            "{\n"
            '  "desc_disc_img": "Detailed description of the image and why it doesn\'t match the subtitle"\n'
            "}\n"
        )
    
    def _load_images_batch(self, image_paths):
        """Load multiple images in parallel"""
        def safe_load(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception as e:
                print(f"❌ Error loading {p}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            images = list(executor.map(safe_load, image_paths))
        return [img for img in images if img is not None]
    
    def generate_descriptions_batch(self, image_paths, prompts):
        """Generate descriptions for a batch of images - MUCH FASTER"""
        try:
            # ✅ Load images in parallel
            images = self._load_images_batch(image_paths)
            
            if len(images) != len(prompts):
                print(f"⚠️  Image count mismatch: {len(images)} images vs {len(prompts)} prompts")
                return [None] * len(prompts)
            
            # ✅ Prepare batch messages
            messages_batch = []
            for img, prompt in zip(images, prompts):
                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ])
            
            # ✅ Apply chat template for all
            texts = [
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ) for msgs in messages_batch
            ]
            
            # ✅ Process batch inputs
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            # ✅ Generate for entire batch with increased tokens
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # ✅ Increased to avoid truncation
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # ✅ Decode all outputs
            results = []
            for i in range(len(images)):
                generated_ids = output_ids[i][inputs.input_ids[i].shape[0]:]
                generated_text = self.processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                results.append(self._parse_json_response(generated_text))
            
            # Clean up
            del inputs, output_ids, images
            torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"❌ Batch generation error: {e}")
            import traceback
            traceback.print_exc()
            return [None] * len(image_paths)
    
    def _parse_json_response(self, generated_text):
        """Parse JSON response from model output with better error handling"""
        try:
            text = generated_text.strip()
            
            # Remove markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            text = text.strip()
            
            # Try to find JSON in the text
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                text = text[json_start:json_end]
            
            # ✅ Try to fix common JSON issues
            # Handle unterminated strings by finding last complete quote
            if text.count('"') % 2 != 0:
                # Odd number of quotes - try to close the string
                last_quote = text.rfind('"')
                if last_quote > 0:
                    text = text[:last_quote] + '"}'
            
            # Parse JSON
            parsed = json.loads(text)
            return parsed
            
        except json.JSONDecodeError as e:
            # ✅ Enhanced fallback: extract partial content
            try:
                # Try to extract the description value even if JSON is incomplete
                if "desc_correct_img" in generated_text:
                    match = re.search(r'"desc_correct_img"\s*:\s*"([^"]*)', generated_text)
                    if match:
                        return {"desc_correct_img": match.group(1) + "..."}
                elif "desc_disc_img" in generated_text:
                    match = re.search(r'"desc_disc_img"\s*:\s*"([^"]*)', generated_text)
                    if match:
                        return {"desc_disc_img": match.group(1) + "..."}
            except:
                pass
            
            # Last resort: return cleaned text
            clean_text = generated_text.replace('```json', '').replace('```', '').strip()
            if "correct" in clean_text.lower() or "match" in clean_text.lower():
                return {"desc_correct_img": clean_text[:200]}
            else:
                return {"desc_disc_img": clean_text[:200]}
    
    def _load_checkpoint(self, output_path):
        """Load existing checkpoint if it exists"""
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    checkpoint_data = json.load(f)
                print(f"✅ Loaded checkpoint with {len(checkpoint_data)} processed samples")
                return checkpoint_data
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                return []
        return []
    
    def _save_checkpoint(self, enhanced_dataset, output_path):
        """Save checkpoint to file"""
        try:
            # ✅ Create backup before saving
            if os.path.exists(output_path):
                backup_path = output_path + ".backup"
                import shutil
                shutil.copy2(output_path, backup_path)
            
            with open(output_path, 'w') as f:
                json.dump(enhanced_dataset, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            return False
    
    def _get_processed_indices(self, enhanced_dataset, original_dataset):
        """Get set of indices that have already been processed"""
        processed = set()
        
        # Create a mapping of original items to their indices
        for idx, orig_item in enumerate(original_dataset):
            for enh_item in enhanced_dataset:
                # Compare image paths to identify processed items
                if enh_item["image"] == orig_item["image"]:
                    processed.add(idx)
                    break
        
        return processed
    
    def process_dataset(self, json_path, image_base_path, output_path, save_every=5):
        """Process entire dataset with BATCHING for speed"""
        # Load dataset
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"📊 Total samples in dataset: {len(dataset)}")
        
        # ✅ Load existing checkpoint if available
        enhanced_dataset = self._load_checkpoint(output_path)
        processed_indices = self._get_processed_indices(enhanced_dataset, dataset)
        
        if processed_indices:
            print(f"🔄 Resuming from checkpoint. Already processed: {len(processed_indices)} samples")
            print(f"📝 Remaining: {len(dataset) - len(processed_indices)} samples")
        
        # ✅ Prepare batches
        remaining_items = [(idx, item) for idx, item in enumerate(dataset) if idx not in processed_indices]
        
        # Process in batches
        batch_count = 0
        for batch_start in tqdm(range(0, len(remaining_items), self.batch_size), desc="Processing batches"):
            batch_items = remaining_items[batch_start:batch_start + self.batch_size]
            
            # Prepare batch data
            correct_image_paths = []
            distractor_image_paths = []
            correct_prompts = []
            distractor_prompts = []
            batch_metadata = []
            
            for idx, item in batch_items:
                try:
                    images = item["image"]
                    conversation = item["conversations"][0]["value"]
                    correct_answer = item["conversations"][1]["value"]
                    
                    subtitle, prev_text, next_text = self._parse_prompt_from_conversation(conversation)
                    
                    # Determine correct and distractor images
                    if correct_answer == "0":
                        correct_img_path = Path(image_base_path) / images[0]
                        distractor_img_path = Path(image_base_path) / images[1]
                    else:
                        correct_img_path = Path(image_base_path) / images[1]
                        distractor_img_path = Path(image_base_path) / images[0]
                    
                    correct_image_paths.append(correct_img_path)
                    distractor_image_paths.append(distractor_img_path)
                    
                    correct_prompts.append(self._create_correct_image_prompt(subtitle, prev_text, next_text))
                    distractor_prompts.append(self._create_distractor_image_prompt(subtitle, prev_text, next_text))
                    
                    batch_metadata.append((idx, item))
                    
                except Exception as e:
                    print(f"❌ Error preparing sample {idx}: {e}")
                    continue
            
            if not batch_metadata:
                continue
            
            # ✅ Generate descriptions for entire batch
            correct_descs = self.generate_descriptions_batch(correct_image_paths, correct_prompts)
            distractor_descs = self.generate_descriptions_batch(distractor_image_paths, distractor_prompts)
            
            # ✅ Create enhanced items
            for (idx, item), correct_desc, distractor_desc in zip(batch_metadata, correct_descs, distractor_descs):
                enhanced_item = {
                    "image": item["image"],
                    "conversations": item["conversations"],
                    "descriptions": {
                        "desc_correct_img": correct_desc.get("desc_correct_img", "") if correct_desc else "",
                        "desc_disc_img": distractor_desc.get("desc_disc_img", "") if distractor_desc else ""
                    }
                }
                enhanced_dataset.append(enhanced_item)
            
            batch_count += 1
            
            # ✅ Save checkpoint every N batches
            if batch_count % save_every == 0:
                if self._save_checkpoint(enhanced_dataset, output_path):
                    print(f"\n💾 Checkpoint saved ({len(enhanced_dataset)}/{len(dataset)} samples)")
        
        # Final save
        self._save_checkpoint(enhanced_dataset, output_path)
        
        print(f"\n✅ Processing complete! Saved {len(enhanced_dataset)} enhanced samples to {output_path}")
        return enhanced_dataset


def main():
    # Paths
    json_path = "/data/scratch/projects/punim2198/heetha_scratch/data/youtube_VR_one_disc_train_2.JSON"
    image_base_path = "/data/scratch/projects/punim2198/heetha_scratch/data"
    output_path = "/data/scratch/projects/punim2198/heetha_scratch/data/youtube_VR_one_disc_train_enhanced_2.json"
    
    # ✅ Smaller batch size to reduce errors
    generator = ImageDescriptionGenerator(batch_size=64)
    
    # Process dataset with checkpointing every 5 batches
    enhanced_dataset = generator.process_dataset(
        json_path, 
        image_base_path, 
        output_path,
        save_every=1  # Save every 5 batches (20 samples)
    )
    
    print(f"\n📊 Summary:")
    print(f"   Total samples processed: {len(enhanced_dataset)}")
    print(f"   Output saved to: {output_path}")


if __name__ == "__main__":
    main()