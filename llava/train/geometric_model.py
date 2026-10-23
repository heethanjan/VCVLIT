import torch
import torch.nn as nn
import json
import re
# import copy # ✅ NEW: Import the copy module
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
)
from .geometric_contrastive import (
    GeometricContrastiveLoss,
    VisionProjectorMLP,
    TextProjectorMLP,
    DescriptionExtractor,
)

class GeometricContrastiveWrapper(nn.Module):
    """Wrapper that adds geometric contrastive learning to Qwen VL models"""
    def __init__(self, base_model, model_args, processor):
        super().__init__()
        self.base_model = base_model
        self.model_args = model_args
        self.processor = processor
        self.enable_contrastive = model_args.enable_contrastive_learning
        self.contrastive_only = model_args.contrastive_only
        
        self.training_step = 0

        # # ✅ NEW: Add a separate, isolated model for generation to prevent DDP deadlocks.
        # self.generator_model = copy.deepcopy(base_model)
        # self.generator_model.eval() # Keep it permanently in eval mode
        # print("✅ Created a separate, isolated generator model.")

        # # ✅ NEW: Get the update frequency from model_args
        # self.generator_update_steps = getattr(model_args, "generator_update_steps", 500)
        # print(f"🔄 Generator model will be updated every {self.generator_update_steps} steps.")
        
        # ✅ FIX: Get correct vision hidden size from merger output
        if hasattr(base_model, "visual") and hasattr(base_model.visual, "merger"):
            # For Qwen2.5-VL, the merger outputs to LLM hidden size
            vision_hidden = base_model.config.hidden_size  # This is 2048 for 3B model
            print(f"🔍 Using LLM hidden size for vision projection: {vision_hidden}")
        elif hasattr(base_model, "visual") and hasattr(base_model.visual, "config"):
            vision_hidden = base_model.visual.config.hidden_size
            print(f"🔍 Using visual.config.hidden_size: {vision_hidden}")
        elif hasattr(base_model, "vision_model") and hasattr(base_model.vision_model, "config"):
            vision_hidden = base_model.vision_model.config.hidden_size
            print(f"🔍 Using vision_model.config.hidden_size: {vision_hidden}")
        else:
            raise ValueError("Cannot find vision hidden size on base_model.")
    
        text_hidden = base_model.config.hidden_size
        print(f"🔍 Text hidden size: {text_hidden}")

        if self.enable_contrastive:
            # Load frozen text-embedding model
            self.text_embedding_model = AutoModel.from_pretrained(
                "Qwen/Qwen3-Embedding-0.6B", 
                torch_dtype=torch.bfloat16
            )
            self.text_embedding_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-0.6B"
            )
            for p in self.text_embedding_model.parameters():
                p.requires_grad = False


            # ✅ FIX: Ensure vision_projector gets correct input_dim (2048)
            self.vision_projector = VisionProjectorMLP(
                input_dim=vision_hidden,  # Should be 2048 for Qwen2.5-VL-3B
                output_dim=model_args.vision_embed_dim  # 256
            )
            
            print(f"✅ Vision Projector initialized: input_dim={vision_hidden}, output_dim={model_args.vision_embed_dim}")

            text_embed_dim = self.text_embedding_model.config.hidden_size  # 896
            self.text_projector = TextProjectorMLP(
                input_dim=text_embed_dim, 
                output_dim=getattr(model_args, "text_embed_dim", model_args.vision_embed_dim)
            )
            
            print(f"✅ Text Projector initialized: input_dim={text_embed_dim}, output_dim={model_args.vision_embed_dim}")

            self.contrastive_loss = GeometricContrastiveLoss(
                temperature=model_args.contrastive_temperature,
                margin=model_args.contrastive_margin,
            )
            self.contrastive_loss_weight = model_args.contrastive_loss_weight

            self.description_extractor = DescriptionExtractor(processor)

        # Buffers
        self.vision_tokens_list = []
        self.current_num_images = 0

        # Register hooks
        self._register_hooks()

        # ✅ Store initial weights for comparison
        self.initial_weights = {}
        if self.enable_contrastive:
            self._save_initial_weights()
    
    def _save_initial_weights(self):
        """Save initial weights of VLM components on GPU"""
        device = next(self.base_model.parameters()).device  # ✅ Get model device
        for name, param in self.base_model.named_parameters():
            if param.requires_grad and ('visual' in name or 'merger' in name):
                self.initial_weights[name] = param.data.clone().detach().to(device)  # ✅ Keep on GPU
        print(f"📸 Saved {len(self.initial_weights)} initial weights for tracking (device: {device})")

    def check_weight_changes(self):
        """Check if VLM weights have changed"""
        changes = {}
        device = next(self.base_model.parameters()).device  # ✅ Get current device
        
        for name, initial_weight in self.initial_weights.items():
            try:
                current_weight = dict(self.base_model.named_parameters())[name].data
                
                # ✅ FIX: Move initial_weight to same device as current_weight
                if initial_weight.device != current_weight.device:
                    initial_weight = initial_weight.to(current_weight.device)
                
                diff = (current_weight - initial_weight).abs().mean().item()
                changes[name] = diff
            except KeyError:
                # Parameter might have been removed
                continue
    
        # Print top 5 largest changes
        if changes:
            sorted_changes = sorted(changes.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🔍 Top 5 Weight Changes:")
            for name, diff in sorted_changes[:5]:
                print(f"   {name}: Δ = {diff:.8f}")
        else:
            print("\n⚠️  No weight changes tracked")
        
        return changes
    
    def _register_hooks(self):
        """Register smart hook that distinguishes training from generation"""
        def vision_hook(module, inputs, output):
            # ✅ SIMPLIFIED HOOK: The hook is only on the base_model, so it only runs during training.
            # No need for the _is_generating flag anymore.
            if self.base_model.training:
                self.vision_tokens_list.append(output)

        if hasattr(self.base_model, "visual") and hasattr(self.base_model.visual, "merger"):
            self.base_model.visual.merger.register_forward_hook(vision_hook)
            print("✅ Hook registered on base_model.visual.merger")
        else:
            raise RuntimeError("Cannot find visual.merger for hook")


    def _parse_prompt(self, prompt):
        """
        Extract subtitle, previous text, and next text from the prompt string.
        
        Args:
            prompt: String containing the full prompt with CURRENT SUBTITLE, PREVIOUS CONTEXT, and NEXT CONTEXT
            
        Returns:
            tuple: (subtitle, prev_text, next_text)
        """
        import re
        
        # Extract CURRENT SUBTITLE
        subtitle_match = re.search(r'CURRENT SUBTITLE:\s*"([^"]+)"', prompt)
        subtitle = subtitle_match.group(1) if subtitle_match else ""
        
        # Extract PREVIOUS CONTEXT
        prev_match = re.search(r'PREVIOUS CONTEXT:\s*"([^"]+)"', prompt)
        prev_text = prev_match.group(1) if prev_match else ""
        
        # Extract NEXT CONTEXT
        next_match = re.search(r'NEXT CONTEXT:\s*"([^"]+)"', prompt)
        next_text = next_match.group(1) if next_match else ""
        
        print(f"📝 Parsed prompt:")
        print(f"   Subtitle: {subtitle}")
        print(f"   Previous: {prev_text}")
        print(f"   Next: {next_text}")
        
        return subtitle, prev_text, next_text

    def _create_description_prompt(self, subtitle, prev_text, next_text):
        """
        Create task-aware prompt for description generation.
        This prompt matches the training task (image selection based on subtitle).
        """
        return (
            "Please analyze the images and select which one best matches the subtitle below. "
            "Give highest consideration to the CURRENT SUBTITLE, then use PREVIOUS and NEXT contexts as supporting clues.\n\n"
            f"CURRENT SUBTITLE: \"{subtitle}\"\n"
            f"PREVIOUS CONTEXT: \"{prev_text}\"\n"
            f"NEXT CONTEXT: \"{next_text}\"\n\n"
            "For each image:\n"
            "1. Describe what you see in detail (objects, text, scenes, actions)\n"
            "2. Explain how well it matches the current subtitle\n"
            "3. Consider the previous and next context for additional support\n\n"
            "Return your answer in JSON format with 0-indexed image numbers:\n"
            "{\n"
            '  "selected_frame": <0 for first image, 1 for second image, etc.>,\n'
            '  "desc_IMAGE_0": "Detailed description of first image and why it matches/doesn\'t match the subtitle",\n'
            '  "desc_IMAGE_1": "Detailed description of second image and why it matches/doesn\'t match the subtitle",\n'
            '  "reasoning": "Overall reasoning for your selection"\n'
            "}\n"
        )
            
        
    def parse_json_response(self, generated_text):
        """Parse the JSON response from the model"""
        try:
            text = generated_text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            
            text = text.strip()
            
            # Parse JSON
            descriptions = json.loads(text)
            return descriptions
            
        except json.JSONDecodeError:
            # Fallback: try regex patterns
            descriptions = {}
            patterns = [
                r'"desc_IMAGE_(\d+)":\s*"([^"]+)"',
                r'desc_IMAGE_(\d+):\s*([^\n]+)',
                r'IMAGE (\d+):\s*([^\n]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, generated_text, re.DOTALL)
                for img_num, desc in matches:
                    descriptions[f"desc_IMAGE_{img_num}"] = desc.strip()
            
            return descriptions

    def split_vision_tokens_by_image(self, vision_tokens, num_images):
        """Split concatenated vision tokens back into per-image tokens."""
        if vision_tokens is None or num_images == 0:
            return []
        
        print(f"🔍 Vision tokens shape: {vision_tokens.shape}, num_images: {num_images}")
        
        # ✅ Handle 2D tensor: [total_tokens, hidden_dim]
        if vision_tokens.dim() == 2:
            total_tokens = vision_tokens.size(0)  # 2736
            hidden_dim = vision_tokens.size(1)     # 2048
            tokens_per_image = total_tokens // num_images  # 2736 // 5 = 547
            
            image_tokens = []
            for i in range(num_images):
                start_idx = i * tokens_per_image
                end_idx = start_idx + tokens_per_image if i < num_images - 1 else total_tokens
                
                # ✅ Index 2D tensor correctly: [start:end, :]
                img_tokens = vision_tokens[start_idx:end_idx, :]  # [tokens_per_img, hidden_dim]
                
                # Add batch dimension for compatibility: [1, tokens_per_img, hidden_dim]
                img_tokens = img_tokens.unsqueeze(0)
                
                image_tokens.append(img_tokens)
                print(f"  Image {i}: tokens shape {img_tokens.shape}")
            
            return image_tokens
        
        # ✅ Handle 3D tensor: [batch_size, total_tokens, hidden_dim]
        elif vision_tokens.dim() == 3:
            batch_size = vision_tokens.size(0)
            total_tokens = vision_tokens.size(1)
            hidden_dim = vision_tokens.size(2)
            tokens_per_image = total_tokens // num_images
            
            image_tokens = []
            for i in range(num_images):
                start_idx = i * tokens_per_image
                end_idx = start_idx + tokens_per_image if i < num_images - 1 else total_tokens
                
                # Index 3D tensor: [batch, start:end, :]
                img_tokens = vision_tokens[:, start_idx:end_idx, :]  # [batch, tokens_per_img, hidden_dim]
                image_tokens.append(img_tokens)
                print(f"  Image {i}: tokens shape {img_tokens.shape}")
            
            return image_tokens
        
        else:
            raise ValueError(f"Unexpected vision_tokens shape: {vision_tokens.shape}. Expected 2D or 3D tensor.")

    def extract_text_embeddings_with_qwen3(self, descriptions):
        """Encode descriptions using Qwen3-Embedding-0.6B"""
        if not descriptions:
            return None
        
        device = next(self.text_embedding_model.parameters()).device
        tok = self.text_embedding_tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():  # ✅ Freeze Qwen3-Embedding
            out = self.text_embedding_model(**tok)
        
        attn = tok["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * attn).sum(dim=1) / (attn.sum(dim=1).clamp(min=1e-6))
        return pooled.float()

    def extract_image_embeddings(self, vision_tokens):
        """Mean-pool vision tokens and project."""
        if vision_tokens is None:
            return None
        return self.vision_projector(vision_tokens)

    def extract_text_embeddings(self, text_embeddings):
        """Project text embeddings to contrastive space."""
        if text_embeddings is None:
            return None
        return self.text_projector(text_embeddings)

    def forward(self, **kwargs):
        # Extract metadata
        image_groups = kwargs.pop("image_groups", None)
        descriptions_batch = kwargs.pop("descriptions_batch", None) 
        prompts = kwargs.pop("prompts", None)
        valid_indices = kwargs.pop("valid_indices", None)
        
        self.vision_tokens_list = []
        
        # --- Step 1: Run the main VLM forward pass ---
        if self.contrastive_only:
            with torch.no_grad():
                outputs = self.base_model(**kwargs)
            outputs.loss = torch.tensor(0.0).to(self.base_model.device)
        else:
            outputs = self.base_model(**kwargs)
        
        original_vlm_loss = outputs.loss
        vlm_loss_value = original_vlm_loss.item() if original_vlm_loss is not None else 0.0

        # --- Step 2: On-the-fly generation and contrastive loss calculation ---
        if self.enable_contrastive and self.training and image_groups and descriptions_batch:
            
            all_txt_emb_raw = []
            with torch.no_grad():
                self.text_embedding_model.eval()
                
                for batch_idx, descriptions in enumerate(descriptions_batch):
                    if not descriptions:
                        all_txt_emb_raw.append(None)
                        continue
                    
                    # Get text embeddings for pre-generated descriptions
                    txt_emb_raw = self.extract_text_embeddings_with_qwen3(descriptions)
                    all_txt_emb_raw.append(txt_emb_raw)
                
            
                
                torch.cuda.empty_cache()
            
            # --- Sub-step 2b: Compute contrastive loss ---
            if not self.vision_tokens_list:
                return outputs
            
            vision_tokens = self.vision_tokens_list[-1]
            
            self.vision_projector.train()
            self.text_projector.train()
            
            all_img_embeddings = []
            all_txt_embeddings = []
            
            # ✅ FIX: Align with the generation loop
            for batch_idx, (images, txt_emb_raw) in enumerate(zip(image_groups, all_txt_emb_raw)):
                if txt_emb_raw is None:
                    continue
                
                # Project vision tokens
                per_image_tokens = self.split_vision_tokens_by_image(vision_tokens, len(images))
                img_embeddings_list = [self.extract_image_embeddings(img_tokens) for img_tokens in per_image_tokens]
                img_emb = torch.cat(img_embeddings_list, dim=0)
                
                # Project text embeddings
                txt_emb = self.extract_text_embeddings(txt_emb_raw.to(img_emb.device))
                
                all_img_embeddings.append(img_emb)
                all_txt_embeddings.append(txt_emb)
            
            if all_img_embeddings and all_txt_embeddings:
                img_emb_batch = torch.cat(all_img_embeddings, dim=0)
                txt_emb_batch = torch.cat(all_txt_embeddings, dim=0)
                
                contrastive_loss = self.contrastive_loss(img_emb_batch, txt_emb_batch) * self.contrastive_loss_weight
                contrastive_loss_value = contrastive_loss.item()  # ✅ FIX: Define this variable
                
                if self.contrastive_only:
                    total_loss = contrastive_loss
                else:
                    total_loss = original_vlm_loss + contrastive_loss
                
                # ✅ FIX: Update outputs.loss for BOTH modes (moved outside else)
                if isinstance(outputs, dict):
                    outputs["loss"] = total_loss
                else:
                    outputs.loss = total_loss
                
                # ✅ W&B LOGGING: Log all losses
                try:
                    import wandb
                    if wandb.run is not None:
                        # Compute cosine similarity for monitoring
                        cos_sim = torch.nn.functional.cosine_similarity(
                            img_emb_batch, txt_emb_batch, dim=-1
                        ).mean().item()
                        
                        # ✅ Prepare logging dict
                        log_dict = {
                            "train/contrastive_loss": contrastive_loss_value,
                            "train/total_loss": total_loss.item(),
                            "contrastive/num_samples": len(img_emb_batch),
                            "contrastive/num_groups": len(all_img_embeddings),
                            "contrastive/negatives_per_positive": len(img_emb_batch) - 1,
                            "contrastive/img_emb_norm": torch.norm(img_emb_batch, dim=-1).mean().item(),
                            "contrastive/txt_emb_norm": torch.norm(txt_emb_batch, dim=-1).mean().item(),
                            "contrastive/cosine_similarity": cos_sim,
                            "contrastive/img_emb_std": img_emb_batch.std().item(),
                            "contrastive/txt_emb_std": txt_emb_batch.std().item(),
                        }
                        
                        # ✅ Add VLM loss if not in contrastive_only mode
                        if not self.contrastive_only and vlm_loss_value is not None:
                            log_dict["train/vlm_loss"] = vlm_loss_value
                        
                        # ✅ Add mode indicator
                        log_dict["train/contrastive_only_mode"] = 1.0 if self.contrastive_only else 0.0
                        
                        wandb.log(log_dict, step=self.training_step)
                        
                        print(f"📊 W&B Metrics:")
                        print(f"   Total Loss: {total_loss.item():.4f}")
                        if not self.contrastive_only:
                            print(f"   VLM Loss: {vlm_loss_value:.4f}")
                        print(f"   Contrastive Loss: {contrastive_loss_value:.4f}")
                        print(f"   Cosine Similarity: {cos_sim:.4f}")
                except ImportError:
                    print("⚠️  wandb not available, skipping logging")

        elif self.training and vlm_loss_value is not None:
            try:
                import wandb
                if wandb.run is not None:
                    log_dict = {
                        "train/vlm_loss": vlm_loss_value,
                        "train/total_loss": vlm_loss_value,
                        "train/contrastive_only_mode": 0.0,
                        "train/enable_contrastive": 0.0,
                    }
                    wandb.log(log_dict, step=self.training_step)
            except ImportError:
                pass
            
        self.training_step += 1
        
        # ✅ Check weight changes every 10 steps (moved outside the if block)
        if self.training_step > 0 and self.training_step % 10 == 0:
            self.check_weight_changes()
        
        return outputs
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)