import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .geometric_contrastive import (
    GeometricContrastiveLoss,
    VisionProjectorMLP,
    TextProjectorMLP,
)
import wandb
import traceback
import os


class LLaVAGeometricWrapper(nn.Module):
    """Wrapper that adds geometric contrastive learning to LLaVA models"""
    
    def __init__(self, base_model, model_args):
        super().__init__()
        self.base_model = base_model
        self.model_args = model_args
        self.enable_contrastive = getattr(model_args, 'enable_contrastive_learning', False)
        self.contrastive_only = getattr(model_args, 'contrastive_only', False)
        
        self.training_step = 0
        
        # Get LLM hidden size
        if hasattr(base_model, 'config'):
            text_hidden = base_model.config.hidden_size
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'config'):
            text_hidden = base_model.model.config.hidden_size
        else:
            raise ValueError("Cannot determine text hidden size from model")
        
        print(f"🔍 LLaVA Text hidden size: {text_hidden}")
        
        if self.enable_contrastive:
            # Load frozen text-embedding model
            self.text_embedding_model = AutoModel.from_pretrained(
                "BAAI/bge-large-en-v1.5",
                torch_dtype=torch.bfloat16
            ).to("cuda")

            self.text_embedding_tokenizer = AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5"
            )
            for p in self.text_embedding_model.parameters():
                p.requires_grad = False
            self.text_embedding_model.eval()
            
            print("✅ Loaded frozen text embedding model: BAAI/bge-large-en-v1.5")
            
            # Initialize projectors
            embed_dim = getattr(model_args, "vision_embed_dim", 256)
            self.vision_projector = VisionProjectorMLP(
                input_dim=text_hidden,
                hidden_dim=512,
                output_dim=embed_dim,
                dropout=0.1
            )
            
            text_embed_dim = self.text_embedding_model.config.hidden_size
            self.text_projector = TextProjectorMLP(
                input_dim=text_embed_dim,
                hidden_dim=512,
                output_dim=embed_dim,
                dropout=0.1
            )
            
            # ✅ NEW: Load pre-trained projectors if paths are provided
            pretrained_vision_proj = getattr(model_args, 'pretrained_vision_projector', None)
            pretrained_text_proj = getattr(model_args, 'pretrained_text_projector', None)
            
            if pretrained_vision_proj and os.path.exists(pretrained_vision_proj):
                self._load_vision_projector(pretrained_vision_proj)
            else:
                print(f"⚠️  No pre-trained vision projector found at: {pretrained_vision_proj}")
                print(f"   Initializing with random weights")
            
            if pretrained_text_proj and os.path.exists(pretrained_text_proj):
                self._load_text_projector(pretrained_text_proj)
            else:
                print(f"⚠️  No pre-trained text projector found at: {pretrained_text_proj}")
                print(f"   Initializing with random weights")
            
            # ✅ Optionally freeze projectors
            freeze_projectors = getattr(model_args, 'freeze_contrastive_projectors', False)
            if freeze_projectors:
                print("🔒 Freezing contrastive projectors...")
                for p in self.vision_projector.parameters():
                    p.requires_grad = False
                for p in self.text_projector.parameters():
                    p.requires_grad = False
                print("✅ Contrastive projectors frozen")
            else:
                print("🔓 Contrastive projectors are trainable")
            
            # Contrastive loss
            self.contrastive_loss = GeometricContrastiveLoss(
                temperature=getattr(model_args, 'contrastive_temperature', 0.07),
                margin=getattr(model_args, 'contrastive_margin', 0.5),
            )
            self.contrastive_loss_weight = getattr(model_args, 'contrastive_loss_weight', 0.1)
            print(f"✅ Contrastive loss weight: {self.contrastive_loss_weight}")
        
        self.vision_tokens_list = []
        self._register_hooks()
    
    def _load_vision_projector(self, checkpoint_path):
        """Load pre-trained vision projector weights"""
        try:
            print(f"\n{'='*80}")
            print(f"📥 LOADING VISION PROJECTOR")
            print(f"   Path: {checkpoint_path}")
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle if state_dict contains nested structure
            if isinstance(state_dict, dict) and 'vision_projector' in state_dict:
                state_dict = state_dict['vision_projector']
            
            missing, unexpected = self.vision_projector.load_state_dict(state_dict, strict=True)
            
            print(f"✅ Vision projector loaded successfully!")
            print(f"   Parameters: {len(state_dict)}")
            if missing:
                print(f"   ⚠️  Missing keys: {missing}")
            if unexpected:
                print(f"   ⚠️  Unexpected keys: {unexpected}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Failed to load vision projector: {e}")
            traceback.print_exc()
            print(f"⚠️  Continuing with randomly initialized vision projector")
    
    def _load_text_projector(self, checkpoint_path):
        """Load pre-trained text projector weights"""
        try:
            print(f"\n{'='*80}")
            print(f"📥 LOADING TEXT PROJECTOR")
            print(f"   Path: {checkpoint_path}")
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle if state_dict contains nested structure
            if isinstance(state_dict, dict) and 'text_projector' in state_dict:
                state_dict = state_dict['text_projector']
            
            missing, unexpected = self.text_projector.load_state_dict(state_dict, strict=True)
            
            print(f"✅ Text projector loaded successfully!")
            print(f"   Parameters: {len(state_dict)}")
            if missing:
                print(f"   ⚠️  Missing keys: {missing}")
            if unexpected:
                print(f"   ⚠️  Unexpected keys: {unexpected}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Failed to load text projector: {e}")
            traceback.print_exc()
            print(f"⚠️  Continuing with randomly initialized text projector")
    
    def _register_hooks(self):
        """Register forward hook on mm_projector to capture projected embeddings"""
        def projector_hook(module, inputs, output):
            if self.base_model.training and self.enable_contrastive:
                self.vision_tokens_list.append(output)
        
        projector = None
        if hasattr(self.base_model, 'get_model'):
            model = self.base_model.get_model()
            if hasattr(model, 'mm_projector'):
                projector = model.mm_projector
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'mm_projector'):
            projector = self.base_model.model.mm_projector
        
        if projector is not None:
            projector.register_forward_hook(projector_hook)
            print("✅ Hook registered on LLaVA mm_projector")
        else:
            print("⚠️  Cannot find mm_projector, hook not registered")
    
    def extract_text_embeddings(self, descriptions):
        """Encode descriptions using sentence transformer"""
        if not descriptions or len(descriptions) == 0:
            return None
        
        device = "cuda"
        
        inputs = self.text_embedding_tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.inference_mode():
            outputs = self.text_embedding_model(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / \
                         attention_mask.sum(dim=1).clamp(min=1e-9)
    
        embeddings = embeddings.to(dtype=torch.bfloat16)
        embeddings = self.text_projector(embeddings)
        
        return embeddings
    
    def extract_vision_embeddings(self, vision_tokens, num_images_per_sample):
        """Extract and project vision embeddings"""
        if vision_tokens is None or len(vision_tokens) == 0:
            return None
        
        vision_features = vision_tokens.mean(dim=1)
        vision_embeddings = self.vision_projector(vision_features)
        
        return vision_embeddings
    
    def forward(self, **kwargs):
        descriptions_batch = kwargs.pop("descriptions_batch", None)
        num_images_list = kwargs.pop("num_images_list", None)
        
        # Debug: Print detailed data for first 3 training steps
        if self.training_step < 3:
            print(f"\n{'#'*100}")
            print(f"🔍 TRAINING STEP {self.training_step} - DETAILED INPUT DATA")
            print(f"{'#'*100}")
            
            # Get tokenizer for decoding
            tokenizer = None
            try:
                if hasattr(self.base_model, 'get_model'):
                    model = self.base_model.get_model()
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                if tokenizer is None and hasattr(self.base_model, 'tokenizer'):
                    tokenizer = self.base_model.tokenizer
            except:
                pass
            
            # Show all kwargs
            print(f"\n📦 All kwargs keys: {list(kwargs.keys())}")
            
            # Show images (handle both list and tensor)
            if 'images' in kwargs and kwargs['images'] is not None:
                images = kwargs['images']
                print(f"\n🖼️  IMAGES:")
                if isinstance(images, list):
                    print(f"   Type: list of {len(images)} tensors")
                    for i, img in enumerate(images):
                        if img is not None and isinstance(img, torch.Tensor):
                            print(f"   Image {i}: shape={img.shape}, dtype={img.dtype}, device={img.device}")
                            if img.numel() > 0:
                                print(f"              min={img.min().item():.3f}, max={img.max().item():.3f}")
                elif isinstance(images, torch.Tensor):
                    print(f"   Type: single tensor")
                    print(f"   Shape: {images.shape}")
                    print(f"   Dtype: {images.dtype}")
                    print(f"   Device: {images.device}")
                    if images.numel() > 0:
                        print(f"   Min/Max: {images.min().item():.3f} / {images.max().item():.3f}")
                else:
                    print(f"   Type: {type(images)}")
            else:
                print(f"\n⚠️  NO IMAGES in kwargs")
            
            # Show input_ids and decode prompts
            if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
                input_ids = kwargs['input_ids']
                print(f"\n📝 INPUT_IDS (Prompts):")
                print(f"   Shape: {input_ids.shape}")
                print(f"   Dtype: {input_ids.dtype}")
                
                # Decode prompts for each sample in batch
                batch_size = input_ids.shape[0]
                for i in range(min(batch_size, 2)):  # Show first 2 samples max
                    print(f"\n   Sample {i}:")
                    if tokenizer is not None:
                        try:
                            decoded = tokenizer.decode(input_ids[i], skip_special_tokens=False)
                            print(f"   DECODED PROMPT:")
                            print(f"   {'-'*80}")
                            print(f"   {decoded[:1000]}...")  # Show first 1000 chars
                            print(f"   {'-'*80}")
                        except Exception as e:
                            print(f"   ⚠️  Failed to decode: {e}")
                            print(f"   Token IDs (first 50): {input_ids[i][:50].tolist()}")
                    else:
                        print(f"   ⚠️  No tokenizer available")
                        print(f"   Token IDs (first 50): {input_ids[i][:50].tolist()}")
            
            # Show labels
            if 'labels' in kwargs and kwargs['labels'] is not None:
                labels = kwargs['labels']
                print(f"\n🎯 LABELS (Expected outputs):")
                print(f"   Shape: {labels.shape}")
                print(f"   Dtype: {labels.dtype}")
                print(f"   Non-ignore tokens: {(labels != -100).sum().item()}")
                
                if tokenizer is not None:
                    try:
                        # Show what tokens should be predicted
                        for i in range(min(labels.shape[0], 1)):
                            valid_labels = labels[i][labels[i] != -100]
                            if len(valid_labels) > 0:
                                decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=False)
                                print(f"\n   Sample {i} expected output:")
                                print(f"   {'-'*80}")
                                print(f"   {decoded_labels[:500]}...")
                                print(f"   {'-'*80}")
                    except Exception as e:
                        print(f"   ⚠️  Failed to decode labels: {e}")
            
            # Show attention mask
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                print(f"\n👁️  ATTENTION_MASK:")
                print(f"   Shape: {kwargs['attention_mask'].shape}")
                print(f"   Active tokens: {kwargs['attention_mask'].sum().item()}")
            
            # Show descriptions for contrastive learning
            if descriptions_batch is not None:
                print(f"\n📄 DESCRIPTIONS (for contrastive learning):")
                print(f"   Number of descriptions: {len(descriptions_batch)}")
                for i, desc in enumerate(descriptions_batch[:2]):
                    print(f"\n   Description {i}:")
                    if desc:
                        for key, val in desc.items():
                            print(f"      {key}: {val[:200]}...")
            
            print(f"\n{'#'*100}\n")
        
        self.vision_tokens_list = []
        
        if self.contrastive_only:
            with torch.no_grad():
                outputs = self.base_model(**kwargs)
            outputs.loss = None
        else:
            outputs = self.base_model(**kwargs)
        
        original_lm_loss = outputs.loss
        lm_loss_value = original_lm_loss.item() if original_lm_loss is not None else 0.0
        
        contrastive_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.enable_contrastive and self.training and descriptions_batch:
            try:
                if len(self.vision_tokens_list) > 0:
                    vision_tokens = self.vision_tokens_list[-1]
                    vision_embeddings = self.extract_vision_embeddings(vision_tokens, num_images_list)
                    
                    all_descriptions = []
                    for desc_dict in descriptions_batch:
                        if desc_dict:
                            if 'desc_img_0' in desc_dict:
                                all_descriptions.append(desc_dict['desc_img_0'])
                            if 'desc_img_1' in desc_dict:
                                all_descriptions.append(desc_dict['desc_img_1'])
                    
                    if all_descriptions and vision_embeddings is not None:
                        text_embeddings = self.extract_text_embeddings(all_descriptions)
                        
                        if text_embeddings is not None and vision_embeddings.shape[0] == text_embeddings.shape[0]:
                            cos_sim = F.cosine_similarity(vision_embeddings, text_embeddings, dim=-1)
                            contrastive_loss = self.contrastive_loss(vision_embeddings, text_embeddings)
                            
                            try:
                                wandb.log({
                                    'train/cosine_similarity_mean': cos_sim.mean().item(),
                                    'train/contrastive_loss': contrastive_loss.item(),
                                    'train/lm_loss': lm_loss_value,
                                    'train/step': self.training_step,
                                })
                            except Exception:
                                pass
                
            except Exception as e:
                print(f"❌ Error computing contrastive loss: {e}")
                traceback.print_exc()
    
        if original_lm_loss is not None and contrastive_loss.item() > 0:
            total_loss = original_lm_loss + self.contrastive_loss_weight * contrastive_loss
            outputs.loss = total_loss
        elif self.contrastive_only and contrastive_loss.item() > 0:
            outputs.loss = contrastive_loss
    
        self.training_step += 1
        
        return outputs
    
    def state_dict(self, *args, **kwargs):
        state = {}
        if hasattr(self.base_model, 'state_dict'):
            state.update(self.base_model.state_dict(*args, **kwargs))
        
        if self.enable_contrastive:
            for k, v in self.vision_projector.state_dict().items():
                state[f'wrapper.vision_projector.{k}'] = v
            for k, v in self.text_projector.state_dict().items():
                state[f'wrapper.text_projector.{k}'] = v
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        wrapper_keys = {k: v for k, v in state_dict.items() if k.startswith('wrapper.')}
        base_keys = {k: v for k, v in state_dict.items() if not k.startswith('wrapper.')}
        
        if base_keys:
            self.base_model.load_state_dict(base_keys, strict=False)
            print(f"✅ Loaded base model weights ({len(base_keys)} keys)")
        
        if self.enable_contrastive and wrapper_keys:
            vision_proj_state = {}
            text_proj_state = {}
            
            for k, v in wrapper_keys.items():
                if 'vision_projector' in k:
                    new_key = k.replace('wrapper.vision_projector.', '')
                    vision_proj_state[new_key] = v
                elif 'text_projector' in k:
                    new_key = k.replace('wrapper.text_projector.', '')
                    text_proj_state[new_key] = v
            
            if vision_proj_state:
                self.vision_projector.load_state_dict(vision_proj_state)
                print(f"✅ Loaded vision_projector weights")
            
            if text_proj_state:
                self.text_projector.load_state_dict(text_proj_state)
                print(f"✅ Loaded text_projector weights")
        
        return ([], [])
    
    def save_pretrained(self, output_dir, **kwargs):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.base_model.save_pretrained(output_dir, **kwargs)
        
        if self.enable_contrastive:
            torch.save({
                'vision_projector': self.vision_projector.state_dict(),
                'text_projector': self.text_projector.state_dict(),
            }, os.path.join(output_dir, 'contrastive_projectors.bin'))
            print(f"✅ Saved contrastive projectors to {output_dir}/contrastive_projectors.bin")
    
    def named_parameters(self, *args, **kwargs):
        for name, param in self.base_model.named_parameters(*args, **kwargs):
            yield name, param
        
        if self.enable_contrastive:
            for name, param in self.vision_projector.named_parameters(*args, **kwargs):
                yield f'wrapper.vision_projector.{name}', param
            for name, param in self.text_projector.named_parameters(*args, **kwargs):
                yield f'wrapper.text_projector.{name}', param
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
