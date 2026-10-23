import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer

class GeometricContrastiveLoss(nn.Module):
    """Geometric contrastive loss using geodesic distance on hypersphere"""
    
    def __init__(self, temperature=0.07, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.phi = math.pi 
    
    def geodesic_distance(self, x1, x2):
        """Compute geodesic distance on unit hypersphere using arccos"""
        # Ensure vectors are L2 normalized
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        
        # Compute cosine similarity
        cos_sim = torch.sum(x1 * x2, dim=-1)
        
        # Clamp to avoid numerical issues with arccos
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Geodesic distance is the angle between vectors (arccos of dot product)
        geodesic_dist = torch.acos(cos_sim)
        return geodesic_dist
    
    def forward(self, image_embeddings, text_embeddings, positive_pairs=None):
        """
        Args:
            image_embeddings: [batch_size, embed_dim]
            text_embeddings: [batch_size, embed_dim] 
            positive_pairs: Optional tensor indicating positive pairs
        """
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings to unit hypersphere
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute pairwise geodesic distances
        image_expanded = image_embeddings.unsqueeze(1)  # [batch, 1, dim]
        text_expanded = text_embeddings.unsqueeze(0)    # [1, batch, dim]
        
        # Compute all pairwise geodesic distances
        distances = self.geodesic_distance(
            image_expanded.expand(batch_size, batch_size, -1),
            text_expanded.expand(batch_size, batch_size, -1)
        )  # [batch_size, batch_size]
        
        # Convert distances to similarities (smaller distance = higher similarity)
        # Use negative distance in exponential to make smaller distances have higher similarity
        # print("##############################################CONTRASTIVE DISTANCES##############################################")
        # print(1.0 - (distances / self.phi))
        similarities = torch.exp((1.0 - (distances / self.phi)) / self.temperature)
        
        # Create positive pairs mask (diagonal elements by default)
        if positive_pairs is None:
            positive_mask = torch.eye(batch_size, device=image_embeddings.device)
            # print("Using default positive pairs (diagonal)")
            
        else:
            positive_mask = positive_pairs
        
        negative_mask = 1 - positive_mask
        
        # Compute contrastive loss
        positive_similarities = similarities * positive_mask
        # print(positive_mask)
        negative_similarities = similarities * negative_mask
        
        # InfoNCE-style loss with geodesic distances
        # Image-to-text loss
        i2t_numerator = positive_similarities.sum(dim=1)
        i2t_denominator = similarities.sum(dim=1)
        i2t_loss = -torch.log(i2t_numerator / (i2t_denominator + 1e-8)).mean()
        
        # Text-to-image loss
        t2i_numerator = positive_similarities.sum(dim=0)
        t2i_denominator = similarities.sum(dim=0)
        t2i_loss = -torch.log(t2i_numerator / (t2i_denominator + 1e-8)).mean()
        
        return (i2t_loss + t2i_loss) / 2

class VisionProjectorMLP(nn.Module):
    """MLP to project vision tokens to contrastive embedding space"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: vision tokens [batch_size, num_tokens, input_dim]
        Returns:
            embeddings: [batch_size, output_dim] - pooled and projected embeddings
        """
        # Pool over spatial/token dimensions (mean pooling)
        if x.dim() == 3:  # [batch, tokens, dim]
            x = x.mean(dim=1)  # [batch, dim]
        
        # Project through MLP
        embeddings = self.mlp(x)
        
        # L2 normalize to unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings

class TextProjectorMLP(nn.Module):
    """MLP to project text embeddings to contrastive embedding space"""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: text embeddings [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        Returns:
            embeddings: [batch_size, output_dim] - pooled and projected embeddings
        """
        # Pool over sequence length if needed
        if x.dim() == 3:  # [batch, seq_len, dim]
            # Use mean pooling over sequence dimension
            x = x.mean(dim=1)  # [batch, dim]
        
        # Project through MLP
        embeddings = self.mlp(x)
        
        # L2 normalize to unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings

class DescriptionExtractor:
    """Extract generated descriptions from model outputs"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def extract_descriptions_from_output(self, generated_text):
        """
        Extract descriptions from generated text
        Expected format: JSON with desc_IMAGE_, desc_IMAGE_2, etc.
        """
        import json
        import re
        
        try:
            # Try to parse as JSON first
            if generated_text.strip().startswith('{'):
                descriptions = json.loads(generated_text)
                return descriptions
            
            # Fallback: extract using regex patterns
            descriptions = {}
            patterns = [
                r'"desc_IMAGE_(\d+)":\s*"([^"]+)"',
                r'desc_IMAGE_(\d+):\s*([^\n]+)',
                r'IMAGE (\d+):\s*([^\n]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, generated_text, re.IGNORECASE)
                for match in matches:
                    img_num, desc = match
                    descriptions[f"desc_IMAGE_{img_num}"] = desc.strip()
            
            return descriptions
            
        except Exception as e:
            print(f"Error extracting descriptions: {e}")
            return {}
    
    def preprocess_descriptions(self, descriptions_dict):
        """
        Convert descriptions dict to ordered list
        """
        ordered_descriptions = []
        for i in range(1, 6):  # Assuming max 5 images
            key = f"desc_IMAGE_{i}"
            if key in descriptions_dict:
                ordered_descriptions.append(descriptions_dict[key])
        
        return ordered_descriptions