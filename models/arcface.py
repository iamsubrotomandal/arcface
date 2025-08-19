import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .iresnet import iresnet100, load_arcface_weights
from .resnet import resnet100, load_resnet_weights
import os

class ArcFaceHead(nn.Module):
    """ArcFace head for additive angular margin loss in face recognition.

    This module computes the ArcFace logits by normalizing embeddings and weights,
    applying an angular margin to the target class during training, and scaling the output.
    It is designed to be used with a softmax-based loss function, such as cross-entropy,
    to ensure proper gradient flow for both target and non-target classes.

    Args:
        embedding_size (int): Size of the input embeddings.
        num_classes (int): Number of classes for classification.
        s (float, optional): Scaling factor for logits. Defaults to 64.0.
        m (float, optional): Margin for ArcFace loss. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Scaled logits (batch_size, num_classes) and normalized embeddings (batch_size, embedding_size).
    """
    def __init__(self, embedding_size: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        if embedding_size <= 0 or num_classes <= 0:
            raise ValueError("embedding_size and num_classes must be positive integers")
        if s <= 0:
            raise ValueError("Scale parameter s must be positive")
        if m < 0 or m > 1:
            raise ValueError("Margin parameter m should be in a reasonable range, e.g., [0, 1]")
        self.W = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)  # Better than uniform for cosine similarity
        self.s = s
        self.m = m
        self.eps = 1e-6  # Slightly larger for stability

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None, apply_margin: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ArcFace head.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, embedding_size).
            labels (torch.Tensor, optional): Class labels of shape (batch_size,). Defaults to None.
            apply_margin (bool, optional): Whether to apply the margin if labels are provided during training. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Scaled logits and normalized embeddings.
        """
        # Ensure device consistency - move weights to embeddings device if needed
        if self.W.device != embeddings.device:
            self.W.data = self.W.data.to(embeddings.device)
        
        # Normalize weights and embeddings
        W_norm = F.normalize(self.W, p=2, dim=1)
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine = torch.matmul(x_norm, W_norm.t())
        
        if labels is not None and self.training and apply_margin:
            # Validate labels
            if labels.max() >= self.W.size(0) or labels.min() < 0:
                raise ValueError("Labels must be in range [0, num_classes)")
            
            # Ensure numerical stability
            cosine = torch.clamp(cosine, -1 + self.eps, 1 - self.eps)
            
            # Calculate theta and apply margin
            theta = torch.acos(cosine)
            theta = torch.where(torch.isnan(theta), torch.zeros_like(theta), theta)  # Handle potential NaNs
            
            target_cosine = torch.cos(theta + self.m)
            
            # Apply margin only to the correct class using scatter for efficiency
            output = cosine.clone()
            output.scatter_(1, labels.unsqueeze(1), target_cosine.gather(1, labels.unsqueeze(1)))
        else:
            output = cosine
        
        # Scale logits
        output = output * self.s
        
        return output, x_norm

class ArcFaceRecognizer(nn.Module):
    """ArcFaceRecognizer for face recognition, integrating a backbone network with an optional ArcFace head.

    This module processes input face images through a backbone network (e.g., iresnet100 or resnet100)
    to produce embeddings, which can be used directly for inference or passed to an ArcFaceHead for
    training with the ArcFace loss. Supports loading pretrained weights, freezing the backbone, and
    device management for compatibility with downstream components.

    Args:
        backbone (str, optional): Name of the backbone network ('iresnet100' or 'resnet100'). Defaults to 'iresnet100'.
        embedding_size (int, optional): Size of the output embeddings. Defaults to 512.
        weight_path (str, optional): Path to pretrained weights. Defaults to None.
        freeze (bool, optional): Whether to freeze the backbone parameters. Defaults to False.
        device (str, optional): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        arcface_head (nn.Module, optional): Optional ArcFaceHead for training. Defaults to None.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Embeddings during inference, or (logits, normalized embeddings)
            during training if arcface_head is provided.
    """
    def __init__(self, backbone: str = "iresnet100", embedding_size: int = 512, 
                 weight_path: Optional[str] = None, freeze: bool = False, 
                 device: str = "cpu", arcface_head: Optional[nn.Module] = None):
        super().__init__()
        # Input validation
        if embedding_size <= 0:
            raise ValueError("embedding_size must be a positive integer")
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if weight_path is not None and not isinstance(weight_path, str):
            raise ValueError("weight_path must be a string if provided")

        self.backbone_name = backbone.lower()
        self.embedding_size = embedding_size
        self.device = torch.device(device)
        self.arcface_head = arcface_head
        self.weights_loaded = False

        # Backbone selection
        backbones = {
            'iresnet100': lambda: iresnet100(embedding_size=embedding_size),
            'resnet100': lambda: resnet100(embedding_size=embedding_size)
        }
        
        if self.backbone_name not in backbones:
            raise ValueError(f"Unsupported backbone {backbone}. Available: {list(backbones.keys())}")
        
        self.backbone = backbones[self.backbone_name]()
        self.backbone.to(self.device)
        
        # Load weights if provided
        if weight_path:
            self.weights_loaded = self.load_weights(weight_path)
        
        # Freeze backbone if requested
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze backbone parameters and set to eval mode.

        This method disables gradient computation for all backbone parameters and
        sets the backbone to evaluation mode. Idempotent and safe to call multiple times.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def load_weights(self, weight_path: str) -> bool:
        """Load pretrained weights with robust error handling.

        Args:
            weight_path (str): Path to the pretrained weights file.

        Returns:
            bool: True if weights are loaded successfully, False otherwise.
        """
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        try:
            state = torch.load(weight_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
        
        if 'state_dict' in state:
            state = state['state_dict']
        
        # Clean keys robustly
        cleaned_state = {}
        for key, value in state.items():
            new_key = key.replace('module.', '').replace('backbone.', '')
            cleaned_state[new_key] = value
        
        # Load with detailed error reporting
        try:
            missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state, strict=False)
            if missing_keys:
                print(f"Warning: {len(missing_keys)} missing keys in state dict")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys in state dict")
            return True
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return False

    @torch.inference_mode()
    def extract(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from face images in inference mode.

        Args:
            face_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_size).

        Raises:
            ValueError: If face_tensor has invalid shape or type.
        """
        if not isinstance(face_tensor, torch.Tensor):
            raise ValueError("face_tensor must be a torch.Tensor")
        if face_tensor.dim() != 4:
            raise ValueError("face_tensor must have shape (batch_size, channels, height, width)")
        
        was_training = self.backbone.training
        try:
            if was_training:
                self.backbone.eval()
            face_tensor = face_tensor.to(self.device)
            return self.backbone(face_tensor)
        finally:
            if was_training:
                self.backbone.train()

    def forward(self, face_tensor: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for ArcFaceRecognizer.

        During training, if arcface_head is provided, returns logits and normalized embeddings.
        During inference or if no arcface_head, returns embeddings only.

        Args:
            face_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            labels (torch.Tensor, optional): Class labels of shape (batch_size,). Defaults to None.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Embeddings or (logits, normalized embeddings).

        Raises:
            ValueError: If face_tensor has invalid shape or type, or if labels are invalid when arcface_head is used.
        """
        if not isinstance(face_tensor, torch.Tensor):
            raise ValueError("face_tensor must be a torch.Tensor")
        if face_tensor.dim() != 4:
            raise ValueError("face_tensor must have shape (batch_size, channels, height, width)")
        
        face_tensor = face_tensor.to(self.device)
        embeddings = self.backbone(face_tensor)
        
        if self.arcface_head is not None and self.training:
            if labels is not None and not isinstance(labels, torch.Tensor):
                raise ValueError("labels must be a torch.Tensor if provided")
            return self.arcface_head(embeddings, labels=labels)
        return embeddings

    def create_head(self, num_classes: int, s: float = 64.0, m: float = 0.5) -> ArcFaceHead:
        """Create an ArcFace head for training with the current embedding size.
        
        Args:
            num_classes (int): Number of classes for classification
            s (float, optional): Scaling factor for logits. Defaults to 64.0.
            m (float, optional): Margin for ArcFace loss. Defaults to 0.5.
            
        Returns:
            ArcFaceHead: Configured ArcFace head for training
        """
        return ArcFaceHead(self.embedding_size, num_classes, s, m)
    
    def forward_with_head(self, face_tensor: torch.Tensor, head: ArcFaceHead, 
                         labels: Optional[torch.Tensor] = None, apply_margin: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through backbone and head for training.
        
        Args:
            face_tensor (torch.Tensor): Input face tensor
            head (ArcFaceHead): ArcFace head for classification
            labels (torch.Tensor, optional): Class labels for training. Defaults to None.
            apply_margin (bool, optional): Whether to apply margin during training. Defaults to True.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits and normalized embeddings
        """
        embeddings = self.backbone(face_tensor)
        return head(embeddings, labels, apply_margin)
