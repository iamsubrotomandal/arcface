"""
FAS-TD: Face Anti-Spoofing with Temporal Difference
Implements temporal difference analysis for video-based anti-spoofing detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class TemporalDifferenceBlock(nn.Module):
    """
    Temporal Difference Block for analyzing frame-to-frame changes
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class SpatialAttention(nn.Module):
    """
    Spatial attention module for focusing on discriminative regions
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class FAS_TD(nn.Module):
    """
    Face Anti-Spoofing with Temporal Difference Network
    
    Analyzes temporal differences between consecutive frames to detect spoofing attacks.
    Works by computing frame differences and analyzing motion patterns.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(input_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Temporal difference blocks
            TemporalDifferenceBlock(32, 64),
            TemporalDifferenceBlock(64, 128),
            TemporalDifferenceBlock(128, 256),
            TemporalDifferenceBlock(256, 512),
        )
        
        # Spatial attention
        self.attention = SpatialAttention(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Temporal buffer for frame differences
        self.frame_buffer = []
        self.max_buffer_size = 5
        
    def compute_temporal_difference(self, current_frame: torch.Tensor, 
                                  previous_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal difference between consecutive frames
        
        Args:
            current_frame: Current frame tensor [B, C, H, W]
            previous_frame: Previous frame tensor [B, C, H, W]
            
        Returns:
            Temporal difference tensor
        """
        if previous_frame is None:
            # First frame, return zeros
            return torch.zeros_like(current_frame)
        
        # Compute absolute difference
        diff = torch.abs(current_frame - previous_frame)
        
        # Apply Gaussian smoothing to reduce noise
        diff = F.conv2d(diff, self._get_gaussian_kernel(current_frame.device), 
                        padding=1, groups=current_frame.size(1))
        
        return diff
    
    def _get_gaussian_kernel(self, device: torch.device) -> torch.Tensor:
        """Generate 3x3 Gaussian kernel for smoothing"""
        kernel = torch.tensor([[[
            [1, 2, 1],
            [2, 4, 2], 
            [1, 2, 1]
        ]]], dtype=torch.float32, device=device) / 16.0
        return kernel.repeat(3, 1, 1, 1)  # Repeat for RGB channels
    
    def forward(self, x: torch.Tensor, previous_frame: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal difference analysis
        
        Args:
            x: Current frame tensor [B, C, H, W]
            previous_frame: Previous frame tensor [B, C, H, W]
            
        Returns:
            Tuple of (spoof_logits, temporal_features)
        """
        # Compute temporal difference
        temp_diff = self.compute_temporal_difference(x, previous_frame)
        
        # Concatenate original frame with temporal difference
        if temp_diff.sum() > 0:  # If we have meaningful temporal difference
            x_input = torch.cat([x, temp_diff], dim=1)
            # Adjust input channels for concatenated input
            backbone_first_layer = self.backbone[0]
            if isinstance(backbone_first_layer, nn.Conv2d):
                expected_channels = backbone_first_layer.in_channels
                if x_input.size(1) != expected_channels:
                    # Use a 1x1 conv to adjust channels
                    if not hasattr(self, 'channel_adapter'):
                        self.channel_adapter = nn.Conv2d(x_input.size(1), expected_channels, 1).to(x.device)
                    x_input = self.channel_adapter(x_input)
        else:
            x_input = x
        
        # Feature extraction
        features = self.backbone(x_input)
        
        # Apply spatial attention
        features = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, features
    
    def update_frame_buffer(self, frame: torch.Tensor):
        """Update the frame buffer for temporal analysis"""
        self.frame_buffer.append(frame.clone())
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_previous_frame(self) -> Optional[torch.Tensor]:
        """Get the previous frame from buffer"""
        if len(self.frame_buffer) >= 2:
            return self.frame_buffer[-2]
        return None
    
    def predict_spoof_score(self, frame: torch.Tensor) -> float:
        """
        Predict anti-spoofing score for a single frame
        
        Args:
            frame: Input frame tensor [B, C, H, W]
            
        Returns:
            Spoof score (0-1, higher means more likely to be spoofed)
        """
        self.eval()
        with torch.no_grad():
            previous_frame = self.get_previous_frame()
            logits, _ = self.forward(frame, previous_frame)
            
            # Convert logits to probability (assuming class 1 is spoof)
            probs = F.softmax(logits, dim=1)
            spoof_score = probs[:, 1].item() if probs.size(0) == 1 else probs[:, 1].mean().item()
            
            # Update buffer for next frame
            self.update_frame_buffer(frame)
            
            return spoof_score
    
    def reset_buffer(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()

def create_fas_td_model(pretrained: bool = False) -> FAS_TD:
    """
    Create FAS-TD model
    
    Args:
        pretrained: Whether to load pretrained weights
        
    Returns:
        FAS_TD model instance
    """
    model = FAS_TD(input_channels=3, num_classes=2)
    
    if pretrained:
        try:
            # Load pretrained weights if available
            checkpoint_path = "weights/fas_td_pretrained.pth"
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded FAS-TD pretrained weights from {checkpoint_path}")
        except FileNotFoundError:
            print("FAS-TD pretrained weights not found, using random initialization")
    
    return model

if __name__ == "__main__":
    # Test the model
    model = create_fas_td_model()
    print(f"FAS-TD model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    current_frame = torch.randn(batch_size, 3, 224, 224)
    previous_frame = torch.randn(batch_size, 3, 224, 224)
    
    logits, features = model(current_frame, previous_frame)
    print(f"Output logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    
    # Test spoof score prediction
    single_frame = torch.randn(1, 3, 224, 224)
    spoof_score = model.predict_spoof_score(single_frame)
    print(f"Spoof score: {spoof_score:.4f}")
