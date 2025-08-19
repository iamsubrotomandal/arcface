import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import logging
import os
from collections import deque
from models.iresnet import SELayer  # Import SELayer from iresnet.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Defaults to 3.
        stride (int): Convolution stride. Defaults to 1.
        padding (int): Convolution padding. Defaults to 1.

    Raises:
        ValueError: If parameters are invalid.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size <= 0 or stride <= 0 or padding < 0:
            raise ValueError("kernel_size, stride must be positive, padding must be non-negative")
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TemporalDifferenceBlock(nn.Module):
    """Temporal Difference Block with enhanced feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_se (bool): If True, include SE layer. Defaults to True.
    """
    def __init__(self, in_channels: int, out_channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.pool(x)
        return x

class SpatialAttention(nn.Module):
    """Enhanced spatial attention module.

    Args:
        in_channels (int): Number of input channels.

    Raises:
        ValueError: If in_channels is invalid.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class TemporalDifferenceModule(nn.Module):
    """Learnable temporal difference module.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.

    Raises:
        ValueError: If in_channels is invalid.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        if in_channels != 3:
            raise ValueError("in_channels must be 3 for RGB frames")
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, current_frame: torch.Tensor, previous_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute learnable temporal difference.

        Args:
            current_frame (torch.Tensor): Current frame tensor (batch_size, 3, height, width).
            previous_frame (Optional[torch.Tensor]): Previous frame tensor or None.

        Returns:
            torch.Tensor: Temporal features (batch_size, 3, height, width).

        Raises:
            ValueError: If frame shapes are invalid or mismatched.
        """
        if current_frame.dim() != 4 or current_frame.shape[1] != 3:
            raise ValueError("current_frame must have shape (batch_size, 3, height, width)")
        if previous_frame is not None and previous_frame.shape != current_frame.shape:
            raise ValueError("previous_frame must match current_frame shape")
        
        previous_frame = previous_frame if previous_frame is not None else current_frame
        stacked = torch.cat([current_frame, previous_frame], dim=1)
        return self.temporal_conv(stacked)

class FAS_TD(nn.Module):
    """Face Anti-Spoofing with Temporal Difference Network.

    Processes pairs of RGB frames to detect spoofing, producing classification logits and a spoof confidence score.
    Integrates with StillImageFacePipeline for video-based liveness detection.
    Expects inputs in RGB format, normalized to [0, 1], with size 112x112 (configurable).

    Args:
        input_channels (int): Number of input channels (3 for RGB). Defaults to 3.
        num_classes (int): Number of classes (2 for live/spoof). Defaults to 2.
        buffer_size (int): Number of frames to store in buffer. Defaults to 5.
        input_size (Tuple[int, int]): Input frame size (height, width). Defaults to (112, 112).
        use_attention (bool): If True, use spatial and channel attention. Defaults to True.
        use_se (bool): If True, use SE layers in backbone. Defaults to True.
        dropout_rate (float): Dropout probability. Defaults to 0.5.
        device (str): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Raises:
        ValueError: If parameters are invalid.
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 2, 
                 buffer_size: int = 5, input_size: Tuple[int, int] = (112, 112),
                 use_attention: bool = True, use_se: bool = True, 
                 dropout_rate: float = 0.5, device: str = 'cpu'):
        super().__init__()
        if input_channels != 3:
            raise ValueError("input_channels must be 3 for RGB frames")
        if num_classes != 2:
            raise ValueError("num_classes must be 2 for live/spoof classification")
        if buffer_size < 2:
            raise ValueError("buffer_size must be at least 2")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be in [0, 1]")
        if not (isinstance(input_size, tuple) and len(input_size) == 2 and all(isinstance(n, int) and n > 0 for n in input_size)):
            raise ValueError("input_size must be a tuple of two positive integers")
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.input_size = input_size
        self.device = torch.device(device)
        self.buffer_resolution = (112, 112)  # Fixed to match pipeline
        
        self.frame_buffer = deque(maxlen=buffer_size)
        
        self.temporal_module = TemporalDifferenceModule(input_channels)
        
        self.input_processor = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.backbone = nn.Sequential(
            TemporalDifferenceBlock(32, 64, use_se=use_se),
            TemporalDifferenceBlock(64, 128, use_se=use_se),
            TemporalDifferenceBlock(128, 256, use_se=use_se),
            TemporalDifferenceBlock(256, 512, use_se=use_se),
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.spatial_attention = SpatialAttention(512)
            self.channel_attention = SELayer(512)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_temporal_features(self, current_frame: torch.Tensor, 
                                previous_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute enhanced temporal features.

        Args:
            current_frame (torch.Tensor): Current frame tensor (batch_size, 3, input_size[0], input_size[1]).
            previous_frame (Optional[torch.Tensor]): Previous frame tensor or None.

        Returns:
            torch.Tensor: Temporal features (batch_size, 3, input_size[0], input_size[1]).
        """
        return self.temporal_module(current_frame, previous_frame)

    def forward(self, x: torch.Tensor, previous_frame: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with multiple outputs.

        Args:
            x (torch.Tensor): Current frame tensor (batch_size, 3, input_size[0], input_size[1]) in RGB, normalized to [0, 1].
            previous_frame (Optional[torch.Tensor]): Previous frame tensor or None.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'logits' (batch_size, num_classes),
                                    'spoof_confidence' (batch_size, 1), 'features' (batch_size, 512, h, w),
                                    and 'temporal_features' (batch_size, 3, input_size[0], input_size[1]).

        Raises:
            ValueError: If input tensor shapes are invalid.
        """
        if x.dim() != 4 or x.shape[1] != 3 or x.shape[2] != self.input_size[0] or x.shape[3] != self.input_size[1]:
            raise ValueError(f"Input must have shape (batch_size, 3, {self.input_size[0]}, {self.input_size[1]})")
        
        x = x.to(self.device)
        if previous_frame is not None:
            previous_frame = previous_frame.to(self.device)
        
        temporal_features = self.compute_temporal_features(x, previous_frame)
        combined_input = torch.cat([x, temporal_features], dim=1)
        features = self.input_processor(combined_input)
        backbone_features = self.backbone(features)
        
        if self.use_attention:
            backbone_features = self.spatial_attention(backbone_features)
            backbone_features = self.channel_attention(backbone_features)
        
        pooled = self.global_pool(backbone_features)
        pooled = pooled.view(pooled.size(0), -1)
        pooled = self.dropout(pooled)
        
        logits = self.classifier(pooled)
        spoof_confidence = self.regressor(pooled)
        
        return {
            'logits': logits,
            'spoof_confidence': spoof_confidence,
            'features': backbone_features,
            'temporal_features': temporal_features
        }

    def update_frame_buffer(self, frame: torch.Tensor):
        """Update frame buffer with downsampled frame.

        Args:
            frame (torch.Tensor): Frame tensor to store (batch_size, 3, height, width).

        Raises:
            ValueError: If frame shape is invalid.
        """
        if frame.dim() != 4 or frame.shape[1] != 3:
            raise ValueError("Frame must have shape (batch_size, 3, height, width)")
        
        with torch.no_grad():
            frame = frame.to(self.device)
            if frame.shape[2:] != self.buffer_resolution:
                frame = F.interpolate(frame, size=self.buffer_resolution, mode='bilinear', align_corners=False)
            self.frame_buffer.append(frame.detach().clone())

    def get_previous_frame(self) -> Optional[torch.Tensor]:
        """Get previous frame from buffer.

        Returns:
            Optional[torch.Tensor]: Previous frame tensor or None.
        """
        if len(self.frame_buffer) >= 2:
            return self.frame_buffer[-2]
        return None

    def predict(self, frame: torch.Tensor) -> Dict[str, Any]:
        """Comprehensive prediction method.

        Args:
            frame (torch.Tensor): Input frame tensor (batch_size, 3, input_size[0], input_size[1]).

        Returns:
            Dict[str, Any]: Dictionary with 'spoof_score' (batch_size,), 'probabilities' (batch_size, num_classes),
                            'prediction' (batch_size,), and 'is_live' (batch_size,).
        """
        if frame.dim() != 4 or frame.shape[1] != 3 or frame.shape[2] != self.input_size[0] or frame.shape[3] != self.input_size[1]:
            raise ValueError(f"Frame must have shape (batch_size, 3, {self.input_size[0]}, {self.input_size[1]})")
        
        self.eval()
        with torch.no_grad():
            previous_frame = self.get_previous_frame()
            outputs = self.forward(frame, previous_frame)
            
            probs = F.softmax(outputs['logits'], dim=1)
            spoof_scores = outputs['spoof_confidence'].squeeze(-1)
            predictions = torch.argmax(probs, dim=1)
            is_live = spoof_scores < 0.5
            
            self.update_frame_buffer(frame)
            
            return {
                'spoof_score': spoof_scores.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'prediction': predictions.cpu().numpy(),
                'is_live': is_live.cpu().numpy()
            }

    def reset_buffer(self):
        """Reset frame buffer."""
        self.frame_buffer.clear()

    def train_sequence(self, frames: List[torch.Tensor], labels: List[int]) -> torch.Tensor:
        """Train on a sequence of frames with labels.

        Args:
            frames (List[torch.Tensor]): List of frame tensors (1, 3, input_size[0], input_size[1]).
            labels (List[int]): List of labels (0 for live, 1 for spoof).

        Returns:
            torch.Tensor: Mean loss over the sequence.

        Raises:
            ValueError: If frames or labels are invalid.
        """
        if len(frames) != len(labels):
            raise ValueError("Number of frames must match number of labels")
        if not all(f.dim() == 4 and f.shape[1] == 3 and f.shape[2] == self.input_size[0] and f.shape[3] == self.input_size[1] for f in frames):
            raise ValueError(f"Frames must have shape (1, 3, {self.input_size[0]}, {self.input_size[1]})")
        if not all(l in [0, 1] for l in labels):
            raise ValueError("Labels must be 0 (live) or 1 (spoof)")
        
        self.train()
        losses = []
        self.reset_buffer()
        
        for frame, label in zip(frames, labels):
            previous_frame = self.get_previous_frame()
            outputs = self.forward(frame, previous_frame)
            
            cls_loss = F.cross_entropy(outputs['logits'], torch.tensor([label], device=self.device))
            reg_loss = F.binary_cross_entropy(outputs['spoof_confidence'].squeeze(-1), 
                                            torch.tensor([float(label)], device=self.device))
            loss = cls_loss + reg_loss
            losses.append(loss)
            
            self.update_frame_buffer(frame)
        
        return torch.stack(losses).mean()

    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_bn(self):
        """Freeze batch normalization layers for inference."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def load_weights(self, weight_path: Optional[str], strict: bool = True, verbose: bool = True) -> Tuple[bool, List[str], List[str]]:
        """Load pretrained weights with flexible key cleaning.

        Args:
            weight_path (Optional[str]): Path to the weight file.
            strict (bool): If True, enforce strict key matching. Defaults to True.
            verbose (bool): If True, log loading status. Defaults to True.

        Returns:
            Tuple[bool, List[str], List[str]]: Success status, missing keys, unexpected keys.
        """
        if not weight_path:
            if verbose:
                logger.debug("No weight path provided")
            return False, [], []
        
        if not os.path.isfile(weight_path):
            if verbose:
                logger.error(f"Weight file not found: {weight_path}")
            return False, [], []
        
        try:
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            cleaned_state = {}
            prefixes = ['module.', 'model.', 'features.']
            for key, value in state_dict.items():
                new_key = key
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                cleaned_state[new_key] = value
            
            missing_keys, unexpected_keys = self.load_state_dict(cleaned_state, strict=strict)
            if verbose:
                if missing_keys:
                    logger.debug(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    logger.debug(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                if not missing_keys and not unexpected_keys:
                    logger.debug("All weights loaded successfully")
            return True, missing_keys, unexpected_keys
        
        except Exception as e:
            if verbose:
                logger.error(f"Error loading weights: {e}")
            return False, [], []

def create_fas_td_model(model_type: str = "standard", pretrained: bool = False, 
                       device: str = 'cpu', **kwargs) -> FAS_TD:
    """Factory function to create FAS-TD models.

    Args:
        model_type (str): Model type ('standard', 'lightweight', 'enhanced'). Defaults to 'standard'.
        pretrained (bool): If True, attempt to load pretrained weights. Defaults to False.
        device (str): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        **kwargs: Additional arguments for FAS_TD constructor.

    Returns:
        FAS_TD: Configured FAS-TD model.

    Raises:
        ValueError: If model_type or device is invalid.
    """
    if model_type not in ['standard', 'lightweight', 'enhanced']:
        raise ValueError(f"Unknown model type: {model_type}. Available: ['standard', 'lightweight', 'enhanced']")
    if device not in ['cpu', 'cuda']:
        raise ValueError("device must be 'cpu' or 'cuda'")
    
    configs = {
        'standard': {'use_attention': True, 'use_se': True, 'dropout_rate': 0.5, 'buffer_size': 5},
        'lightweight': {'use_attention': False, 'use_se': False, 'dropout_rate': 0.3, 'buffer_size': 5},
        'enhanced': {'use_attention': True, 'use_se': True, 'dropout_rate': 0.4, 'buffer_size': 8}
    }
    
    config = configs[model_type]
    config.update(kwargs)
    config['device'] = device
    
    model = FAS_TD(**config)
    
    if pretrained:
        weight_path = kwargs.get('weight_path', None)
        success, missing, unexpected = model.load_weights(weight_path, verbose=True)
        if not success:
            logger.warning("Failed to load pretrained weights; using random initialization")
    
    return model

def validate_fas_td_model(model: FAS_TD, input_size: Tuple[int, int, int, int] = (1, 3, 112, 112), 
                         device: str = 'cpu') -> Dict[str, Any]:
    """Validate FAS-TD model with comprehensive checks.

    Args:
        model (FAS_TD): Model to validate.
        input_size (Tuple[int, int, int, int]): Input shape (batch, channels, height, width). Defaults to (1, 3, 112, 112).
        device (str): Device for validation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Dict[str, Any]: Validation results including model info, forward pass status, and errors.
    """
    model.eval()
    model = model.to(device)
    
    results = {
        'model_name': model.__class__.__name__,
        'num_parameters': model.get_num_parameters(),
        'input_size': model.input_size,
        'device': str(next(model.parameters()).device),
        'forward_pass': False,
        'logits_shape': None,
        'spoof_confidence_shape': None,
        'gradient_check': False,
        'error': None
    }
    
    try:
        dummy_input = torch.randn(input_size).to(device)
        dummy_prev = torch.randn(input_size).to(device)
        outputs = model(dummy_input, dummy_prev)
        results['forward_pass'] = True
        results['logits_shape'] = tuple(outputs['logits'].shape)
        results['spoof_confidence_shape'] = tuple(outputs['spoof_confidence'].shape)
        
        model.train()
        dummy_input.requires_grad = True
        outputs = model(dummy_input, dummy_prev)
        loss = outputs['logits'].sum() + outputs['spoof_confidence'].sum()
        loss.backward()
        results['gradient_check'] = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    except Exception as e:
        results['error'] = str(e)
    
    return results

if __name__ == "__main__":
    model = create_fas_td_model("standard", device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"FAS-TD model created with {model.get_num_parameters():,} parameters")
    
    current_frame = torch.randn(2, 3, 112, 112)
    previous_frame = torch.randn(2, 3, 112, 112)
    outputs = model(current_frame, previous_frame)
    logger.info(f"Logits shape: {outputs['logits'].shape}")
    logger.info(f"Spoof confidence shape: {outputs['spoof_confidence'].shape}")
    
    test_frame = torch.randn(1, 3, 112, 112)
    prediction = model.predict(test_frame)
    logger.info(f"Prediction: {prediction}")
    
    validation_results = validate_fas_td_model(model)
    logger.info(f"Validation results: {validation_results}")