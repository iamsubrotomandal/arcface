import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union, Dict, Any
import math
import warnings
import os

class CDCConv(nn.Module):
    """Central Difference Convolution module for face anti-spoofing.

    Combines regular convolution with central difference convolution for enhanced feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (int, optional): Convolution padding. Defaults to 1.
        dilation (int, optional): Convolution dilation. Defaults to 1.
        groups (int, optional): Convolution groups. Defaults to 1.
        theta (float, optional): Weight for central difference component. Defaults to 0.7.
        learnable_theta (bool, optional): If True, theta is learnable. Defaults to True.
        bias (bool, optional): If True, include bias in convolution. Defaults to True.

    Raises:
        ValueError: If parameters are invalid.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1, 
                 groups: int = 1, theta: float = 0.7, learnable_theta: bool = True, 
                 bias: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.theta = nn.Parameter(torch.tensor(float(theta))) if learnable_theta else float(theta)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, groups, bias=bias)
        self.diff_kernel = self._get_central_diff_kernel()

    def _get_central_diff_kernel(self) -> torch.Tensor:
        """Generate central difference kernel for 4-connected neighbors."""
        kernel_center = self.kernel_size // 2
        diff_kernel = torch.zeros(self.kernel_size, self.kernel_size)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == kernel_center and j == kernel_center:
                    continue
                if abs(i - kernel_center) + abs(j - kernel_center) == 1:  # 4-connected
                    diff_kernel[i, j] = -0.25
                    diff_kernel[kernel_center, kernel_center] += 0.25
        return diff_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining regular and central difference convolutions."""
        out_normal = self.conv(x)
        if isinstance(self.theta, float) and self.theta == 0:
            return out_normal
        
        # Apply central difference convolution to input
        diff_kernel = self.diff_kernel.to(x.device).view(1, 1, self.kernel_size, self.kernel_size)
        diff_kernel = diff_kernel.repeat(x.shape[1], 1, 1, 1)
        
        # Apply difference convolution to input with same padding as regular conv
        out_diff_input = F.conv2d(x, diff_kernel, bias=None, stride=self.stride, 
                                 padding=self.padding, dilation=self.dilation, groups=x.shape[1])
        
        # Apply regular convolution to the difference result to match output channels
        # Use same padding as the original convolution to maintain spatial dimensions
        out_diff = F.conv2d(out_diff_input, self.conv.weight, bias=self.conv.bias,
                           stride=1, padding=self.conv.padding[0], groups=self.groups)
        
        theta = self.theta if isinstance(self.theta, torch.Tensor) else torch.tensor(self.theta, device=x.device)
        return out_normal - theta * out_diff

class CDCNBlock(nn.Module):
    """CDCN block with optional residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (int, optional): Convolution padding. Defaults to 1.
        use_residual (bool, optional): If True, include residual connection. Defaults to False.
        theta (float, optional): Weight for central difference component. Defaults to 0.7.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_residual: bool = False,
                 theta: float = 0.7):
        super().__init__()
        self.use_residual = use_residual
        self.cdc_conv = CDCConv(in_channels, out_channels, kernel_size, stride, 
                               padding, theta=theta, learnable_theta=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if use_residual and in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        elif use_residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.cdc_conv(x)
        out = self.bn(out)
        if self.residual is not None:
            identity = self.residual(x)
            out += identity
        out = self.relu(out)
        return out

class CDCN(nn.Module):
    """Central Difference Convolutional Network for face anti-spoofing.

    Produces classification logits (real vs. spoof) and depth maps for detected faces.
    Expects input images of shape (batch_size, 3, 112, 112) in RGB format from
    RetinaFaceDetector.preprocess_faces.

    Based on: "Central Difference Convolutional Networks for Face Anti-Spoofing"

    Args:
        num_classes (int, optional): Number of classes (e.g., 2 for real/spoof). Defaults to 2.
        input_size (Tuple[int, int], optional): Input image size (height, width). Defaults to (112, 112).
        theta (float, optional): Weight for central difference component. Defaults to 0.7.
        use_residual (bool, optional): If True, use residual connections. Defaults to True.
        return_features (bool, optional): If True, return features alongside outputs. Defaults to False.
        device (str, optional): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Raises:
        ValueError: If parameters are invalid.
    """
    def __init__(self, num_classes: int = 2, input_size: Tuple[int, int] = (112, 112),
                 theta: float = 0.7, use_residual: bool = True, return_features: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if not (isinstance(input_size, tuple) and len(input_size) == 2 and all(isinstance(n, int) and n > 0 for n in input_size)):
            raise ValueError("input_size must be a tuple of two positive integers")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1]")
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.return_features = return_features
        self.device = torch.device(device)
        
        self.features = nn.Sequential(
            self._make_layer(3, 64, 2, theta=theta, use_residual=use_residual),  # 112x112
            nn.MaxPool2d(2, 2),  # 56x56
            self._make_layer(64, 128, 2, theta=theta, use_residual=use_residual),  # 56x56
            nn.MaxPool2d(2, 2),  # 28x28
            self._make_layer(128, 256, 3, theta=theta, use_residual=use_residual),  # 28x28
            nn.MaxPool2d(2, 2),  # 14x14
            self._make_layer(256, 512, 3, theta=theta, use_residual=use_residual),  # 14x14
            nn.MaxPool2d(2, 2),  # 7x7
        )
        
        self.depth_branch = nn.Sequential(
            CDCConv(512, 256, 3, 1, 1, theta=theta, learnable_theta=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CDCConv(256, 128, 3, 1, 1, theta=theta, learnable_theta=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Conv2d(513, 512, 1, 1, 0)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
        self.to(self.device)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int,
                    theta: float = 0.7, use_residual: bool = True) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            stride = 1
            layers.append(
                CDCNBlock(in_channels if i == 0 else out_channels, 
                         out_channels, stride=stride, 
                         use_residual=use_residual, theta=theta)
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights for better convergence."""
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

    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass for anti-spoofing, returning logits and depth map.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, input_size[0], input_size[1]) in RGB format.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Classification logits (batch_size, num_classes), upsampled depth map (batch_size, 1, input_size[0], input_size[1]),
                and optionally features (batch_size, 512, 7, 7) if return_features is True.

        Raises:
            ValueError: If input tensor has invalid shape.
        """
        if x.dim() != 4 or x.shape[1] != 3 or x.shape[2] != self.input_size[0] or x.shape[3] != self.input_size[1]:
            raise ValueError(f"Input must be of shape (batch_size, 3, {self.input_size[0]}, {self.input_size[1]})")
        
        x = x.to(self.device)
        features = self.features(x)  # (batch_size, 512, 7, 7)
        depth_map = self.depth_branch(features)  # (batch_size, 1, 7, 7)
        depth_map_upsampled = F.interpolate(depth_map, size=self.input_size, 
                                          mode='nearest', align_corners=None)
        
        fused_features = torch.cat([features, depth_map], dim=1)  # (batch_size, 513, 7, 7)
        fused_features = self.fusion_conv(fused_features)  # (batch_size, 512, 7, 7)
        cls_logits = self.classifier(fused_features)  # (batch_size, num_classes)
        
        if self.return_features:
            return cls_logits, depth_map_upsampled, features
        return cls_logits, depth_map_upsampled

    def get_liveness_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get liveness score (probability of being real).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, input_size[0], input_size[1]).

        Returns:
            torch.Tensor: Probability of real class (batch_size,).
        """
        result = self.forward(x)
        cls_logits = result[0]  # First element is always classification logits
        return torch.softmax(cls_logits, dim=1)[:, 1]

    def get_depth_map(self, x: torch.Tensor) -> torch.Tensor:
        """Get depth map for input image.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, input_size[0], input_size[1]).

        Returns:
            torch.Tensor: Depth map (batch_size, 1, input_size[0], input_size[1]).
        """
        result = self.forward(x)
        depth_map = result[1]  # Second element is always depth map
        return depth_map

    def freeze_bn(self):
        """Freeze batch normalization layers for inference."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

def create_cdcn(model_type: str = 'base', **kwargs) -> CDCN:
    """Create CDCN variant by type.

    Args:
        model_type (str, optional): Model variant ('base', 'large', 'small'). Defaults to 'base'.
        **kwargs: Additional arguments to pass to CDCN constructor.

    Returns:
        CDCN: Configured CDCN model.

    Raises:
        ValueError: If model_type is invalid.
    """
    configs = {
        'base': {'theta': 0.7, 'use_residual': True, 'input_size': (112, 112)},
        'large': {'theta': 0.5, 'use_residual': True, 'input_size': (112, 112)},
        'small': {'theta': 0.9, 'use_residual': False, 'input_size': (112, 112)},
    }
    if model_type not in configs:
        raise ValueError(f"Unknown CDCN type: {model_type}. Available: {list(configs.keys())}")
    
    config = configs[model_type]
    config.update(kwargs)
    return CDCN(**config)

def load_cdcn_weights(model: nn.Module, weight_path: str, strict: bool = True, 
                      verbose: bool = True) -> Tuple[bool, List[str], List[str]]:
    """Load pretrained CDCN weights with error handling.

    Args:
        model (nn.Module): CDCN model to load weights into.
        weight_path (str): Path to the weight file.
        strict (bool, optional): If True, enforce strict key matching. Defaults to True.
        verbose (bool, optional): If True, print loading status. Defaults to True.

    Returns:
        Tuple[bool, List[str], List[str]]: Success status, missing keys, unexpected keys.
    """
    if not os.path.isfile(weight_path):
        if verbose:
            warnings.warn(f"Weight file not found: {weight_path}")
        return False, [], []
    
    try:
        state_dict = torch.load(weight_path, map_location=model.device if hasattr(model, 'device') else 'cpu', 
                               weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        cleaned_state = {}
        prefixes = ['module.', 'backbone.', 'model.']
        for k, v in state_dict.items():
            new_key = k
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            cleaned_state[new_key] = v
        
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state, strict=strict)
        if verbose:
            if missing_keys:
                warnings.warn(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                warnings.warn(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            if not missing_keys and not unexpected_keys:
                print("✅ All weights loaded successfully!")
        return len(missing_keys) == 0 and len(unexpected_keys) == 0, missing_keys, unexpected_keys
    except Exception as e:
        if verbose:
            warnings.warn(f"Error loading weights: {e}")
        return False, [], []

def validate_cdcn_model(model: CDCN, input_size: Tuple[int, int, int, int] = (1, 3, 112, 112), 
                        device: str = 'cpu') -> Dict[str, Any]:
    """Validate CDCN model with comprehensive checks.

    Args:
        model (CDCN): Model to validate.
        input_size (Tuple[int, int, int, int], optional): Input shape (batch, channels, height, width). Defaults to (1, 3, 112, 112).
        device (str, optional): Device for validation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Dict[str, Any]: Validation results including model info, forward pass status, and errors.
    """
    model.eval()
    model = model.to(device)
    
    results = {
        'model_name': model.__class__.__name__,
        'num_classes': model.num_classes,
        'input_size': model.input_size,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': str(next(model.parameters()).device),
        'forward_pass': False,
        'logits_shape': None,
        'depth_map_shape': None,
        'liveness_score': False,
        'feature_extraction': False,
        'gradient_check': False
    }
    
    try:
        dummy_input = torch.randn(input_size).to(device)
        with torch.no_grad():
            logits, depth_map = model(dummy_input)
            results['forward_pass'] = True
            results['logits_shape'] = tuple(logits.shape)
            results['depth_map_shape'] = tuple(depth_map.shape)
        
        with torch.no_grad():
            score = model.get_liveness_score(dummy_input)
            results['liveness_score'] = True
            results['score_shape'] = tuple(score.shape)
        
        if model.return_features:
            logits, depth_map, features = model(dummy_input)
            results['feature_extraction'] = True
            results['feature_shape'] = tuple(features.shape)
        
        model.train()
        dummy_input.requires_grad = True
        logits, depth_map = model(dummy_input)
        loss = logits.sum() + depth_map.sum()
        loss.backward()
        results['gradient_check'] = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_cdcn('base', num_classes=2, device=device)
    x = torch.randn(2, 3, 112, 112).to(device)
    cls_logits, depth_map = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Classification logits shape: {cls_logits.shape}")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    validation_results = validate_cdcn_model(model, device=device)
    print("Validation results:", validation_results)