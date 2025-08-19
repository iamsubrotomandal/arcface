import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IRBlock(nn.Module):
    """Improved Residual Block with pre-activation and optional Squeeze-and-Excitation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for convolution. Defaults to 1.
        downsample (Optional[nn.Module]): Downsample layer for residual connection.
        groups (int): Number of convolution groups. Defaults to 1.
        base_width (int): Base width for channels. Defaults to 64.
        dilation (int): Dilation factor. Defaults to 1.
        use_se (bool): If True, include SE layer. Defaults to False.

    Raises:
        ValueError: If groups or base_width are invalid.
        NotImplementedError: If dilation > 1.
    """
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, use_se: bool = False):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("IRBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IRBlock")
        
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.prelu1 = nn.PReLU(in_channels)
        self.prelu2 = nn.PReLU(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SELayer(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.prelu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention.

    Args:
        channel (int): Number of input channels.
        reduction (int): Reduction ratio for SE layer. Defaults to 16.

    Raises:
        ValueError: If reduction < 2.
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        if reduction < 2:
            raise ValueError("reduction must be at least 2")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class IResNet(nn.Module):
    """IR-ResNet for ArcFace embeddings (default input 3x112x112).

    Produces normalized embeddings for face recognition, used as the backbone in ArcFaceRecognizer.
    Expects inputs in RGB format, normalized to [0, 1].

    Based on: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

    Args:
        layers (List[int]): Number of blocks in each layer (e.g., [3, 4, 14, 3] for IResNet-50).
        embedding_size (int): Size of output embeddings. Defaults to 512.
        input_size (Tuple[int, int]): Input image size (height, width). Defaults to (112, 112).
        zero_init_residual (bool): If True, zero-initialize residual connections. Defaults to False.
        groups (int): Number of convolution groups. Defaults to 1.
        width_per_group (int): Base width for channels. Defaults to 64.
        use_se (bool): If True, include SE layers. Defaults to False.
        dropout_prob (float): Dropout probability. Defaults to 0.4.
        device (str): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Raises:
        ValueError: If parameters are invalid.
    """
    def __init__(self, layers: List[int], embedding_size: int = 512, 
                 input_size: Tuple[int, int] = (112, 112), zero_init_residual: bool = False, 
                 groups: int = 1, width_per_group: int = 64, use_se: bool = False,
                 dropout_prob: float = 0.4, device: str = 'cpu'):
        super().__init__()
        if not (isinstance(layers, list) and len(layers) == 4 and all(isinstance(n, int) and n > 0 for n in layers)):
            raise ValueError("layers must be a list of four positive integers")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if not 0 <= dropout_prob <= 1:
            raise ValueError("dropout_prob must be in [0, 1]")
        if not (isinstance(input_size, tuple) and len(input_size) == 2 and all(isinstance(n, int) and n > 0 for n in input_size)):
            raise ValueError("input_size must be a tuple of two positive integers")
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.use_se = use_se
        self.input_size = input_size
        self.device = torch.device(device)
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes, eps=1e-5),
            nn.PReLU(self.inplanes)
        )
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.bn = nn.BatchNorm2d(512 * IRBlock.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.flatten = nn.Flatten()
        output_size = input_size[0] // 16  # Three stride=2 layers
        self.fc = nn.Linear(512 * IRBlock.expansion * output_size * output_size, embedding_size, bias=False)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        
        self._init_weights(zero_init_residual)
        self.to(self.device)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * IRBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * IRBlock.expansion, stride),
                nn.BatchNorm2d(planes * IRBlock.expansion, eps=1e-5),
            )

        layers = []
        layers.append(IRBlock(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, self.dilation, self.use_se
        ))
        self.inplanes = planes * IRBlock.expansion
        for _ in range(1, blocks):
            layers.append(IRBlock(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                use_se=self.use_se
            ))

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IRBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    if m.downsample is not None:
                        for sm in m.downsample:
                            if isinstance(sm, nn.Conv2d):
                                nn.init.constant_(sm.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to produce normalized embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, input_size[0], input_size[1]) in RGB format, normalized to [0, 1].

        Returns:
            torch.Tensor: Normalized embeddings of shape (batch_size, embedding_size).

        Raises:
            ValueError: If input tensor has invalid shape.
        """
        if x.dim() != 4 or x.shape[1] != 3 or x.shape[2] != self.input_size[0] or x.shape[3] != self.input_size[1]:
            raise ValueError(f"Input must be of shape (batch_size, 3, {self.input_size[0]}, {self.input_size[1]})")
        
        x = x.to(self.device)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn(x)
        x = F.adaptive_avg_pool2d(x, (self.input_size[0] // 16, self.input_size[0] // 16))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def get_embedding_size(self) -> int:
        """Get the output embedding size."""
        return self.features.num_features

    def freeze_bn(self):
        """Freeze batch normalization layers for inference."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters, including stem.

        Args:
            freeze (bool): If True, freeze backbone parameters; else, unfreeze.
        """
        for name, param in self.named_parameters():
            if 'fc' not in name and 'features' not in name:
                param.requires_grad = not freeze

def iresnet100(embedding_size: int = 512, use_se: bool = False, device: str = 'cpu') -> IResNet:
    """Create IResNet-100 model."""
    return IResNet([3, 13, 30, 3], embedding_size=embedding_size, use_se=use_se, device=device)

def iresnet50(embedding_size: int = 512, use_se: bool = False, device: str = 'cpu') -> IResNet:
    """Create IResNet-50 model."""
    return IResNet([3, 4, 14, 3], embedding_size=embedding_size, use_se=use_se, device=device)

def iresnet34(embedding_size: int = 512, use_se: bool = False, device: str = 'cpu') -> IResNet:
    """Create IResNet-34 model."""
    return IResNet([3, 4, 6, 3], embedding_size=embedding_size, use_se=use_se, device=device)

def iresnet18(embedding_size: int = 512, use_se: bool = False, device: str = 'cpu') -> IResNet:
    """Create IResNet-18 model."""
    return IResNet([2, 2, 2, 2], embedding_size=embedding_size, use_se=use_se, device=device)

def load_arcface_weights(model: nn.Module, weight_path: Optional[str], 
                        strict: bool = True, verbose: bool = True) -> Tuple[bool, List[str], List[str]]:
    """Load ArcFace pretrained weights with flexible key cleaning.

    Args:
        model (nn.Module): IResNet model to load weights into.
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
        device = model.device if hasattr(model, 'device') else 'cpu'
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        cleaned_state = {}
        prefixes = ['module.', 'backbone.', 'model.', 'features.']
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            cleaned_state[new_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state, strict=strict)
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

def validate_iresnet_model(model: IResNet, input_size: Tuple[int, int, int, int] = (1, 3, 112, 112), 
                          device: str = 'cpu') -> Dict[str, Any]:
    """Validate IResNet model with comprehensive checks.

    Args:
        model (IResNet): Model to validate.
        input_size (Tuple[int, int, int, int]): Input shape (batch, channels, height, width). Defaults to (1, 3, 112, 112).
        device (str): Device for validation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Dict[str, Any]: Validation results including model info, forward pass status, and errors.
    """
    model.eval()
    model = model.to(device)
    
    results = {
        'model_name': model.__class__.__name__,
        'embedding_size': model.get_embedding_size(),
        'input_size': model.input_size,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': str(next(model.parameters()).device),
        'forward_pass': False,
        'output_shape': None,
        'output_norm': None,
        'gradient_check': False,
        'error': None
    }
    
    try:
        dummy_input = torch.randn(input_size).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            results['forward_pass'] = True
            results['output_shape'] = tuple(output.shape)
            results['output_norm'] = torch.norm(output, p=2, dim=1).mean().item()
        
        model.train()
        dummy_input.requires_grad = True
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        results['gradient_check'] = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    except Exception as e:
        results['error'] = str(e)
    
    return results

# Model configuration dictionary
IRESNET_CONFIGS = {
    'iresnet18': {'layers': [2, 2, 2, 2]},
    'iresnet34': {'layers': [3, 4, 6, 3]},
    'iresnet50': {'layers': [3, 4, 14, 3]},
    'iresnet100': {'layers': [3, 13, 30, 3]},
}

def create_iresnet(model_name: str, embedding_size: int = 512, **kwargs) -> IResNet:
    """Create IResNet model by name.

    Args:
        model_name (str): Model name ('iresnet18', 'iresnet34', 'iresnet50', 'iresnet100').
        embedding_size (int): Size of output embeddings. Defaults to 512.
        **kwargs: Additional arguments for IResNet constructor.

    Returns:
        IResNet: Configured IResNet model.

    Raises:
        ValueError: If model_name is invalid.
    """
    model_name = model_name.lower()
    if model_name not in IRESNET_CONFIGS:
        raise ValueError(f"Unknown IResNet model: {model_name}. Available: {list(IRESNET_CONFIGS.keys())}")
    
    config = IRESNET_CONFIGS[model_name]
    return IResNet(config['layers'], embedding_size=embedding_size, **kwargs)

if __name__ == "__main__":
    try:
        model = iresnet100(embedding_size=512, device='cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(2, 3, 112, 112)
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Embedding size: {model.get_embedding_size()}")
        print(f"Output norm: {torch.norm(output, p=2, dim=1)}")
        
        validation_results = validate_iresnet_model(model)
        print("Validation results:", validation_results)
    except Exception as e:
        print(f"Main block execution failed: {e}")