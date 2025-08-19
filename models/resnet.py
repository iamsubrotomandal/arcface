# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional
# from torchvision import models

# class BasicBlock(nn.Module):
#     """Basic ResNet block for ResNet-18/34."""
#     expansion = 1

#     def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class Bottleneck(nn.Module):
#     """Bottleneck block for ResNet-50/101/152."""
#     expansion = 4

#     def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
#         super().__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class ResNet(nn.Module):
#     """ResNet architecture for face recognition embeddings."""
    
#     def __init__(self, block, layers: List[int], embedding_size: int = 512):
#         super().__init__()
#         self.inplanes = 64
        
#         # Input stem
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # ResNet layers
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
#         # Embedding head
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, embedding_size)
#         self.bn_fc = nn.BatchNorm1d(embedding_size)
        
#         self._init_weights()
    
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.bn_fc(x)
#         x = F.normalize(x, dim=1)
        
#         return x

# def resnet100(embedding_size: int = 512) -> ResNet:
#     """ResNet-100 for still image face recognition."""
#     # ResNet-100: [3, 13, 30, 3] but with Bottleneck blocks
#     return ResNet(Bottleneck, [3, 13, 30, 3], embedding_size=embedding_size)

# def resnet50(embedding_size: int = 512) -> ResNet:
#     """ResNet-50 for comparison."""
#     return ResNet(Bottleneck, [3, 4, 6, 3], embedding_size=embedding_size)

# def load_resnet_weights(model: nn.Module, weight_path: Optional[str]) -> bool:
#     """Load ResNet pretrained weights with flexible key cleaning."""
#     if not weight_path:
#         return False
#     import os
#     if not os.path.isfile(weight_path):
#         return False
#     state = torch.load(weight_path, map_location='cpu')
#     if 'state_dict' in state:
#         state = state['state_dict']
#     # Clean common prefixes
#     cleaned = {}
#     for k, v in state.items():
#         nk = k
#         if nk.startswith('module.'):
#             nk = nk[7:]
#         if nk.startswith('backbone.'):
#             nk = nk[len('backbone.'):]
#         cleaned[nk] = v
#     missing, unexpected = model.load_state_dict(cleaned, strict=False)
#     if missing or unexpected:
#         print(f"[ResNet] Loaded with missing={len(missing)} unexpected={len(unexpected)}")
#     return True

# __all__ = ['ResNet', 'resnet100', 'resnet50', 'load_resnet_weights', 'BasicBlock', 'Bottleneck']

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import os
import warnings

class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18/34."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet architecture for face recognition embeddings.

    Produces L2-normalized embeddings for use with ArcFaceHead in a face recognition pipeline.
    Expects input images of shape (batch_size, 3, 112, 112) in RGB format, typically from
    RetinaFaceDetector.preprocess_faces.

    Args:
        block: Block type (BasicBlock or Bottleneck).
        layers (List[int]): Number of blocks in each layer.
        embedding_size (int, optional): Size of output embeddings. Defaults to 512.
        device (str, optional): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Raises:
        ValueError: If embedding_size or layers are invalid, or device is unsupported.
    """
    def __init__(self, block, layers: List[int], embedding_size: int = 512, device: str = 'cpu'):
        super().__init__()
        if embedding_size <= 0:
            raise ValueError("embedding_size must be a positive integer")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise ValueError("layers must be a list of positive integers")
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        self.device = torch.device(device)
        self.inplanes = 64
        
        # Input stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Embedding head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, embedding_size)
        self.bn_fc = nn.BatchNorm1d(embedding_size)
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            if downsample is not None:
                nn.init.kaiming_normal_(downsample[0].weight, mode='fan_out', nonlinearity='relu')
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Forward pass with optional feature return.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 112, 112) in RGB format.
            return_features (bool, optional): If True, returns features before embedding projection. Defaults to False.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Normalized embeddings or (embeddings, features).

        Raises:
            ValueError: If input tensor has invalid shape.
        """
        if x.dim() != 4 or x.shape[1] != 3 or x.shape[2] != 112 or x.shape[3] != 112:
            raise ValueError("Input must be of shape (batch_size, 3, 112, 112)")
        
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        features = self.avgpool(x)
        features = torch.flatten(features, 1)
        
        embeddings = self.fc(features)
        embeddings = self.bn_fc(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_features:
            return embeddings, features
        return embeddings

    def get_embedding_size(self) -> int:
        """Get the output embedding size."""
        return self.fc.out_features

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters, keeping the final layer trainable.

        Args:
            freeze (bool, optional): If True, freeze the backbone. If False, unfreeze. Defaults to True.
        """
        for name, param in self.named_parameters():
            if not name.startswith(('fc', 'bn_fc')):
                param.requires_grad = not freeze
            else:
                param.requires_grad = True

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count number of parameters.

        Args:
            trainable_only (bool, optional): If True, count only trainable parameters. Defaults to False.

        Returns:
            int: Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

# Model variants
def resnet18(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """ResNet-18 variant for face recognition."""
    return ResNet(BasicBlock, [2, 2, 2, 2], embedding_size=embedding_size, device=device)

def resnet34(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """ResNet-34 variant for face recognition."""
    return ResNet(BasicBlock, [3, 4, 6, 3], embedding_size=embedding_size, device=device)

def resnet50(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """ResNet-50 variant for face recognition."""
    return ResNet(Bottleneck, [3, 4, 6, 3], embedding_size=embedding_size, device=device)

def resnet100(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """ResNet-100 variant for still image face recognition."""
    return ResNet(Bottleneck, [3, 13, 30, 3], embedding_size=embedding_size, device=device)

def resnet101(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """ResNet-101 variant for face recognition."""
    return ResNet(Bottleneck, [3, 4, 23, 3], embedding_size=embedding_size, device=device)

def resnet152(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """ResNet-152 variant for face recognition."""
    return ResNet(Bottleneck, [3, 8, 36, 3], embedding_size=embedding_size, device=device)

# Note: iresnet100 is referenced in ArcFaceRecognizer but not defined here.
# If iresnet100 is distinct from resnet100, it must be implemented separately.
def iresnet100(embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """Placeholder for iresnet100. Currently uses resnet100 architecture."""
    warnings.warn("iresnet100 is not implemented; using resnet100 as fallback", UserWarning)
    return resnet100(embedding_size=embedding_size, device=device)

# Configuration dictionary
RESNET_CONFIGS = {
    'resnet18': {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
    'resnet34': {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
    'resnet50': {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
    'resnet100': {'block': Bottleneck, 'layers': [3, 13, 30, 3]},
    'resnet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
    'resnet152': {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
}

def create_resnet(model_name: str, embedding_size: int = 512, device: str = 'cpu') -> ResNet:
    """Create ResNet model by name.

    Args:
        model_name (str): Name of the ResNet variant (e.g., 'resnet100').
        embedding_size (int, optional): Size of output embeddings. Defaults to 512.
        device (str, optional): Device to place the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        ResNet: Configured ResNet model.

    Raises:
        ValueError: If model_name is invalid.
    """
    model_name = model_name.lower()
    if model_name not in RESNET_CONFIGS:
        raise ValueError(f"Unknown ResNet model: {model_name}. Available: {list(RESNET_CONFIGS.keys())}")
    config = RESNET_CONFIGS[model_name]
    return ResNet(config['block'], config['layers'], embedding_size=embedding_size, device=device)

def load_resnet_weights(model: nn.Module, weight_path: Optional[str], 
                        strict: bool = False, verbose: bool = True) -> bool:
    """Load ResNet pretrained weights with error handling.

    Args:
        model (nn.Module): ResNet model to load weights into.
        weight_path (Optional[str]): Path to the weight file.
        strict (bool, optional): If True, enforce strict key matching. Defaults to False.
        verbose (bool, optional): If True, print loading status. Defaults to True.

    Returns:
        bool: True if weights are loaded successfully, False otherwise.
    """
    if not weight_path:
        if verbose:
            warnings.warn("No weight path provided")
        return False
    
    if not os.path.isfile(weight_path):
        if verbose:
            warnings.warn(f"Weight file not found: {weight_path}")
        return False
    
    try:
        state = torch.load(weight_path, map_location=model.device if hasattr(model, 'device') else 'cpu', weights_only=True)
    except Exception as e:
        if verbose:
            warnings.warn(f"Error loading weights: {e}")
        return False
    
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'model' in state:
        state = state['model']
    
    cleaned = {}
    prefixes = ['module.', 'backbone.', 'model.', 'encoder.', 'feature_extractor.']
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned[new_key] = value
    
    try:
        missing_keys, unexpected_keys = model.load_state_dict(cleaned, strict=strict)
        if verbose:
            if missing_keys:
                warnings.warn(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                warnings.warn(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            if not missing_keys and not unexpected_keys:
                print("✅ All weights loaded successfully!")
        return len(missing_keys) == 0 and len(unexpected_keys) == 0
    except Exception as e:
        if verbose:
            warnings.warn(f"Error loading state dict: {e}")
        return False

def validate_resnet_model(model: ResNet, input_size: Tuple[int, int, int, int] = (1, 3, 112, 112), 
                         device: str = 'cpu') -> Dict[str, Any]:
    """Validate ResNet model with comprehensive checks.

    Args:
        model (ResNet): Model to validate.
        input_size (Tuple[int, int, int, int], optional): Input shape (batch, channels, height, width). Defaults to (1, 3, 112, 112).
        device (str, optional): Device for validation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Dict[str, Any]: Validation results including model info, forward pass status, and errors.
    """
    model.eval()
    model = model.to(device)
    
    results = {
        'model_name': model.__class__.__name__,
        'embedding_size': model.get_embedding_size(),
        'total_params': model.get_num_parameters(),
        'trainable_params': model.get_num_parameters(trainable_only=True),
        'device': str(next(model.parameters()).device),
        'forward_pass': False,
        'output_shape': None,
        'embedding_norm': None,
        'feature_extraction': False,
        'backbone_freezing': False,
        'gradient_check': False
    }
    
    try:
        # Test forward pass
        dummy_input = torch.randn(input_size).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            results['forward_pass'] = True
            results['output_shape'] = tuple(output.shape)
            results['embedding_norm'] = torch.norm(output, dim=1).mean().item()
        
        # Test feature extraction
        with torch.no_grad():
            embeddings, features = model(dummy_input, return_features=True)
            results['feature_extraction'] = True
            results['feature_shape'] = tuple(features.shape)
        
        # Test backbone freezing
        original_grad_state = [p.requires_grad for p in model.parameters()]
        model.freeze_backbone(True)
        fc_params_trainable = all(p.requires_grad for name, p in model.named_parameters() 
                                 if name.startswith(('fc', 'bn_fc')))
        backbone_params_frozen = any(not p.requires_grad for name, p in model.named_parameters() 
                                   if not name.startswith(('fc', 'bn_fc')))
        model.freeze_backbone(False)
        results['backbone_freezing'] = fc_params_trainable and backbone_params_frozen
        
        # Test gradient flow (for training)
        model.train()
        dummy_input = torch.randn(input_size).to(device)  # Create new tensor without gradient requirement
        dummy_input.requires_grad = True
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        results['gradient_check'] = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def test_all_resnet_variants(embedding_size: int = 512, device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
    """Test all ResNet variants for validation.

    Args:
        embedding_size (int, optional): Embedding size for models. Defaults to 512.
        device (str, optional): Device for validation ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Dict[str, Dict[str, Any]]: Validation results for each model variant.
    """
    results = {}
    for model_name in RESNET_CONFIGS.keys():
        try:
            model = create_resnet(model_name, embedding_size=embedding_size, device=device)
            validation_results = validate_resnet_model(model, device=device)
            results[model_name] = validation_results
            if validation_results['forward_pass']:
                print(f"✅ {model_name}: {validation_results['total_params']:,} params, "
                      f"output {validation_results['output_shape']}")
            else:
                print(f"❌ {model_name}: Failed validation")
        except Exception as e:
            results[model_name] = {'error': str(e)}
            print(f"❌ {model_name}: Error - {e}")
    return results

__all__ = [
    'ResNet', 'BasicBlock', 'Bottleneck',
    'resnet18', 'resnet34', 'resnet50', 'resnet100', 'resnet101', 'resnet152', 'iresnet100',
    'create_resnet', 'load_resnet_weights', 'validate_resnet_model', 'test_all_resnet_variants',
    'RESNET_CONFIGS'
]