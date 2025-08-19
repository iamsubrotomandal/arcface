import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class IRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                conv3x3(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.prelu(out)
        return out

class IResNet(nn.Module):
    """Simplified IR-ResNet for ArcFace embeddings (input 3x112x112)."""
    def __init__(self, layers: List[int], embedding_size: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.inplanes = 64
        self.stage1 = self._make_layer(64, layers[0])
        self.stage2 = self._make_layer(128, layers[1], stride=2)
        self.stage3 = self._make_layer(256, layers[2], stride=2)
        self.stage4 = self._make_layer(512, layers[3], stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten()
        # After adaptive pool to 7x7, channels 512 -> 512*7*7 = 25088
        self.embedding = nn.Linear(512 * 7 * 7, embedding_size, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_size, affine=True)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        layers = [IRBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IRBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn(x)
        x = F.adaptive_avg_pool2d(x, (7,7))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.embedding(x)
        x = self.embedding_bn(x)
        x = F.normalize(x, dim=1)
        return x

def iresnet100(embedding_size: int = 512) -> IResNet:
    return IResNet([3,13,30,3], embedding_size=embedding_size)

def load_arcface_weights(model: nn.Module, weight_path: Optional[str]):
    if not weight_path:
        return False
    import os
    if not os.path.isfile(weight_path):
        return False
    state = torch.load(weight_path, map_location='cpu')
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    return True
