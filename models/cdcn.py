import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CDCConv(nn.Module):
    """Central Difference Convolution module."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, theta: float = 0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.theta = theta
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        
        # Generate central difference kernel
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0)
        
        return out_normal - self.theta * out_diff

class CDCN(nn.Module):
    """Central Difference Convolutional Network for Face Anti-Spoofing.
    
    Based on: "Central Difference Convolutional Networks for Face Anti-Spoofing"
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Feature extraction with CDC blocks
        self.conv1 = nn.Sequential(
            CDCConv(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            CDCConv(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            CDCConv(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            CDCConv(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Depth estimation branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, 1, 1)
        )
        
        # Classification branch
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature extraction
        x1 = self.conv1(x)  # B x 64 x 56 x 56
        x2 = self.conv2(x1) # B x 128 x 28 x 28
        x3 = self.conv3(x2) # B x 256 x 14 x 14
        x4 = self.conv4(x3) # B x 512 x 7 x 7
        
        # Depth estimation
        depth_map = self.depth_conv(x4)  # B x 1 x 7 x 7
        depth_map = F.interpolate(depth_map, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Classification (liveness score)
        cls_logits = self.classifier(x4)  # B x 2
        
        # Convert to binary score (real vs spoof)
        spoof_score = torch.softmax(cls_logits, dim=1)[:, 0]  # probability of being real
        
        return spoof_score, depth_map
