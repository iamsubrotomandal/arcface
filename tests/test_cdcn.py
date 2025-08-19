import pytest
import torch
from models.cdcn import CDCN, CDCConv


def test_cdc_conv():
    cdc = CDCConv(3, 64, 3, 1, 1)
    x = torch.randn(2, 3, 112, 112)
    out = cdc(x)
    assert out.shape == (2, 64, 112, 112)


def test_cdcn_forward():
    model = CDCN()
    x = torch.randn(2, 3, 112, 112)
    spoof_score, depth_map = model(x)
    assert spoof_score.shape == (2,)
    assert depth_map.shape == (2, 1, 112, 112)
    assert 0 <= spoof_score.min() <= spoof_score.max() <= 1  # probabilities


def test_cdcn_integration():
    # Test that it works as liveness detector in pipeline style
    model = CDCN()
    model.eval()
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        score, depth = model(x)
    assert score.item() >= 0 and score.item() <= 1
