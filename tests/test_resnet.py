import torch
from models.resnet import resnet100, resnet50
from models.arcface import ArcFaceRecognizer


def test_resnet100():
    model = resnet100(embedding_size=512)
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 512)
    # Check normalization
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_resnet50():
    model = resnet50(embedding_size=512)
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    assert out.shape == (2, 512)


def test_arcface_with_resnet100():
    recognizer = ArcFaceRecognizer(backbone="resnet100", embedding_size=512)
    x = torch.randn(1, 3, 112, 112)
    emb = recognizer.extract(x)
    assert emb.shape == (1, 512)


def test_arcface_with_iresnet100():
    recognizer = ArcFaceRecognizer(backbone="iresnet100", embedding_size=512)
    x = torch.randn(1, 3, 112, 112)
    emb = recognizer.extract(x)
    assert emb.shape == (1, 512)
