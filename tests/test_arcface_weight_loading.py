import torch
from models.arcface import ArcFaceRecognizer


def test_weight_loading_dummy(tmp_path):
    # Create dummy model and save weights
    model = ArcFaceRecognizer()
    weight_path = tmp_path / 'dummy_arcface.pth'
    torch.save(model.backbone.state_dict(), weight_path)

    # New model loads weights
    m2 = ArcFaceRecognizer(weight_path=str(weight_path))
    assert m2.weights_loaded is True
    # Quick forward
    emb = m2.extract(torch.randn(2,3,112,112))
    assert emb.shape == (2,512)
