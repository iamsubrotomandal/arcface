import pytest
import torch
from models.arcface import ArcFaceRecognizer

@pytest.mark.skip(reason='Requires insightface download (network). Enable manually when network allowed.')
def test_auto_download_mapping(tmp_path):
    # Placeholder: would invoke auto_download_arcface script then verify load.
    model = ArcFaceRecognizer()
    emb = model.extract(torch.randn(1,3,112,112))
    assert emb.shape == (1,512)
