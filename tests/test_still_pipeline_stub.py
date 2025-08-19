import numpy as np
import torch
from pipelines.still_image_pipeline import StillImageFacePipeline


def test_still_pipeline_no_faces():
    pipe = StillImageFacePipeline()
    img = np.zeros((480,640,3), dtype=np.uint8)
    out = pipe.process_image(img)
    assert isinstance(out, list)
    assert len(out) == 0  # blank image should have no detections


def test_recognizer_forward():
    pipe = StillImageFacePipeline()
    pipe.recognizer.eval()
    dummy = torch.randn(1,3,112,112, device=pipe.device)
    with torch.inference_mode():
        emb = pipe.recognizer.extract(dummy)
    assert emb.shape[-1] == 512
