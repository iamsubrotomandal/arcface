## Pretrained Weights
Place your ArcFace IR-100 weights (state_dict .pth) into `weights/` and instantiate:
```python
from models.arcface import ArcFaceRecognizer
rec = ArcFaceRecognizer(weight_path='weights/iresnet100.pth', freeze=True)
```
# Face Recognition Pipelines (Still Image & Live Video)

Two separate pipelines implemented primarily in PyTorch (minimal fallback dependencies) per your specification.

## Pipelines

1. **Still Images**
   - Detection: RetinaFace (placeholder) with MTCNN fallback.
   - Recognition: ArcFace (IResNet-100 placeholder backbone stub).
   - Liveness: CDCN (simplified stub producing pseudo depth & score).

2. **Live Video**
   - Detection: RetinaFace (ResNet-50 placeholder).
      - Recognition: ArcFace (IR-ResNet-100 simplified implementation, supports external pretrained weights).
   - Liveness: CDCN + second CDCN (placeholder for FAS-TD fusion by averaging scores).
   - Emotion: HSEmotion model.

> NOTE: Current RetinaFace, ArcFace backbone, CDCN, and FAS-TD components are lightweight placeholder skeletons. Integrate real pretrained weights / full architectures for production accuracy.

## Install

```bash
pip install -r requirements.txt
```

## Usage (Still Image)
```python
import cv2
from pipelines.still_image_pipeline import StillImageFacePipeline

img = cv2.imread('example.jpg')
pipeline = StillImageFacePipeline()
results = pipeline.process_image(img)
for r in results:
    print(r['box'], r['liveness'], r['embedding'].shape)
```

## Usage (Live Video)
```python
from pipelines.video_pipeline import LiveVideoFacePipeline
pipeline = LiveVideoFacePipeline()
pipeline.run_webcam(0)
```

## Next Steps (Recommended)
- Replace placeholder RetinaFace with official implementation and load pretrained weights.
- Replace `IResNet100` stub with full IR-SE-100 or similar backbone + pretrained ArcFace weights.
- Implement proper face alignment (5-point landmarks) before embedding extraction.
- Integrate real CDCN / FAS-TD architectures with depth map supervision and thresholds.
- Add embedding database management (enroll & search with cosine similarity / ANN index).
- Calibration for liveness score thresholds; integrate temporal cues (blink, motion).
- Batch processing & GPU mixed precision for speed.
- Add unit tests and benchmarking scripts.

## Disclaimer
This scaffold is for structural demonstration only; performance metrics will be poor until real models are integrated.
