import cv2
import numpy as np
from typing import Iterable, Tuple

# Reference 5-point template for ArcFace 112x112
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def align_face(image_bgr: np.ndarray,
               landmarks: Iterable[Tuple[float, float]],
               output_size: int = 112,
               template: np.ndarray = ARCFACE_TEMPLATE,
               border_mode: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Align face to ArcFace template using 5 landmarks.

    landmarks: iterable of 5 (x,y) in original image coordinates.
    Returns aligned BGR image (output_size x output_size) or None on failure.
    """
    try:
        pts = np.array(list(landmarks), dtype=np.float32)
        if pts.shape != (5,2):
            return None
        dst = template.copy()
        if output_size != 112:
            dst *= (output_size / 112.0)
        M, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(image_bgr, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=border_mode)
        return aligned
    except Exception:
        return None

__all__ = ["align_face", "ARCFACE_TEMPLATE"]
