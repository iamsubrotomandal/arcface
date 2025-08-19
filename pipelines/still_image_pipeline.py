import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from facenet_pytorch import MTCNN

from models.retinaface import RetinaFaceDetector
from models.arcface import ArcFaceRecognizer
from config import get_arcface_weight_path
from models.cdcn import CDCN
from utils.alignment import align_face
from utils.face_db import FaceDB

class StillImageFacePipeline:
    def __init__(self, device: Optional[str] = None, face_db: Optional[FaceDB] = None, match_threshold: float = 0.35):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_primary = RetinaFaceDetector(backbone="resnet50", device=self.device)
        self.detector_fallback = MTCNN(keep_all=True, device=self.device)
        arcface_w = get_arcface_weight_path()
        self.recognizer = ArcFaceRecognizer(backbone="resnet100", weight_path=arcface_w, device=self.device)
        self.liveness = CDCN().to(self.device)
        self.face_db = face_db
        self.match_threshold = match_threshold

    def detect_faces(self, img_bgr: np.ndarray):
        dets = self.detector_primary.detect(img_bgr)
        if len(dets) == 0:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes, _ = self.detector_fallback.detect(rgb)
            if boxes is None:
                return []
            return [{"box": (int(x1), int(y1), int(x2), int(y2), 1.0), "landmarks": None} for x1,y1,x2,y2 in boxes]
        return dets

    def crop_and_preprocess(self, img_bgr: np.ndarray, box, landmarks=None, size=112):
        # If landmarks available, perform alignment
        aligned = None
        if landmarks is not None:
            aligned = align_face(img_bgr, landmarks, output_size=size)
        if aligned is None:
            x1,y1,x2,y2,_ = box
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            face = img_bgr[y1:y2, x1:x2]
            if face.size == 0:
                return None
            aligned = cv2.resize(face, (size,size))
        face_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(face_rgb).permute(2,0,1).float()/255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def recognize(self, face_tensor: torch.Tensor):
        emb = self.recognizer.extract(face_tensor)
        return emb

    def liveness_score(self, face_tensor: torch.Tensor):
        score, depth = self.liveness(face_tensor)
        return score.item()

    def process_image(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        detections = self.detect_faces(img_bgr)
        for det in detections:
            box = det["box"] if isinstance(det, dict) else det
            lms = det.get("landmarks") if isinstance(det, dict) else None
            tensor = self.crop_and_preprocess(img_bgr, box, landmarks=lms)
            if tensor is None:
                continue
            emb = self.recognize(tensor)
            live = self.liveness_score(tensor)
            record = {
                "box": box,
                "landmarks": lms,
                "embedding": emb.cpu().numpy(),
                "liveness": live
            }
            if self.face_db is not None:
                pid, score = self.face_db.match(record["embedding"], threshold=self.match_threshold)
                record["identity"] = pid
                record["match_score"] = score
            results.append(record)
        return results
