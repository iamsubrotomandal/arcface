import cv2
import numpy as np
import torch
from typing import Dict, Any, Optional
from hsemotion.facial_emotions import HSEmotionRecognizer

from models.retinaface import RetinaFaceDetector
from models.arcface import ArcFaceRecognizer
from config import get_arcface_weight_path
from models.cdcn import CDCN
from models.fas_td import create_fas_td_model
from utils.alignment import align_face
from utils.face_db import FaceDB

class LiveVideoFacePipeline:
    def __init__(self, device: Optional[str] = None, face_db: Optional[FaceDB] = None, match_threshold: float = 0.35):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = RetinaFaceDetector(backbone="resnet50", device=self.device)
        arcface_w = get_arcface_weight_path()
        self.recognizer = ArcFaceRecognizer(backbone="iresnet100", weight_path=arcface_w, device=self.device)
        
        # CDCN + FAS-TD Integration for enhanced liveness detection
        self.liveness_cdc = CDCN().to(self.device)
        self.liveness_fas_td = create_fas_td_model(pretrained=False).to(self.device)
        
        self.emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
        self.face_db = face_db
        self.match_threshold = match_threshold
        
        # Store previous frame for FAS-TD temporal analysis
        self.previous_frame_tensor = None

    def detect_faces(self, frame_bgr):
        return self.detector.detect(frame_bgr)

    def crop_and_preprocess(self, frame_bgr, box, landmarks=None, size=112):
        aligned = None
        if landmarks is not None:
            aligned = align_face(frame_bgr, landmarks, output_size=size)
        if aligned is None:
            x1,y1,x2,y2,_ = box
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                return None
            aligned = cv2.resize(face,(size,size))
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def recognize(self, face_tensor):
        return self.recognizer.extract(face_tensor)

    def liveness_scores(self, face_tensor):
        """
        CDCN + FAS-TD Integration for comprehensive liveness detection
        
        Combines:
        - CDCN: Central Difference CNN for spatial anti-spoofing
        - FAS-TD: Temporal Difference analysis for motion-based detection
        """
        # Ensure input is on the correct device
        face_tensor = face_tensor.to(self.device)
        
        # Set models to eval mode for inference
        self.liveness_cdc.eval()
        self.liveness_fas_td.eval()
        
        with torch.no_grad():
            # CDCN spatial analysis
            cdcn_score, cdcn_depth = self.liveness_cdc(face_tensor)
            cdcn_score = float(cdcn_score.item() if hasattr(cdcn_score, 'item') else cdcn_score)
            
            # FAS-TD temporal analysis
            # Ensure previous frame is on the same device if it exists
            previous_frame_device = self.previous_frame_tensor.to(self.device) if self.previous_frame_tensor is not None else None
            
            fas_td_logits, _ = self.liveness_fas_td(face_tensor, previous_frame_device)
            
            # Convert logits to spoof probability (sigmoid for binary classification)
            fas_td_score = torch.sigmoid(fas_td_logits[:, 1]).item() if fas_td_logits.size(1) > 1 else 0.5
        
        # Update previous frame for next iteration
        self.previous_frame_tensor = face_tensor.clone()
        
        # Weighted combination of CDCN and FAS-TD scores
        # CDCN weight: 0.6 (spatial features are generally more reliable)
        # FAS-TD weight: 0.4 (temporal features add valuable motion analysis)
        combined_score = 0.6 * cdcn_score + 0.4 * fas_td_score
        
        return float(combined_score)

    def emotion(self, frame_bgr, box):
        x1,y1,x2,y2,_ = box
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return self.emotion_model.predict_emotions(rgb)

    def process_frame(self, frame_bgr) -> Dict[str, Any]:
        outputs = []
        dets = self.detect_faces(frame_bgr)
        for det in dets:
            box = det["box"] if isinstance(det, dict) else det
            lms = det.get("landmarks") if isinstance(det, dict) else None
            t = self.crop_and_preprocess(frame_bgr, box, landmarks=lms)
            if t is None:
                continue
            emb = self.recognize(t)
            live = self.liveness_scores(t)
            emo = self.emotion(frame_bgr, box)
            rec = {
                "box": box,
                "landmarks": lms,
                "embedding": emb.cpu().numpy(),
                "liveness": live,
                "emotion": emo
            }
            if self.face_db is not None:
                pid, score = self.face_db.match(rec["embedding"], threshold=self.match_threshold)
                rec["identity"] = pid
                rec["match_score"] = score
            outputs.append(rec)
        return {"results": outputs}

    def run_webcam(self, cam_index=0):
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            data = self.process_frame(frame)
            for item in data["results"]:
                x1,y1,x2,y2,_ = item["box"]
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                label = f"live:{item['liveness']:.2f}"
                if 'identity' in item and item['identity'] is not None:
                    label += f" {item['identity']}:{item.get('match_score',0):.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
                if item["emotion"]:
                    cv2.putText(frame, item["emotion"][0][0], (int(x1), int(y2)+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
            cv2.imshow("Live", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
