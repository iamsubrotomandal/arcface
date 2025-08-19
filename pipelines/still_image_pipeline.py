import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from facenet_pytorch import MTCNN
from PIL import Image

from models.retinaface import RetinaFaceDetector
from models.arcface import ArcFaceRecognizer
from models.cdcn import CDCN, create_cdcn
from config import get_arcface_weight_path
from utils.alignment import align_face
from utils.face_db import FaceDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StillImageFacePipeline:
    """Face processing pipeline for detection, recognition, and liveness detection.

    Integrates RetinaFaceDetector for face detection, CDCN for liveness detection,
    and ArcFaceRecognizer for face recognition, with optional FaceDB for identity matching.

    Args:
        device (Optional[str]): Torch device ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        face_db (Optional[FaceDB]): Face database for recognition.
        match_threshold (float): Cosine similarity threshold for face matching (0 to 1).
        liveness_threshold (float): Threshold for liveness detection (0 to 1).
        min_face_size (int): Minimum face size in pixels to process.
        use_liveness (bool): Whether to enable liveness detection.
        mtcnn_conf_threshold (float): Confidence threshold for MTCNN fallback (0 to 1).

    Raises:
        ValueError: If parameters are invalid.
    """
    def __init__(self, 
                 device: Optional[str] = None, 
                 face_db: Optional[FaceDB] = None, 
                 match_threshold: float = 0.35,
                 liveness_threshold: float = 0.7,
                 min_face_size: int = 20,
                 use_liveness: bool = True,
                 mtcnn_conf_threshold: float = 0.9):
        if device is not None and device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if not 0 <= match_threshold <= 1 or not 0 <= liveness_threshold <= 1 or not 0 <= mtcnn_conf_threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")
        if min_face_size <= 0:
            raise ValueError("min_face_size must be positive")
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.face_db = face_db
        self.match_threshold = match_threshold
        self.liveness_threshold = liveness_threshold
        self.min_face_size = min_face_size
        self.use_liveness = use_liveness
        self.mtcnn_conf_threshold = mtcnn_conf_threshold
        
        logger.info(f"Initializing pipeline on device: {self.device}")
        
        # Initialize detectors
        self.detector_primary = RetinaFaceDetector(
            backbone="resnet50", 
            device=str(self.device),
            confidence_threshold=0.6
        )
        
        self.detector_fallback = MTCNN(
            keep_all=True, 
            device=self.device,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Initialize recognizer
        arcface_w = get_arcface_weight_path()
        self.recognizer = ArcFaceRecognizer(
            backbone="resnet100", 
            weight_path=arcface_w, 
            device=str(self.device)
        )
        
        # Initialize liveness detector
        if self.use_liveness:
            self.liveness_model = create_cdcn(
                model_type='base', 
                num_classes=2, 
                input_size=(112, 112), 
                device=str(self.device)
            )
            self.liveness_model.eval()
            logger.info("Liveness detection enabled")
        else:
            self.liveness_model = None
            logger.info("Liveness detection disabled")
        
        # Warm up models
        self._warm_up_models()

    def _warm_up_models(self):
        """Warm up models with a dummy input to catch initialization errors."""
        try:
            dummy_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            detections = self.detect_faces(dummy_img)
            if detections:
                face_tensor = self.crop_and_preprocess(dummy_img, detections[0]["box"], 
                                                     detections[0].get("landmarks"))
                if face_tensor is not None:
                    self.extract_embedding(face_tensor)
                    if self.use_liveness:
                        self.detect_liveness(face_tensor)
            logger.info("Models warmed up successfully")
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")

    def detect_faces(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFaceDetector with MTCNN fallback.

        Args:
            img_bgr (np.ndarray): Input image in BGR format (height, width, 3).

        Returns:
            List[Dict[str, Any]]: List of detection dictionaries with 'box' (x1, y1, x2, y2, score)
                                  and 'landmarks' (5, 2) or None.
        """
        detections = []
        
        if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            logger.error("Invalid input image: must be a BGR numpy array")
            return detections
        
        try:
            detections = self.detector_primary.detect(img_bgr)
            if detections:
                detections = [det for det in detections if self._validate_bbox(det["box"], img_bgr.shape)]
                logger.debug(f"Primary detector found {len(detections)} faces")
                return detections
        except Exception as e:
            logger.warning(f"Primary detector failed: {e}")
        
        # Fallback to MTCNN
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            boxes, probs, landmarks = self.detector_fallback.detect(pil_img, landmarks=True)
            
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < self.mtcnn_conf_threshold:
                        continue
                    x1, y1, x2, y2 = box
                    if not self._validate_bbox((x1, y1, x2, y2, prob), img_bgr.shape):
                        continue
                    mtcnn_landmarks = landmarks[i] if landmarks is not None and i < len(landmarks) else None
                    detections.append({
                        "box": (float(x1), float(y1), float(x2), float(y2), float(prob)),
                        "landmarks": mtcnn_landmarks
                    })
                logger.debug(f"Fallback detector found {len(detections)} faces")
        except Exception as e:
            logger.error(f"Fallback detector failed: {e}")
        
        return detections

    def _validate_bbox(self, bbox: Tuple[float, float, float, float, float], 
                      img_shape: Tuple[int, int]) -> bool:
        """Validate bounding box coordinates.

        Args:
            bbox: Tuple of (x1, y1, x2, y2, score).
            img_shape: Image shape (height, width).

        Returns:
            bool: True if the bounding box is valid.
        """
        x1, y1, x2, y2, score = bbox
        img_h, img_w = img_shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        width, height = x2 - x1, y2 - y1
        if width < self.min_face_size or height < self.min_face_size:
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        return True

    def crop_and_preprocess(self, img_bgr: np.ndarray, box: Tuple[float, float, float, float, float], 
                          landmarks: Optional[np.ndarray] = None, size: int = 112) -> Optional[torch.Tensor]:
        """Crop and preprocess a face region.

        Args:
            img_bgr (np.ndarray): Input image in BGR format.
            box: Bounding box (x1, y1, x2, y2, score).
            landmarks: Landmark points (5, 2) or None.
            size (int): Target size for the cropped face. Defaults to 112.

        Returns:
            Optional[torch.Tensor]: Preprocessed face tensor (1, 3, size, size) in RGB format, normalized to [0, 1].
        """
        try:
            if not self._validate_bbox(box, img_bgr.shape):
                logger.debug("Invalid bounding box")
                return None
            
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), x2, y2])
            
            aligned = None
            if landmarks is not None and landmarks.shape == (5, 2):
                try:
                    aligned = align_face(img_bgr, landmarks, output_size=size)
                except Exception as e:
                    logger.warning(f"Face alignment failed: {e}")
            
            if aligned is None:
                x2 = min(img_bgr.shape[1], x2)
                y2 = min(img_bgr.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    logger.debug("Invalid crop coordinates")
                    return None
                face_region = img_bgr[y1:y2, x1:x2]
                if face_region.size == 0:
                    logger.debug("Empty face region")
                    return None
                aligned = cv2.resize(face_region, (size, size), interpolation=cv2.INTER_LINEAR)
            
            face_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
            if tensor.shape != (3, size, size):
                logger.error("Invalid preprocessed tensor shape")
                return None
            
            return tensor.unsqueeze(0).to(self.device)
        
        except Exception as e:
            logger.error(f"Face cropping failed: {e}")
            return None

    def extract_embedding(self, face_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract face embedding.

        Args:
            face_tensor (torch.Tensor): Preprocessed face tensor (batch_size, 3, 112, 112).

        Returns:
            Optional[torch.Tensor]: Normalized embedding tensor (batch_size, embedding_size).
        """
        try:
            with torch.no_grad():
                return self.recognizer.extract(face_tensor)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def detect_liveness(self, face_tensor: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Detect liveness for a batch of faces.

        Args:
            face_tensor (torch.Tensor): Preprocessed face tensor (batch_size, 3, 112, 112).

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Liveness scores (batch_size,) and depth maps (batch_size, 1, 112, 112).
        """
        if not self.use_liveness or self.liveness_model is None:
            return None, None
        
        try:
            with torch.no_grad():
                cls_logits, depth_map = self.liveness_model(face_tensor)
                liveness_scores = torch.softmax(cls_logits, dim=1)[:, 1]
                return liveness_scores, depth_map
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            return None, None

    def process_image(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Process an image to detect, recognize, and verify faces.

        Args:
            img_bgr (np.ndarray): Input image in BGR format (height, width, 3).

        Returns:
            List[Dict[str, Any]]: List of face records with detection, liveness, and recognition info.
                                  Keys include 'box', 'confidence', 'landmarks', 'embedding',
                                  'liveness_score', 'is_live', 'identity', 'similarity_score', 'is_recognized'.
        """
        results = []
        
        try:
            if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
                logger.error("Invalid input image: must be a BGR numpy array")
                return results
            
            detections = self.detect_faces(img_bgr)
            if not detections:
                logger.debug("No faces detected")
                return results
            
            face_tensors = []
            valid_detections = []
            for det in detections:
                box = det["box"]
                landmarks = det.get("landmarks")
                face_tensor = self.crop_and_preprocess(img_bgr, box, landmarks)
                if face_tensor is not None:
                    face_tensors.append(face_tensor)
                    valid_detections.append(det)
            
            if not face_tensors:
                logger.debug("No valid face tensors after preprocessing")
                return results
            
            face_tensors = torch.cat(face_tensors, dim=0)
            embeddings = self.extract_embedding(face_tensors)
            liveness_scores, depth_maps = self.detect_liveness(face_tensors)
            
            for i, (det, emb) in enumerate(zip(valid_detections, embeddings)):
                liveness_score = liveness_scores[i].item() if liveness_scores is not None else None
                depth_map = depth_maps[i] if depth_maps is not None else None
                record = {
                    "box": det["box"][:4],
                    "confidence": float(det["box"][4]),
                    "landmarks": det.get("landmarks"),
                    "embedding": emb,  # Keep as tensor
                    "liveness_score": liveness_score,
                    "is_live": liveness_score > self.liveness_threshold if liveness_score is not None else True,
                    "depth_map": depth_map
                }
                
                if self.face_db is not None:
                    pid, similarity = self.face_db.match(emb.cpu().numpy().flatten(), 
                                                       threshold=self.match_threshold)
                    record.update({
                        "identity": pid,
                        "similarity_score": similarity,
                        "is_recognized": similarity > self.match_threshold
                    })
                
                results.append(record)
            
            logger.info(f"Processed {len(results)} faces successfully")
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results

    def process_batch_images(self, images: List[np.ndarray], batch_size: int = 4) -> List[List[Dict[str, Any]]]:
        """Process multiple images in batches with true batch processing.

        Args:
            images (List[np.ndarray]): List of input images in BGR format.
            batch_size (int): Number of faces to process in a single batch.

        Returns:
            List[List[Dict[str, Any]]]: List of results for each image.
        """
        results = []
        
        try:
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_results = []
                
                all_face_tensors = []
                all_detections = []
                for img in batch:
                    detections = self.detect_faces(img)
                    face_tensors = []
                    valid_detections = []
                    for det in detections:
                        face_tensor = self.crop_and_preprocess(img, det["box"], det.get("landmarks"))
                        if face_tensor is not None:
                            face_tensors.append(face_tensor)
                            valid_detections.append(det)
                    all_face_tensors.append(face_tensors)
                    all_detections.append(valid_detections)
                    batch_results.append([])  # Placeholder for results
                
                # Process faces in batch
                flat_tensors = [t for tensors in all_face_tensors for t in tensors]
                if flat_tensors:
                    flat_tensors = torch.cat(flat_tensors, dim=0)
                    embeddings = self.extract_embedding(flat_tensors)
                    liveness_scores, depth_maps = self.detect_liveness(flat_tensors)
                    
                    tensor_idx = 0
                    for batch_idx, (detections, face_tensors) in enumerate(zip(all_detections, all_face_tensors)):
                        for det in detections:
                            emb = embeddings[tensor_idx]
                            liveness_score = liveness_scores[tensor_idx].item() if liveness_scores is not None else None
                            depth_map = depth_maps[tensor_idx] if depth_maps is not None else None
                            record = {
                                "box": det["box"][:4],
                                "confidence": float(det["box"][4]),
                                "landmarks": det.get("landmarks"),
                                "embedding": emb,
                                "liveness_score": liveness_score,
                                "is_live": liveness_score > self.liveness_threshold if liveness_score is not None else True,
                                "depth_map": depth_map
                            }
                            
                            if self.face_db is not None:
                                pid, similarity = self.face_db.match(emb.cpu().numpy().flatten(), 
                                                                   threshold=self.match_threshold)
                                record.update({
                                    "identity": pid,
                                    "similarity_score": similarity,
                                    "is_recognized": similarity > self.match_threshold
                                })
                            
                            batch_results[batch_idx].append(record)
                            tensor_idx += 1
                
                results.extend(batch_results)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        
        return results

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'detector_fallback'):
                del self.detector_fallback
            if hasattr(self, 'liveness_model'):
                del self.liveness_model
            if hasattr(self, 'recognizer'):
                del self.recognizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Pipeline resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")