from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
import cv2
import warnings
try:
    from facexlib.detection import RetinaFace
except ImportError:
    RetinaFace = None

class RetinaFaceDetector:
    """RetinaFace detector wrapper for face detection using facexlib.

    Detects faces in an input image, returning bounding boxes, confidence scores, and optionally landmarks.
    Supports preprocessing detected faces for compatibility with face recognition models like ArcFaceRecognizer.

    Args:
        backbone (str, optional): Backbone network ('resnet50' or 'mobilenet0.25'). Defaults to 'resnet50'.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        confidence_threshold (float, optional): Minimum confidence score for detected faces. Defaults to 0.5.
        keep_landmarks (bool, optional): Whether to include landmarks in the output. Defaults to True.

    Raises:
        ValueError: If backbone, device, or confidence_threshold is invalid.
        ImportError: If facexlib is not installed.
    """
    def __init__(self,
                 backbone: str = 'resnet50',
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 keep_landmarks: bool = True):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if backbone.lower() not in ['resnet50', 'mobilenet0.25']:
            raise ValueError("backbone must be 'resnet50' or 'mobilenet0.25'")
        if device is not None and device not in ['cpu', 'cuda']:
            raise ValueError("device must be 'cpu' or 'cuda'")
        
        self.backbone = backbone.lower()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.confidence_threshold = confidence_threshold
        self.keep_landmarks = keep_landmarks
        self.model = None
        
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the RetinaFace model with proper error handling."""
        if RetinaFace is None:
            raise ImportError("facexlib is not installed. Install with: pip install facexlib")
        
        try:
            network_name = 'resnet50' if self.backbone.startswith('resnet') else 'mobilenet0.25'
            self.model = RetinaFace(network_name=network_name, half=False, device=self.device)
            self.model.to(self.device)
            self._warm_up()
        except RuntimeError as e:
            warnings.warn(f"Error initializing RetinaFace model: {e}")
            self.model = None

    def _warm_up(self) -> None:
        """Warm up the model with a dummy input to precompile and catch errors."""
        if self.model is not None:
            try:
                dummy_input = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                self.detect(dummy_input)
            except RuntimeError as e:
                warnings.warn(f"Model warm-up failed: {e}")

    def _validate_image(self, image: Any) -> bool:
        """Validate input image format.

        Args:
            image: Input image to validate.

        Returns:
            bool: True if the image is valid, False otherwise.
        """
        if image is None or not isinstance(image, np.ndarray):
            return False
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            return False
        if image.dtype != np.uint8:
            warnings.warn("Image dtype is not uint8, may cause unexpected behavior")
        return True

    def _process_landmarks(self, landmarks: Any, mask: np.ndarray) -> Optional[np.ndarray]:
        """Process and filter landmarks.

        Args:
            landmarks: Raw landmarks from model output.
            mask: Boolean mask for filtering based on confidence.

        Returns:
            Optional[np.ndarray]: Processed landmarks with shape (num_faces, 5, 2) or None.
        """
        if landmarks is None or not self.keep_landmarks:
            return None
        
        try:
            landmarks = np.asarray(landmarks, dtype=np.float32)
            if landmarks.shape[-1] != 10 or landmarks.ndim not in [2, 3]:
                warnings.warn("Invalid landmarks shape, ignoring landmarks")
                return None
            if len(landmarks) != len(mask):
                warnings.warn("Landmarks count does not match bboxes, ignoring landmarks")
                return None
            landmarks = landmarks[mask].reshape(-1, 5, 2)
            return landmarks
        except Exception as e:
            warnings.warn(f"Error processing landmarks: {e}")
            return None

    def _process_detection_output(self, out: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and filter detection output.

        Args:
            out: Dictionary containing model output (bboxes and landmarks).

        Returns:
            List[Dict[str, Any]]: List of dictionaries with 'box' and optionally 'landmarks'.
        """
        bboxes = out.get('bbox', out.get('bboxes', None))
        landmarks = out.get('landmarks', out.get('kps', None))
        
        if bboxes is None or len(bboxes) == 0:
            return []
        
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.shape[1] == 4:
            bboxes = np.column_stack([bboxes, np.ones(bboxes.shape[0], dtype=np.float32)])
        elif bboxes.shape[1] != 5:
            warnings.warn("Invalid bounding box format")
            return []
        
        scores = bboxes[:, 4]
        mask = scores >= self.confidence_threshold
        bboxes = bboxes[mask]
        
        if bboxes.size == 0:
            return []
        
        landmarks = self._process_landmarks(landmarks, mask)
        return self._sort_and_format_results(bboxes, landmarks)

    def _sort_and_format_results(self, bboxes: np.ndarray, landmarks: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """Sort results by confidence and format output.

        Args:
            bboxes: Bounding boxes with shape (num_faces, 5).
            landmarks: Landmarks with shape (num_faces, 5, 2) or None.

        Returns:
            List[Dict[str, Any]]: Formatted detection results.
        """
        sort_indices = np.argsort(bboxes[:, 4])[::-1]
        bboxes = bboxes[sort_indices]
        if landmarks is not None:
            landmarks = landmarks[sort_indices]
        
        results = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            result = {
                'box': (float(x1), float(y1), float(x2), float(y2), float(score)),
                'landmarks': landmarks[i] if self.keep_landmarks and landmarks is not None and i < len(landmarks) else None
            }
            results.append(result)
        return results

    @torch.inference_mode()
    def detect(self, image_bgr: Any) -> List[Dict[str, Any]]:
        """Detect faces in an image or list of images.

        Args:
            image_bgr (np.ndarray or list): Input image(s) in BGR format with shape (height, width, 3).

        Returns:
            List[Dict[str, Any]]: List of dictionaries with keys 'box' (x1, y1, x2, y2, score) and optionally 'landmarks' (5, 2).
        """
        if isinstance(image_bgr, (list, tuple)):
            results = []
            for img in image_bgr:
                if not self._validate_image(img):
                    warnings.warn("Invalid image in batch, skipping")
                    continue
                results.extend(self._detect_single(img))
            return results
        if not self._validate_image(image_bgr):
            raise ValueError("image_bgr must be a NumPy array with shape (height, width, 3)")
        return self._detect_single(image_bgr)

    def _detect_single(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a single BGR image.

        Args:
            image_bgr: Input image in BGR format with shape (height, width, 3).

        Returns:
            List[Dict[str, Any]]: List of detection dictionaries.
        """
        if self.model is None:
            return []
        
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            with torch.inference_mode():
                out = self.model.detect_faces(img_rgb)
        except RuntimeError as e:
            warnings.warn(f"Model inference failed: {e}")
            return []
        
        if not isinstance(out, dict):
            warnings.warn("Unexpected output format from model")
            return []
        
        return self._process_detection_output(out)

    def preprocess_faces(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]], 
                        target_size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
        """Preprocess detected faces for face recognition.

        Crops, resizes, and normalizes detected faces to match ArcFaceRecognizer input requirements.

        Args:
            image_bgr: Input image in BGR format with shape (height, width, 3).
            detections: Detections from detect method.
            target_size: Target size for resized faces. Defaults to (112, 112).

        Returns:
            torch.Tensor: Tensor of shape (num_faces, 3, target_size[0], target_size[1]) in RGB format, normalized to [0, 1].
        """
        if not self._validate_image(image_bgr):
            raise ValueError("image_bgr must be a NumPy array with shape (height, width, 3)")
        
        faces = []
        for det in detections:
            x1, y1, x2, y2, _ = det['box']
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                warnings.warn("Invalid bounding box, skipping")
                continue
            face = image_bgr[y1:y2, x1:x2]
            if face.size == 0:
                warnings.warn("Empty face crop, skipping")
                continue
            face = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
            faces.append(face)
        
        return torch.stack(faces).to(self.device) if faces else torch.tensor([], device=self.device)

    def detect_legacy(self, image_bgr: Any) -> List[Tuple[int, int, int, int, float]]:
        """Legacy method to detect faces and return integer bounding boxes.

        Deprecated: Use detect() method instead for more detailed output.

        Args:
            image_bgr (np.ndarray or list): Input image(s) in BGR format with shape (height, width, 3).

        Returns:
            List[Tuple[int, int, int, int, float]]: List of (x1, y1, x2, y2, score) for detected faces.
        """
        warnings.warn("detect_legacy is deprecated; use detect() instead", DeprecationWarning)
        return [(int(d['box'][0]), int(d['box'][1]), int(d['box'][2]), int(d['box'][3]), float(d['box'][4]))
                for d in self.detect(image_bgr)]