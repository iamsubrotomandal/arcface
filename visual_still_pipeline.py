"""
Enhanced Still Image Pipeline with Built-in Visualization
Adds bounding box visualization directly to the pipeline
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import original pipeline
import sys
sys.path.append('.')
from pipelines.still_image_pipeline import StillImageFacePipeline

class VisualStillImagePipeline(StillImageFacePipeline):
    """Enhanced Still Image Pipeline with built-in visualization capabilities"""
    
    def __init__(self, device: Optional[str] = None, face_db: Optional[object] = None, 
                 match_threshold: float = 0.35, show_confidence: bool = True,
                 show_liveness: bool = True, show_identity: bool = True):
        super().__init__(device, face_db, match_threshold)
        self.show_confidence = show_confidence
        self.show_liveness = show_liveness
        self.show_identity = show_identity
    
    def draw_detections(self, img: np.ndarray, results: List[Dict[str, Any]], 
                       save_path: Optional[str] = None) -> np.ndarray:
        """
        Draw bounding boxes and annotations on detected faces
        
        Args:
            img: Original image (BGR format)
            results: Detection results from process_image()
            save_path: Optional path to save annotated image
            
        Returns:
            Annotated image with bounding boxes
        """
        annotated_img = img.copy()
        
        if len(results) == 0:
            # No faces detected
            cv2.putText(annotated_img, "No faces detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_img, "Pipeline: RetinaFace+MTCNN -> ArcFace ResNet-100 -> CDCN", 
                       (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            for i, result in enumerate(results):
                box = result['box']
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = box[4] if len(box) > 4 else 1.0
                liveness = result.get('liveness', 0.0)
                
                # Color coding based on liveness score
                if liveness > 0.5:
                    color = (0, 255, 0)  # Green for likely live
                    status = "LIVE"
                elif liveness > 0.3:
                    color = (0, 165, 255)  # Orange for uncertain
                    status = "CHECK"
                else:
                    color = (0, 0, 255)  # Red for likely spoof
                    status = "SPOOF"
                
                # Draw main bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                
                # Face number label
                label_y = y1 - 10
                cv2.putText(annotated_img, f"Face {i+1}", (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Liveness information
                if self.show_liveness:
                    label_y -= 25
                    cv2.putText(annotated_img, f"Liveness: {liveness:.3f} ({status})", 
                               (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Confidence score
                if self.show_confidence and confidence < 1.0:
                    label_y -= 20
                    cv2.putText(annotated_img, f"Conf: {confidence:.2f}", 
                               (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Identity information
                if self.show_identity and 'identity' in result and result['identity'] is not None:
                    identity = result['identity']
                    match_score = result.get('match_score', 0.0)
                    
                    # Identity box below main detection
                    id_y = y2 + 25
                    cv2.putText(annotated_img, f"ID: {identity}", (x1, id_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(annotated_img, f"Match: {match_score:.2f}", (x1, id_y + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw landmarks if available
                if result.get('landmarks') is not None:
                    landmarks = result['landmarks']
                    if landmarks is not None and len(landmarks) >= 5:
                        # Draw 5-point landmarks (eyes, nose, mouth corners)
                        for point in landmarks:
                            if len(point) >= 2:
                                cv2.circle(annotated_img, (int(point[0]), int(point[1])), 
                                          2, (255, 255, 0), -1)
            
            # Pipeline information footer
            footer_y = img.shape[0] - 40
            cv2.putText(annotated_img, "Still Image Pipeline: RetinaFace+MTCNN Detection", 
                       (10, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_img, "ArcFace ResNet-100 Recognition + CDCN Anti-Spoofing", 
                       (10, footer_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"✅ Saved annotated image: {save_path}")
        
        return annotated_img
    
    def process_and_visualize(self, img: np.ndarray, save_path: Optional[str] = None, 
                             show_image: bool = False) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Process image and return both results and visualization
        
        Args:
            img: Input image (BGR format)
            save_path: Optional path to save annotated result
            show_image: Whether to display the image
            
        Returns:
            Tuple of (detection_results, annotated_image)
        """
        # Process through pipeline
        results = self.process_image(img)
        
        # Create visualization
        annotated_img = self.draw_detections(img, results, save_path)
        
        # Display if requested
        if show_image:
            cv2.imshow('Still Pipeline Detection Results', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results, annotated_img
    
    def process_image_file(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single image file and save annotated result
        
        Args:
            image_path: Path to input image file
            output_dir: Directory to save results (default: same as input)
            
        Returns:
            Dictionary with processing results and paths
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Determine output path
        input_path = Path(image_path)
        if output_dir is None:
            output_dir = input_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"detected_{input_path.name}"
        
        # Process and visualize
        results, annotated_img = self.process_and_visualize(img, str(output_path))
        
        return {
            'input_path': image_path,
            'output_path': str(output_path),
            'num_faces': len(results),
            'results': results,
            'processing_success': True
        }
    
    def batch_process_directory(self, input_dir: str, output_dir: Optional[str] = None,
                               extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            extensions: List of file extensions to process
            
        Returns:
            List of processing results for each image
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = input_path / "detection_results"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"⚠️ No image files found in {input_dir}")
            return []
        
        print(f"📸 Processing {len(image_files)} images from {input_dir}")
        
        # Process each image
        results = []
        for i, img_file in enumerate(image_files, 1):
            try:
                print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
                
                result = self.process_image_file(str(img_file), str(output_path))
                results.append(result)
                
                print(f"✅ Detected {result['num_faces']} faces in {img_file.name}")
                
            except Exception as e:
                print(f"❌ Error processing {img_file.name}: {e}")
                results.append({
                    'input_path': str(img_file),
                    'output_path': None,
                    'num_faces': 0,
                    'results': [],
                    'processing_success': False,
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r['processing_success'])
        total_faces = sum(r['num_faces'] for r in results if r['processing_success'])
        
        print(f"\n📊 BATCH PROCESSING SUMMARY:")
        print(f"   • Images processed: {successful}/{len(image_files)}")
        print(f"   • Total faces detected: {total_faces}")
        print(f"   • Results saved to: {output_path}")
        
        return results

def demo_visual_pipeline():
    """Demonstration of the Visual Still Image Pipeline"""
    print("🎨 VISUAL STILL IMAGE PIPELINE DEMO")
    print("=" * 50)
    
    # Initialize visual pipeline
    pipeline = VisualStillImagePipeline()
    print("✅ Visual pipeline initialized")
    
    # Test with webcam
    print("\n📹 Testing with webcam...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("📷 Webcam opened. Press 'c' to capture, 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow('Visual Pipeline - Live Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    print("📸 Capturing and processing frame...")
                    results, annotated = pipeline.process_and_visualize(frame)
                    print(f"✅ Detected {len(results)} faces")
                    
                    cv2.imshow('Detection Results', annotated)
                    cv2.waitKey(3000)  # Show for 3 seconds
                    cv2.destroyWindow('Detection Results')
                    
                elif key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Webcam demo completed")
        else:
            print("⚠️ Could not open webcam")
    
    except Exception as e:
        print(f"⚠️ Webcam demo error: {e}")

if __name__ == "__main__":
    demo_visual_pipeline()
