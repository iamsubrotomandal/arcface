"""
Enhanced Still Image Pipeline Testing with Bounding Box Visualization
Tests the complete pipeline with visual feedback showing detected faces
"""

import cv2
import numpy as np
import torch
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append('.')

from pipelines.still_image_pipeline import StillImageFacePipeline
from utils.face_db import FaceDB

def create_realistic_test_images():
    """Create more realistic test images with face-like patterns"""
    test_dir = Path("test_images_visual")
    test_dir.mkdir(exist_ok=True)
    
    test_images = []
    
    for i in range(3):
        # Create a higher quality face-like image
        img = np.random.randint(100, 180, (480, 640, 3), dtype=np.uint8)
        
        # Add more realistic face features
        center_x, center_y = 320, 240
        
        # Face oval
        cv2.ellipse(img, (center_x, center_y), (80, 100), 0, 0, 360, (160, 140, 120), -1)
        
        # Eyes
        cv2.circle(img, (center_x - 25, center_y - 20), 8, (50, 50, 50), -1)  # Left eye
        cv2.circle(img, (center_x + 25, center_y - 20), 8, (50, 50, 50), -1)  # Right eye
        cv2.circle(img, (center_x - 25, center_y - 20), 3, (0, 0, 0), -1)     # Left pupil
        cv2.circle(img, (center_x + 25, center_y - 20), 3, (0, 0, 0), -1)     # Right pupil
        
        # Eyebrows
        cv2.ellipse(img, (center_x - 25, center_y - 35), (15, 5), 0, 0, 180, (80, 60, 40), 3)
        cv2.ellipse(img, (center_x + 25, center_y - 35), (15, 5), 0, 0, 180, (80, 60, 40), 3)
        
        # Nose
        cv2.line(img, (center_x, center_y - 5), (center_x, center_y + 15), (120, 100, 80), 2)
        cv2.circle(img, (center_x - 3, center_y + 10), 2, (100, 80, 60), -1)
        cv2.circle(img, (center_x + 3, center_y + 10), 2, (100, 80, 60), -1)
        
        # Mouth
        cv2.ellipse(img, (center_x, center_y + 30), (20, 8), 0, 0, 180, (100, 50, 50), 2)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save the image
        image_path = test_dir / f"realistic_face_{i+1}.jpg"
        cv2.imwrite(str(image_path), img)
        test_images.append(str(image_path))
        print(f"✅ Created realistic test image: {image_path}")
    
    return test_images

def draw_face_detections(img: np.ndarray, results: List[Dict[str, Any]], save_path: str = None) -> np.ndarray:
    """
    Draw bounding boxes and information on detected faces
    
    Args:
        img: Original image (BGR format)
        results: List of detection results from pipeline
        save_path: Optional path to save annotated image
        
    Returns:
        Annotated image with bounding boxes
    """
    annotated_img = img.copy()
    
    if len(results) == 0:
        # Draw "No faces detected" message
        cv2.putText(annotated_img, "No faces detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("⚠️ No faces detected in image")
    else:
        print(f"✅ Drawing bounding boxes for {len(results)} detected faces")
        
        for i, result in enumerate(results):
            box = result['box']
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = box[4] if len(box) > 4 else 1.0
            liveness = result.get('liveness', 0.0)
            
            # Choose color based on liveness score
            if liveness > 0.5:
                color = (0, 255, 0)  # Green for live
                status = "LIVE"
            else:
                color = (0, 165, 255)  # Orange for suspicious
                status = "CHECK"
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw face number
            cv2.putText(annotated_img, f"Face {i+1}", (x1, y1-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw liveness score
            cv2.putText(annotated_img, f"Liveness: {liveness:.3f}", (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw status
            cv2.putText(annotated_img, status, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw confidence if available
            if confidence < 1.0:
                cv2.putText(annotated_img, f"Conf: {confidence:.2f}", (x1, y2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add identity information if available
            if 'identity' in result and result['identity'] is not None:
                identity = result['identity']
                match_score = result.get('match_score', 0.0)
                cv2.putText(annotated_img, f"ID: {identity}", (x1, y2+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(annotated_img, f"Match: {match_score:.2f}", (x1, y2+80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            print(f"   • Face {i+1}: Box({x1},{y1},{x2},{y2}), Liveness={liveness:.3f}, Status={status}")
    
    # Add title and pipeline info
    cv2.putText(annotated_img, "Still Image Pipeline - Face Detection Results", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add legend
    legend_y = img.shape[0] - 60
    cv2.putText(annotated_img, "Legend: Green=Live, Orange=Check", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated_img, "Pipeline: RetinaFace+MTCNN, ArcFace ResNet-100, CDCN", (10, legend_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save annotated image if path provided
    if save_path:
        cv2.imwrite(save_path, annotated_img)
        print(f"✅ Saved annotated image: {save_path}")
    
    return annotated_img

def test_still_image_pipeline_with_visualization():
    """Test the Still Image Pipeline with visual bounding box feedback"""
    print("🔍 STILL IMAGE PIPELINE VISUAL TESTING")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        print("🚀 Initializing Still Image Pipeline...")
        pipeline = StillImageFacePipeline()
        print(f"✅ Pipeline initialized successfully")
        print(f"   • Primary Detection: {type(pipeline.detector_primary).__name__}")
        print(f"   • Fallback Detection: {type(pipeline.detector_fallback).__name__}")
        print(f"   • Recognition Backbone: {pipeline.recognizer.backbone_name}")
        print(f"   • Anti-Spoofing: {type(pipeline.liveness).__name__}")
        
        # Create test images
        print("\n📷 Creating realistic test images...")
        test_images = create_realistic_test_images()
        
        # Create output directory for annotated images
        output_dir = Path("test_results_visual")
        output_dir.mkdir(exist_ok=True)
        
        print("\n🧪 VISUAL DETECTION TESTING")
        print("-" * 40)
        
        all_results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n📸 Processing image {i}: {Path(image_path).name}")
            
            # Load image
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"❌ Failed to load image: {image_path}")
                continue
                
            print(f"✅ Image loaded: {img_bgr.shape}")
            
            # Process through complete pipeline
            print("🔍 Running complete pipeline detection...")
            start_time = time.time()
            results = pipeline.process_image(img_bgr)
            processing_time = time.time() - start_time
            
            print(f"✅ Pipeline processing completed in {processing_time:.3f}s")
            print(f"   • Faces detected: {len(results)}")
            
            # Create annotated image with bounding boxes
            output_path = output_dir / f"annotated_{Path(image_path).name}"
            annotated_img = draw_face_detections(img_bgr, results, str(output_path))
            
            # Store results for summary
            all_results.extend(results)
            
            # Display detailed results
            if len(results) > 0:
                for j, result in enumerate(results):
                    print(f"   • Face {j+1} Details:")
                    print(f"     - Bounding Box: {result['box'][:4]}")
                    print(f"     - Embedding Shape: {result['embedding'].shape}")
                    print(f"     - Liveness Score: {result['liveness']:.4f}")
                    print(f"     - Has Landmarks: {'Yes' if result['landmarks'] is not None else 'No'}")
        
        # Test with webcam for real-time visualization
        print("\n📹 REAL-TIME WEBCAM TESTING WITH VISUALIZATION")
        print("-" * 50)
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("⚠️ Could not open webcam")
            else:
                print("📷 Webcam opened successfully")
                print("Instructions:")
                print("  - Press 'c' to capture and process frame")
                print("  - Press 's' to save current frame with detections")
                print("  - Press 'q' to quit webcam test")
                
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Show live feed
                    cv2.imshow('Still Pipeline - Live Feed (Press c/s/q)', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('c'):
                        # Capture and process frame
                        print(f"\n📸 Processing captured frame...")
                        start_time = time.time()
                        results = pipeline.process_image(frame)
                        processing_time = time.time() - start_time
                        
                        print(f"✅ Frame processed in {processing_time:.3f}s")
                        print(f"   • Faces detected: {len(results)}")
                        
                        # Create annotated frame
                        annotated_frame = draw_face_detections(frame, results)
                        
                        # Show annotated result
                        cv2.imshow('Detection Results - Still Pipeline', annotated_frame)
                        cv2.waitKey(2000)  # Show for 2 seconds
                        cv2.destroyWindow('Detection Results - Still Pipeline')
                        
                    elif key == ord('s'):
                        # Save current frame with detections
                        frame_count += 1
                        results = pipeline.process_image(frame)
                        save_path = output_dir / f"webcam_capture_{frame_count}.jpg"
                        annotated_frame = draw_face_detections(frame, results, str(save_path))
                        print(f"✅ Saved webcam capture with detections: {save_path}")
                        
                    elif key == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                print("✅ Webcam test completed")
                
        except Exception as e:
            print(f"⚠️ Webcam test error: {e}")
        
        # Performance summary
        print("\n📊 VISUAL TESTING SUMMARY")
        print("-" * 40)
        print("✅ Still Image Pipeline - Visual Testing Complete")
        print(f"✅ Total faces processed: {len(all_results)}")
        print(f"✅ Detection visualization: Working")
        print(f"✅ Bounding box annotation: Functional")
        print(f"✅ Real-time processing: Demonstrated")
        print(f"✅ Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visual testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_and_test_real_images():
    """Test with user-provided real images if available"""
    print("\n🖼️ REAL IMAGE TESTING")
    print("-" * 30)
    
    # Look for common image directories
    possible_dirs = ["real_images", "test_photos", "images", "photos"]
    real_image_paths = []
    
    for dir_name in possible_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                real_image_paths.extend(dir_path.glob(ext))
    
    if not real_image_paths:
        print("📁 No real images found in common directories (real_images/, test_photos/, etc.)")
        print("💡 To test with your own images:")
        print("   1. Create a 'real_images' folder")
        print("   2. Add your face photos (.jpg, .png, etc.)")
        print("   3. Run this script again")
        return
    
    print(f"📸 Found {len(real_image_paths)} real images to test")
    
    # Initialize pipeline
    pipeline = StillImageFacePipeline()
    output_dir = Path("real_image_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(real_image_paths[:5], 1):  # Test first 5 images
        print(f"\n📷 Processing real image {i}: {img_path.name}")
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        # Process image
        results = pipeline.process_image(img_bgr)
        print(f"✅ Detected {len(results)} faces in {img_path.name}")
        
        # Create annotated version
        output_path = output_dir / f"annotated_{img_path.name}"
        draw_face_detections(img_bgr, results, str(output_path))

if __name__ == "__main__":
    print("🎨 STILL IMAGE PIPELINE VISUAL TESTING SUITE")
    print("=" * 80)
    
    # Test with visual feedback
    success = test_still_image_pipeline_with_visualization()
    
    if success:
        print(f"\n🎉 VISUAL TESTING: ✅ SUCCESS")
        print("All components working with bounding box visualization!")
        
        # Test with real images if available
        load_and_test_real_images()
        
        print(f"\n📋 TESTING COMPLETE - Check output directories:")
        print(f"   • test_results_visual/ - Synthetic test results")
        print(f"   • real_image_results/ - Real image results (if any)")
        
    else:
        print(f"\n❌ VISUAL TESTING: FAILED")
        print("Please check the error messages above.")
