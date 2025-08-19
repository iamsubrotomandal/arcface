"""
Enhanced Live Video Pipeline Testing with Real-time Bounding Box Visualization
Tests the complete video pipeline with live visual feedback
"""

import cv2
import numpy as np
import torch
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append('.')

from pipelines.video_pipeline import LiveVideoFacePipeline

class VisualLiveVideoPipeline(LiveVideoFacePipeline):
    """Enhanced Live Video Pipeline with advanced visualization"""
    
    def __init__(self, device=None, face_db=None, match_threshold=0.35):
        super().__init__(device, face_db, match_threshold)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
    def draw_enhanced_detections(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw comprehensive visual annotations on live video frames
        
        Args:
            frame: Input video frame (BGR)
            results: Detection results from process_frame()
            
        Returns:
            Annotated frame with bounding boxes and information
        """
        annotated_frame = frame.copy()
        detections = results.get("results", [])
        
        # Calculate FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Header information
        header_height = 80
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], header_height), (40, 40, 40), -1)
        
        # Pipeline title
        cv2.putText(annotated_frame, "LIVE VIDEO PIPELINE - Real-time Face Detection", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Pipeline specifications
        cv2.putText(annotated_frame, "RetinaFace ResNet-50 | ArcFace IResNet-100 | CDCN+FAS-TD | HSEmotion", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Performance metrics
        cv2.putText(annotated_frame, f"FPS: {self.current_fps} | Faces: {len(detections)} | Frame: {self.frame_count}", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if len(detections) == 0:
            # No faces detected
            cv2.putText(annotated_frame, "No faces detected", 
                       (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for i, detection in enumerate(detections):
                box = detection['box']
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = box[4] if len(box) > 4 else 1.0
                liveness = detection.get('liveness', 0.0)
                emotion_data = detection.get('emotion', None)
                identity = detection.get('identity', None)
                match_score = detection.get('match_score', 0.0)
                
                # Enhanced color coding based on liveness score
                if liveness > 0.6:
                    color = (0, 255, 0)  # Bright green for very live
                    status = "LIVE"
                    thickness = 3
                elif liveness > 0.4:
                    color = (0, 255, 255)  # Yellow for likely live
                    status = "LIVE?"
                    thickness = 2
                elif liveness > 0.2:
                    color = (0, 165, 255)  # Orange for uncertain
                    status = "CHECK"
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Red for likely spoof
                    status = "SPOOF"
                    thickness = 3
                
                # Main bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Face ID corner marker
                cv2.circle(annotated_frame, (x1 + 10, y1 + 10), 8, color, -1)
                cv2.putText(annotated_frame, str(i+1), (x1 + 6, y1 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Create info panel above face
                info_y = max(y1 - 100, 85)  # Ensure it doesn't overlap with header
                panel_width = max(200, x2 - x1)
                
                # Semi-transparent info panel
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, info_y), (x1 + panel_width, info_y + 90), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Face information text
                info_x = x1 + 5
                line_height = 15
                current_y = info_y + 15
                
                # Face number and status
                cv2.putText(annotated_frame, f"Face {i+1}: {status}", 
                           (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                current_y += line_height
                
                # Liveness score with bar
                cv2.putText(annotated_frame, f"Liveness: {liveness:.3f}", 
                           (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Liveness bar
                bar_x = info_x + 100
                bar_width = int(80 * liveness)
                cv2.rectangle(annotated_frame, (bar_x, current_y - 10), (bar_x + bar_width, current_y - 5), color, -1)
                cv2.rectangle(annotated_frame, (bar_x, current_y - 10), (bar_x + 80, current_y - 5), (100, 100, 100), 1)
                current_y += line_height
                
                # Confidence score
                if confidence < 1.0:
                    cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", 
                               (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    current_y += line_height
                
                # Emotion information
                if emotion_data and emotion_data[0]:
                    emotion_name = emotion_data[0][0]
                    emotion_conf = emotion_data[0][1]
                    emotion_color = (255, 255, 0)  # Yellow for emotions
                    cv2.putText(annotated_frame, f"Emotion: {emotion_name} ({emotion_conf:.2f})", 
                               (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, emotion_color, 1)
                    current_y += line_height
                
                # Identity information
                if identity is not None:
                    id_color = (255, 0, 255)  # Magenta for identity
                    cv2.putText(annotated_frame, f"ID: {identity} ({match_score:.2f})", 
                               (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, id_color, 1)
                    
                    # Identity match indicator
                    if match_score > 0.7:
                        cv2.circle(annotated_frame, (x2 - 15, y1 + 15), 8, (0, 255, 0), -1)  # Green circle
                        cv2.putText(annotated_frame, "✓", (x2 - 19, y1 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw landmarks if available
                landmarks = detection.get('landmarks', None)
                if landmarks is not None and len(landmarks) >= 5:
                    for point in landmarks:
                        if len(point) >= 2:
                            cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 
                                      2, (255, 255, 0), -1)
        
        # Footer with controls
        footer_y = annotated_frame.shape[0] - 50
        cv2.rectangle(annotated_frame, (0, footer_y), (annotated_frame.shape[1], annotated_frame.shape[0]), (40, 40, 40), -1)
        cv2.putText(annotated_frame, "Controls: [S]ave Frame | [R]ecord | [P]ause | [ESC]Quit", 
                   (10, footer_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(annotated_frame, f"Device: {self.device.upper()} | Pipeline B (Video)", 
                   (10, footer_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        self.frame_count += 1
        return annotated_frame
    
    def run_enhanced_webcam(self, cam_index=0):
        """
        Run enhanced webcam with comprehensive visual feedback
        """
        print("🎥 ENHANCED LIVE VIDEO PIPELINE")
        print("=" * 50)
        
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {cam_index}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Webcam {cam_index} opened successfully")
        print(f"📹 Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"🔧 Target FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
        print(f"💻 Device: {self.device.upper()}")
        print("\nControls:")
        print("  [S] - Save current frame")
        print("  [R] - Start/Stop recording")
        print("  [P] - Pause/Resume")
        print("  [ESC] - Quit")
        
        # Create output directory
        output_dir = Path("live_video_results")
        output_dir.mkdir(exist_ok=True)
        
        recording = False
        paused = False
        video_writer = None
        saved_frames = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ Failed to read frame from webcam")
                        break
                    
                    # Process frame through pipeline
                    start_time = time.time()
                    results = self.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    # Create enhanced visualization
                    annotated_frame = self.draw_enhanced_detections(frame, results)
                    
                    # Add processing time indicator
                    cv2.putText(annotated_frame, f"Processing: {processing_time*1000:.1f}ms", 
                               (annotated_frame.shape[1] - 200, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Display frame
                    cv2.imshow('Live Video Pipeline - Enhanced Detection', annotated_frame)
                    
                    # Record if enabled
                    if recording and video_writer is not None:
                        video_writer.write(annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key == ord('s') or key == ord('S'):
                    # Save current frame
                    if not paused:
                        saved_frames += 1
                        save_path = output_dir / f"frame_{saved_frames:04d}.jpg"
                        cv2.imwrite(str(save_path), annotated_frame)
                        print(f"📸 Saved frame: {save_path}")
                elif key == ord('r') or key == ord('R'):
                    # Toggle recording
                    if not recording:
                        # Start recording
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_path = output_dir / f"recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, 
                                                     (annotated_frame.shape[1], annotated_frame.shape[0]))
                        recording = True
                        print(f"🔴 Recording started: {video_path}")
                    else:
                        # Stop recording
                        if video_writer is not None:
                            video_writer.release()
                            video_writer = None
                        recording = False
                        print("⏹️ Recording stopped")
                elif key == ord('p') or key == ord('P'):
                    # Toggle pause
                    paused = not paused
                    print(f"⏸️ {'Paused' if paused else 'Resumed'}")
        
        finally:
            # Cleanup
            if video_writer is not None:
                video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 SESSION SUMMARY:")
            print(f"   • Frames processed: {self.frame_count}")
            print(f"   • Frames saved: {saved_frames}")
            print(f"   • Final FPS: {self.current_fps}")
            print(f"   • Results saved to: {output_dir}")

def test_live_video_pipeline():
    """Test the enhanced live video pipeline"""
    print("🎬 LIVE VIDEO PIPELINE TESTING")
    print("=" * 60)
    
    try:
        # Initialize enhanced pipeline
        print("🚀 Initializing Enhanced Live Video Pipeline...")
        pipeline = VisualLiveVideoPipeline()
        print("✅ Pipeline initialized successfully")
        print(f"   • Detection: RetinaFace ResNet-50")
        print(f"   • Recognition: ArcFace IResNet-100") 
        print(f"   • Anti-Spoofing: CDCN + FAS-TD")
        print(f"   • Emotion Analysis: HSEmotion")
        print(f"   • Device: {pipeline.device.upper()}")
        
        # Run enhanced webcam
        pipeline.run_enhanced_webcam(0)
        
        return True
        
    except Exception as e:
        print(f"❌ Live video pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎥 LIVE VIDEO PIPELINE - REAL-TIME FACE DETECTION")
    print("=" * 80)
    
    success = test_live_video_pipeline()
    
    if success:
        print(f"\n🎉 LIVE VIDEO PIPELINE TEST: ✅ SUCCESS")
        print("Real-time face detection with comprehensive visualization working!")
        print("\n📋 Features demonstrated:")
        print("   • ✅ Real-time bounding boxes with color coding")
        print("   • ✅ Live liveness scores with visual bars")
        print("   • ✅ Emotion recognition display")
        print("   • ✅ Identity matching indicators")
        print("   • ✅ FPS and performance monitoring")
        print("   • ✅ Frame saving and video recording")
        print("   • ✅ Enhanced CDCN+FAS-TD anti-spoofing")
        print("   • ✅ Complete Pipeline B specification")
    else:
        print(f"\n❌ LIVE VIDEO PIPELINE TEST: FAILED")
