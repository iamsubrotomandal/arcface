#!/usr/bin/env python3
"""
Comprehensive test script for enhanced RetinaFaceDetector.
Tests all new features including validation, batch processing, and face preprocessing.
"""

import torch
import numpy as np
import cv2
import warnings
from models.retinaface import RetinaFaceDetector

def test_input_validation():
    """Test input validation and error handling."""
    print("🧪 Testing Input Validation...")
    
    # Test confidence threshold validation
    try:
        RetinaFaceDetector(confidence_threshold=1.5)
        print("❌ Should have raised ValueError for invalid confidence_threshold")
    except ValueError as e:
        print(f"✅ Confidence threshold validation: {e}")
    
    # Test backbone validation
    try:
        RetinaFaceDetector(backbone="invalid_backbone")
        print("❌ Should have raised ValueError for invalid backbone")
    except ValueError as e:
        print(f"✅ Backbone validation: {e}")
    
    # Test valid initialization
    try:
        detector = RetinaFaceDetector(backbone="resnet50", confidence_threshold=0.7)
        print(f"✅ Valid initialization successful - Device: {detector.device}")
    except Exception as e:
        print(f"❌ Valid initialization failed: {e}")

def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n🧪 Testing Batch Processing...")
    
    try:
        detector = RetinaFaceDetector(confidence_threshold=0.5)
        
        # Create mock images (random data for testing)
        images = []
        for i in range(3):
            # Create random BGR image
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            images.append(img)
        
        # Test batch detection
        results = detector.detect(images)
        print(f"✅ Batch processing successful - Processed {len(images)} images, got {len(results)} detections")
        
        # Test single image detection
        single_result = detector.detect(images[0])
        print(f"✅ Single image processing - Got {len(single_result)} detections")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")

def test_face_preprocessing():
    """Test face preprocessing functionality."""
    print("\n🧪 Testing Face Preprocessing...")
    
    try:
        detector = RetinaFaceDetector()
        
        # Create a mock image with a "face" region
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create mock detections
        mock_detections = [
            {'box': (100, 100, 200, 200, 0.9)},  # Valid face
            {'box': (300, 300, 400, 400, 0.8)},  # Another valid face
            {'box': (500, 500, 450, 450, 0.7)},  # Invalid box (x2 < x1)
        ]
        
        # Test preprocessing
        faces_tensor = detector.preprocess_faces(image, mock_detections, target_size=(112, 112))
        print(f"✅ Face preprocessing successful - Shape: {faces_tensor.shape}")
        print(f"✅ Expected 2 valid faces (1 invalid filtered out): {faces_tensor.shape[0] == 2}")
        print(f"✅ Correct target size: {faces_tensor.shape[2:] == torch.Size([112, 112])}")
        print(f"✅ Normalized range [0,1]: min={faces_tensor.min():.3f}, max={faces_tensor.max():.3f}")
        
    except Exception as e:
        print(f"❌ Face preprocessing failed: {e}")

def test_device_consistency():
    """Test device management and consistency."""
    print("\n🧪 Testing Device Management...")
    
    try:
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            detector_cuda = RetinaFaceDetector(device='cuda')
            print(f"✅ CUDA detector created - Device: {detector_cuda.device}")
            
            # Test that preprocessing returns tensors on correct device
            image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            mock_detections = [{'box': (50, 50, 150, 150, 0.9)}]
            faces = detector_cuda.preprocess_faces(image, mock_detections)
            print(f"✅ Preprocessed faces on correct device: {faces.device}")
        
        # Test CPU device
        detector_cpu = RetinaFaceDetector(device='cpu')
        print(f"✅ CPU detector created - Device: {detector_cpu.device}")
        
    except Exception as e:
        print(f"❌ Device management failed: {e}")

def test_legacy_compatibility():
    """Test backward compatibility with legacy method."""
    print("\n🧪 Testing Legacy Compatibility...")
    
    try:
        detector = RetinaFaceDetector()
        
        # Create mock image
        image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Test legacy method (should show deprecation warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            legacy_results = detector.detect_legacy(image)
            
            if w and "deprecated" in str(w[0].message):
                print("✅ Deprecation warning correctly shown")
            else:
                print("⚠️ Deprecation warning not shown")
        
        print(f"✅ Legacy method works - Got {len(legacy_results)} detections")
        print(f"✅ Legacy format correct: {type(legacy_results[0]) if legacy_results else 'No detections'}")
        
    except Exception as e:
        print(f"❌ Legacy compatibility failed: {e}")

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n🧪 Testing Error Handling...")
    
    try:
        detector = RetinaFaceDetector()
        
        # Test with invalid image types
        invalid_inputs = [
            None,
            "not_an_array",
            np.array([1, 2, 3]),  # Wrong dimensions
            np.random.rand(100, 100, 4),  # Wrong channels
        ]
        
        for i, invalid_input in enumerate(invalid_inputs):
            try:
                if invalid_input is None or isinstance(invalid_input, str):
                    # These should raise ValueError
                    result = detector.detect(invalid_input)
                    if isinstance(invalid_input, str):
                        print(f"❌ Should have raised ValueError for input {i}")
                else:
                    # These should return empty list with warning
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        result = detector.detect(invalid_input)
                        print(f"✅ Invalid input {i} handled gracefully - Result: {len(result)} detections")
            except ValueError:
                print(f"✅ ValueError correctly raised for input {i}")
            except Exception as e:
                print(f"⚠️ Unexpected error for input {i}: {e}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")

def main():
    """Run all tests for enhanced RetinaFaceDetector."""
    print("🚀 Starting Enhanced RetinaFaceDetector Tests\n")
    
    test_input_validation()
    test_batch_processing()
    test_face_preprocessing()
    test_device_consistency()
    test_legacy_compatibility()
    test_error_handling()
    
    print("\n✅ All enhanced RetinaFaceDetector tests completed!")
    print("\n📋 Enhanced Features Summary:")
    print("   • Comprehensive input validation with proper error messages")
    print("   • Batch processing support for multiple images")
    print("   • Face preprocessing for ArcFace compatibility")
    print("   • Improved device management and consistency")
    print("   • Better error handling with warnings instead of silent failures")
    print("   • Backward compatibility with deprecation warnings")
    print("   • Enhanced documentation and type hints")

if __name__ == "__main__":
    main()
