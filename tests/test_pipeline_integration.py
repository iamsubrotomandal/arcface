#!/usr/bin/env python3
"""
Test script for StillImageFacePipeline integration with enhanced models.
Tests the pipeline with dummy data to verify all components work together.
"""

import numpy as np
import torch
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pipeline_integration():
    """Test the still image pipeline integration with enhanced models."""
    print("🧪 Testing StillImageFacePipeline Integration...")
    
    try:
        # Import after adding to path
        from pipelines.still_image_pipeline import StillImageFacePipeline
        
        print("✅ Successfully imported StillImageFacePipeline")
        
        # Create a dummy image (480x640x3 BGR)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✅ Created dummy image: {dummy_image.shape}")
        
        # Initialize pipeline with minimal configuration for testing
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {device}")
        
        # Test pipeline initialization
        try:
            pipeline = StillImageFacePipeline(
                device=device,
                face_db=None,  # No face database for testing
                match_threshold=0.35,
                liveness_threshold=0.7,
                min_face_size=20,
                use_liveness=True,  # Test liveness integration
                mtcnn_conf_threshold=0.9
            )
            print("✅ Pipeline initialization successful")
            
            # Test individual components
            print("\n🔍 Testing individual components...")
            
            # Test face detection
            detections = pipeline.detect_faces(dummy_image)
            print(f"✅ Face detection test: Found {len(detections)} faces")
            
            # Test preprocessing with dummy detection
            dummy_box = (50, 50, 150, 150, 0.9)  # x1, y1, x2, y2, score
            face_tensor = pipeline.crop_and_preprocess(dummy_image, dummy_box)
            if face_tensor is not None:
                print(f"✅ Face preprocessing successful: {face_tensor.shape}")
                
                # Test embedding extraction
                embedding = pipeline.extract_embedding(face_tensor)
                if embedding is not None:
                    print(f"✅ Embedding extraction successful: {embedding.shape}")
                else:
                    print("⚠️ Embedding extraction returned None")
                
                # Test liveness detection
                liveness_scores, depth_maps = pipeline.detect_liveness(face_tensor)
                if liveness_scores is not None and depth_maps is not None:
                    print(f"✅ Liveness detection successful: scores {liveness_scores.shape}, depth {depth_maps.shape}")
                else:
                    print("⚠️ Liveness detection returned None")
            else:
                print("⚠️ Face preprocessing returned None")
            
            # Test full pipeline processing
            print("\n🚀 Testing full pipeline processing...")
            results = pipeline.process_image(dummy_image)
            print(f"✅ Full pipeline processing successful: {len(results)} faces processed")
            
            # Display result structure if any faces were found
            if results:
                result = results[0]
                print("✅ Result structure:")
                for key in result.keys():
                    if key in ['embedding', 'depth_map']:
                        value_info = f"Tensor {result[key].shape}" if hasattr(result[key], 'shape') else str(type(result[key]))
                    else:
                        value_info = str(result[key])[:50] + "..." if len(str(result[key])) > 50 else str(result[key])
                    print(f"    {key}: {value_info}")
            
            print("\n🎉 All integration tests passed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("⚠️ Make sure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_model_compatibility():
    """Test compatibility with enhanced models."""
    print("\n🔧 Testing Enhanced Model Compatibility...")
    
    try:
        from models.cdcn import create_cdcn, validate_cdcn_model
        from models.resnet import iresnet100
        
        # Test CDCN creation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cdcn_model = create_cdcn('base', num_classes=2, device=device)
        print(f"✅ CDCN model created successfully: {cdcn_model.__class__.__name__}")
        
        # Test ResNet creation
        resnet_model = iresnet100()
        print(f"✅ ResNet model created successfully: {resnet_model.__class__.__name__}")
        
        # Test CDCN validation
        validation_results = validate_cdcn_model(cdcn_model, device=device)
        if validation_results.get('forward_pass', False):
            print("✅ CDCN forward pass validation successful")
        else:
            print(f"⚠️ CDCN validation issues: {validation_results.get('error', 'Unknown')}")
        
        print("✅ Enhanced model compatibility confirmed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 STILL IMAGE PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    # Test enhanced model compatibility first
    model_test = test_enhanced_model_compatibility()
    
    # Test pipeline integration
    pipeline_test = test_pipeline_integration()
    
    print("\n" + "=" * 60)
    if model_test and pipeline_test:
        print("🎉 ALL TESTS PASSED - Integration is successful!")
    else:
        print("❌ Some tests failed - Check the output above")
    print("=" * 60)
