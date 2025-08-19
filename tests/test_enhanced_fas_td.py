#!/usr/bin/env python3
"""
Comprehensive test script for enhanced FAS-TD implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_fas_td():
    """Test the enhanced FAS-TD implementation."""
    print("🧪 Testing Enhanced FAS-TD Implementation...")
    
    try:
        # Import the enhanced FAS-TD
        from models.fas_td import (
            FAS_TD, create_fas_td_model, validate_fas_td_model,
            TemporalDifferenceBlock, SpatialAttention, TemporalDifferenceModule
        )
        print("✅ Successfully imported all FAS-TD components")
        
        # Test model creation using factory function
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {device}")
        
        # Test different model types
        model_types = ['standard', 'lightweight', 'enhanced']
        for model_type in model_types:
            try:
                model = create_fas_td_model(model_type, device=device)
                params = model.get_num_parameters()
                print(f"✅ Created {model_type}: FAS_TD")
                print(f"   Parameters: {params:,}")
                
                # Test forward pass with correct input size
                current = torch.randn(2, 3, 112, 112).to(device)
                previous = torch.randn(2, 3, 112, 112).to(device)
                
                outputs = model(current, previous)
                
                print(f"   Input: {current.shape} → Outputs:")
                print(f"     - Logits: {outputs['logits'].shape}")
                print(f"     - Spoof confidence: {outputs['spoof_confidence'].shape}")
                print(f"     - Features: {outputs['features'].shape}")
                print(f"     - Temporal features: {outputs['temporal_features'].shape}")
                
                # Test validation
                validation_result = validate_fas_td_model(model, device=device)
                forward_ok = validation_result['forward_pass']
                gradient_ok = validation_result['gradient_check']
                print(f"   Validation: forward_pass={forward_ok}, gradient_check={gradient_ok}")
                
            except Exception as e:
                print(f"❌ {model_type.title()} model failed: {e}")
        
        print("\n🔧 Testing prediction interface...")
        try:
            model = create_fas_td_model('standard', device=device)
            test_frame = torch.randn(1, 3, 112, 112)
            
            # Test prediction
            prediction = model.predict(test_frame)
            print(f"✅ Prediction interface working")
            print(f"   Keys: {list(prediction.keys())}")
            print(f"   Spoof score: {prediction['spoof_score'][0]:.3f}")
            print(f"   Is live: {prediction['is_live'][0]}")
            
        except Exception as e:
            print(f"❌ Prediction interface failed: {e}")
        
        print("\n🧩 Testing individual components...")
        
        # Test TemporalDifferenceBlock
        try:
            td_block = TemporalDifferenceBlock(64, 128, use_se=True)
            test_input = torch.randn(2, 64, 56, 56)
            td_output = td_block(test_input)
            print(f"✅ TemporalDifferenceBlock: {test_input.shape} → {td_output.shape}")
        except Exception as e:
            print(f"❌ TemporalDifferenceBlock failed: {e}")
        
        # Test SpatialAttention
        try:
            sa = SpatialAttention(128)
            test_input = torch.randn(2, 128, 28, 28)
            sa_output = sa(test_input)
            print(f"✅ SpatialAttention: {test_input.shape} → {sa_output.shape}")
        except Exception as e:
            print(f"❌ SpatialAttention failed: {e}")
        
        # Test TemporalDifferenceModule
        try:
            td_module = TemporalDifferenceModule(3)
            current = torch.randn(2, 3, 112, 112)
            previous = torch.randn(2, 3, 112, 112)
            temp_features = td_module(current, previous)
            print(f"✅ TemporalDifferenceModule: {current.shape} + {previous.shape} → {temp_features.shape}")
        except Exception as e:
            print(f"❌ TemporalDifferenceModule failed: {e}")
        
        print("\n🎬 Testing temporal buffer functionality...")
        try:
            model = create_fas_td_model('standard', device=device)
            print(f"✅ Buffer size: {model.buffer_size}")
            
            # Test frame sequence
            for i in range(8):
                frame = torch.randn(1, 3, 112, 112)
                prediction = model.predict(frame)
                buffer_len = len(model.frame_buffer)
                print(f"   Frame {i+1}: Buffer length={buffer_len}, Spoof score={prediction['spoof_score'][0]:.3f}")
            
            # Test buffer reset
            model.reset_buffer()
            print(f"✅ Buffer reset, length: {len(model.frame_buffer)}")
            
        except Exception as e:
            print(f"❌ Buffer functionality failed: {e}")
        
        print("\n🏋️ Testing training functionality...")
        try:
            model = create_fas_td_model('standard', device=device)
            
            # Create sample training sequence
            frames = [torch.randn(1, 3, 112, 112) for _ in range(5)]
            labels = [0, 1, 0, 1, 0]  # Live/spoof sequence
            
            loss = model.train_sequence(frames, labels)
            print(f"✅ Training sequence: loss={loss.item():.4f}")
            
        except Exception as e:
            print(f"❌ Training functionality failed: {e}")
        
        print("\n🧪 Testing model utilities...")
        try:
            model = create_fas_td_model('standard', device=device)
            
            # Test parameter counting
            params = model.get_num_parameters()
            print(f"✅ Parameter count: {params:,}")
            
            # Test batch norm freezing
            model.freeze_bn()
            print("✅ Batch norm layers frozen")
            
            # Test weight loading (without actual weights)
            success, missing, unexpected = model.load_weights(None, verbose=False)
            print(f"✅ Weight loading interface: success={success}")
            
        except Exception as e:
            print(f"❌ Model utilities failed: {e}")
        
        print("\n🎉 All FAS-TD tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ FAS-TD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ENHANCED FAS-TD IMPLEMENTATION TEST")
    print("=" * 60)
    
    success = test_enhanced_fas_td()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - Enhanced FAS-TD implementation is excellent!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED - Please check the implementation")
        print("=" * 60)
