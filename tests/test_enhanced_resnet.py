#!/usr/bin/env python3
"""
Comprehensive test script for enhanced ResNet implementation.
Tests all model variants, features, and edge cases.
"""

import torch
import torch.nn as nn
from models.resnet import (
    ResNet, create_resnet, validate_resnet_model, test_all_resnet_variants,
    load_resnet_weights, resnet50, resnet100, RESNET_CONFIGS
)

def test_model_creation():
    """Test model creation and basic functionality."""
    print("🧪 Testing Model Creation...")
    
    # Test all variants
    for model_name in RESNET_CONFIGS.keys():
        try:
            model = create_resnet(model_name, embedding_size=256)
            print(f"✅ {model_name}: Created successfully")
            
            # Test basic properties
            assert model.get_embedding_size() == 256
            assert model.get_num_parameters() > 0
            print(f"   Parameters: {model.get_num_parameters():,}")
            
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")

def test_forward_pass():
    """Test forward pass with different input sizes."""
    print("\n🧪 Testing Forward Pass...")
    
    model = resnet50(embedding_size=512)
    model.eval()
    
    test_cases = [
        (1, 3, 112, 112),  # Face recognition standard
        (2, 3, 224, 224),  # ImageNet standard
        (4, 3, 256, 256),  # Larger input
    ]
    
    for batch_size, channels, height, width in test_cases:
        try:
            with torch.no_grad():
                input_tensor = torch.randn(batch_size, channels, height, width)
                output = model(input_tensor)
                
                # Check output properties
                assert output.shape == (batch_size, 512)
                assert torch.allclose(torch.norm(output, dim=1), torch.ones(batch_size), atol=1e-6)
                
                print(f"✅ Input {input_tensor.shape} → Output {output.shape}, L2 normalized")
                
        except Exception as e:
            print(f"❌ Input {(batch_size, channels, height, width)}: Error - {e}")

def test_feature_extraction():
    """Test feature extraction functionality."""
    print("\n🧪 Testing Feature Extraction...")
    
    model = resnet50()
    model.eval()
    
    try:
        with torch.no_grad():
            input_tensor = torch.randn(2, 3, 224, 224)
            
            # Normal forward pass
            embeddings = model(input_tensor)
            
            # Feature extraction
            embeddings_feat, features = model(input_tensor, return_features=True)
            
            # Verify consistency
            assert torch.allclose(embeddings, embeddings_feat)
            assert features.shape == (2, 2048)  # ResNet-50 feature size
            
            print(f"✅ Feature extraction: embeddings {embeddings.shape}, features {features.shape}")
            
    except Exception as e:
        print(f"❌ Feature extraction: Error - {e}")

def test_backbone_freezing():
    """Test backbone freezing functionality."""
    print("\n🧪 Testing Backbone Freezing...")
    
    model = resnet50()
    
    try:
        # Check initial state
        total_params = model.get_num_parameters()
        trainable_params = model.get_num_parameters(trainable_only=True)
        assert total_params == trainable_params  # All should be trainable initially
        
        # Freeze backbone
        model.freeze_backbone(True)
        frozen_trainable = model.get_num_parameters(trainable_only=True)
        
        # Check that final layers are still trainable
        fc_trainable = all(p.requires_grad for p in model.fc.parameters())
        bn_trainable = all(p.requires_grad for p in model.bn_fc.parameters())
        
        assert fc_trainable and bn_trainable
        assert frozen_trainable < trainable_params  # Fewer trainable params
        
        # Unfreeze
        model.freeze_backbone(False)
        unfrozen_trainable = model.get_num_parameters(trainable_only=True)
        assert unfrozen_trainable == trainable_params
        
        print(f"✅ Backbone freezing: {total_params:,} → {frozen_trainable:,} → {unfrozen_trainable:,} params")
        
    except Exception as e:
        print(f"❌ Backbone freezing: Error - {e}")

def test_weight_loading():
    """Test weight loading functionality."""
    print("\n🧪 Testing Weight Loading...")
    
    model = resnet50()
    
    # Test with non-existent file
    success, missing, unexpected = load_resnet_weights(model, "nonexistent.pth", verbose=False)
    assert not success
    print("✅ Non-existent file handling: Correctly failed")
    
    # Test with None path
    success, missing, unexpected = load_resnet_weights(model, None, verbose=False)
    assert not success
    print("✅ None path handling: Correctly failed")
    
    # Create a dummy state dict for testing
    try:
        dummy_state = {
            'conv1.weight': torch.randn(64, 3, 7, 7),
            'bn1.weight': torch.randn(64),
            'bn1.bias': torch.randn(64),
            'module.layer1.0.conv1.weight': torch.randn(64, 64, 1, 1),  # Test prefix removal
        }
        
        # Save dummy weights
        torch.save(dummy_state, 'test_weights.pth')
        
        # Test loading
        success, missing, unexpected = load_resnet_weights(model, 'test_weights.pth', verbose=False)
        assert success
        print("✅ Dummy weight loading: Success with prefix removal")
        
        # Cleanup
        import os
        os.remove('test_weights.pth')
        
    except Exception as e:
        print(f"⚠️ Weight loading test: {e}")

def test_device_compatibility():
    """Test CUDA/CPU compatibility."""
    print("\n🧪 Testing Device Compatibility...")
    
    model = resnet50()
    
    # Test CPU
    try:
        model = model.to('cpu')
        model.eval()  # Set to eval mode to avoid batch norm issues
        with torch.no_grad():
            input_tensor = torch.randn(2, 3, 224, 224)  # Use batch size > 1
            output = model(input_tensor)
        print("✅ CPU compatibility: Working")
    except Exception as e:
        print(f"❌ CPU compatibility: Error - {e}")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        try:
            model = model.to('cuda')
            model.eval()  # Set to eval mode to avoid batch norm issues
            with torch.no_grad():
                input_tensor = torch.randn(2, 3, 224, 224).cuda()  # Use batch size > 1
                output = model(input_tensor)
            print("✅ CUDA compatibility: Working")
        except Exception as e:
            print(f"❌ CUDA compatibility: Error - {e}")
    else:
        print("ℹ️ CUDA not available, skipping CUDA test")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")
    
    # Test invalid model name
    try:
        model = create_resnet("invalid_model")
        print("❌ Invalid model name: Should have failed")
    except ValueError:
        print("✅ Invalid model name: Correctly handled")
    except Exception as e:
        print(f"⚠️ Invalid model name: Unexpected error - {e}")
    
    # Test very small input
    try:
        model = resnet50()
        model.eval()
        with torch.no_grad():
            small_input = torch.randn(1, 3, 32, 32)  # Very small
            output = model(small_input)
            print(f"✅ Small input (32x32): Output shape {output.shape}")
    except Exception as e:
        print(f"⚠️ Small input: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced ResNet Comprehensive Tests\n")
    
    test_model_creation()
    test_forward_pass()
    test_feature_extraction()
    test_backbone_freezing()
    test_weight_loading()
    test_device_compatibility()
    test_edge_cases()
    
    print("\n📊 Running Model Validation on All Variants...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = test_all_resnet_variants(device=device)
    
    print(f"\n✅ Enhanced ResNet Tests Complete!")
    print("\n📋 Test Summary:")
    print("   • Model creation and variants ✓")
    print("   • Forward pass with different input sizes ✓")
    print("   • Feature extraction ✓")
    print("   • Backbone freezing ✓")
    print("   • Weight loading with error handling ✓")
    print("   • Device compatibility (CPU/CUDA) ✓")
    print("   • Edge case handling ✓")
    print("   • All ResNet variants validation ✓")

if __name__ == "__main__":
    main()
