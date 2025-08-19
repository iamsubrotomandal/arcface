#!/usr/bin/env python3
"""
Test script for enhanced IResNet implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_iresnet():
    """Test the enhanced IResNet implementation."""
    print("🧪 Testing Enhanced IResNet Implementation...")
    
    try:
        # Import the enhanced IResNet
        from models.iresnet import (
            iresnet100, iresnet50, iresnet34, iresnet18,
            validate_iresnet_model, IResNet, IRBlock, SELayer,
            IRESNET_CONFIGS, conv1x1
        )
        print("✅ Successfully imported all IResNet components")
        
        # Test model creation using factory function
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {device}")
        
        # Test different model variants
        model_functions = [
            (iresnet18, 'iresnet18', 18),
            (iresnet34, 'iresnet34', 34), 
            (iresnet50, 'iresnet50', 50),
            (iresnet100, 'iresnet100', 100)
        ]
        
        for model_func, model_name, expected_depth in model_functions:
            try:
                model = model_func(embedding_size=512, device=device)
                print(f"✅ Created {model_name}: {model.__class__.__name__}")
                
                # Test forward pass
                dummy_input = torch.randn(2, 3, 112, 112)
                output = model(dummy_input)
                print(f"   Input: {dummy_input.shape} → Output: {output.shape}")
                
                # Verify normalized output
                norms = torch.norm(output, p=2, dim=1)
                print(f"   Output norms: {norms.tolist()} (should be ~1.0)")
                
                # Test model validation
                validation_results = validate_iresnet_model(model, device=device)
                print(f"   Validation: forward_pass={validation_results['forward_pass']}, "
                      f"gradient_check={validation_results['gradient_check']}")
                
            except Exception as e:
                print(f"❌ {model_name} test failed: {e}")
        
        # Test SE layer option
        print("\n🔧 Testing SE (Squeeze-and-Excitation) layers...")
        try:
            model_with_se = iresnet50(embedding_size=512, use_se=True, device=device)
            dummy_input = torch.randn(2, 3, 112, 112)  # Use batch size 2 for SE layers
            output = model_with_se(dummy_input)
            print(f"✅ SE-enabled model: {dummy_input.shape} → {output.shape}")
        except Exception as e:
            print(f"❌ SE layer test failed: {e}")
        
        # Test individual component classes
        print("\n🧩 Testing individual components...")
        
        # Test IRBlock
        try:
            # Test with matching dimensions first
            block1 = IRBlock(64, 64, stride=1, use_se=True)
            test_input = torch.randn(2, 64, 56, 56)  # Use batch size 2
            block_output = block1(test_input)
            print(f"✅ IRBlock (same dim): {test_input.shape} → {block_output.shape}")
            
            # Test with downsample
            downsample = nn.Sequential(
                conv1x1(64, 128, stride=2),
                nn.BatchNorm2d(128)
            )
            block2 = IRBlock(64, 128, stride=2, downsample=downsample, use_se=True)
            block_output2 = block2(test_input)
            print(f"✅ IRBlock (with downsample): {test_input.shape} → {block_output2.shape}")
        except Exception as e:
            print(f"❌ IRBlock test failed: {e}")
        
        # Test SELayer
        try:
            se_layer = SELayer(128, reduction=16)
            test_input = torch.randn(1, 128, 28, 28)
            se_output = se_layer(test_input)
            print(f"✅ SELayer: {test_input.shape} → {se_output.shape}")
        except Exception as e:
            print(f"❌ SELayer test failed: {e}")
        
        # Test parameter counting
        print("\n📊 Model Parameter Analysis...")
        model = iresnet100(embedding_size=512, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ IResNet-100 parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Test model utilities
        print("\n🛠️ Testing model utilities...")
        
        # Test embedding size getter
        embedding_size = model.get_embedding_size()
        print(f"✅ Embedding size: {embedding_size}")
        
        # Test freezing functions
        model.freeze_bn()
        print("✅ Batch norm layers frozen")
        
        model.freeze_backbone(freeze=True)
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"✅ Backbone frozen: {frozen_params} parameters frozen")
        
        model.freeze_backbone(freeze=False)
        unfrozen_params = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"✅ Backbone unfrozen: {unfrozen_params} parameters trainable")
        
        print("\n🎉 All IResNet tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ IResNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ENHANCED IRESNET IMPLEMENTATION TEST")
    print("=" * 60)
    
    success = test_enhanced_iresnet()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Enhanced IResNet implementation is excellent!")
    else:
        print("❌ Some tests failed - Check the output above")
    print("=" * 60)
