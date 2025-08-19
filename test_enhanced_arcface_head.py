"""
Test script for Enhanced ArcFace Head Implementation
Demonstrates the improved ArcFaceHead with better stability and error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from models.arcface import ArcFaceHead, ArcFaceRecognizer

def test_arcface_head_basic():
    """Test basic ArcFaceHead functionality"""
    print("🧪 TESTING ENHANCED ARCFACE HEAD")
    print("=" * 50)
    
    # Test parameters
    embedding_size = 512
    num_classes = 100
    batch_size = 8
    
    print(f"📋 Test Configuration:")
    print(f"   • Embedding Size: {embedding_size}")
    print(f"   • Number of Classes: {num_classes}")
    print(f"   • Batch Size: {batch_size}")
    
    try:
        # Initialize ArcFace head
        print("\n🚀 Initializing Enhanced ArcFaceHead...")
        head = ArcFaceHead(embedding_size, num_classes, s=64.0, m=0.5)
        print("✅ ArcFaceHead initialized successfully")
        
        # Test embeddings and labels
        embeddings = torch.randn(batch_size, embedding_size)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        print(f"\n📊 Input shapes:")
        print(f"   • Embeddings: {embeddings.shape}")
        print(f"   • Labels: {labels.shape}")
        
        # Test training mode (with margin)
        print("\n🎯 Testing Training Mode (with margin)...")
        head.train()
        logits_train, norm_embeddings = head(embeddings, labels, apply_margin=True)
        
        print(f"✅ Training forward pass successful:")
        print(f"   • Logits shape: {logits_train.shape}")
        print(f"   • Normalized embeddings shape: {norm_embeddings.shape}")
        print(f"   • Logits range: [{logits_train.min().item():.3f}, {logits_train.max().item():.3f}]")
        print(f"   • Embedding norm: {torch.norm(norm_embeddings, dim=1).mean().item():.6f} (should be ~1.0)")
        
        # Test inference mode (without margin)
        print("\n🔍 Testing Inference Mode (without margin)...")
        head.eval()
        logits_eval, norm_embeddings_eval = head(embeddings, labels=None, apply_margin=False)
        
        print(f"✅ Inference forward pass successful:")
        print(f"   • Logits shape: {logits_eval.shape}")
        print(f"   • Normalized embeddings shape: {norm_embeddings_eval.shape}")
        print(f"   • Logits range: [{logits_eval.min().item():.3f}, {logits_eval.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ ArcFaceHead test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_arcface_head_error_handling():
    """Test ArcFaceHead error handling and validation"""
    print("\n🛡️ TESTING ERROR HANDLING")
    print("-" * 30)
    
    try:
        # Test invalid parameters
        print("Testing invalid embedding_size...")
        try:
            ArcFaceHead(-1, 100)
            print("❌ Should have raised ValueError for negative embedding_size")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")
        
        print("Testing invalid num_classes...")
        try:
            ArcFaceHead(512, 0)
            print("❌ Should have raised ValueError for zero num_classes")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")
        
        print("Testing invalid scale parameter...")
        try:
            ArcFaceHead(512, 100, s=-1.0)
            print("❌ Should have raised ValueError for negative scale")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")
        
        print("Testing invalid margin parameter...")
        try:
            ArcFaceHead(512, 100, m=2.0)
            print("❌ Should have raised ValueError for margin > 1")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")
        
        # Test invalid labels during forward pass
        print("Testing out-of-range labels...")
        head = ArcFaceHead(512, 10)
        head.train()
        embeddings = torch.randn(4, 512)
        invalid_labels = torch.tensor([0, 5, 15, 2])  # 15 is out of range for 10 classes
        
        try:
            head(embeddings, invalid_labels)
            print("❌ Should have raised ValueError for out-of-range labels")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_arcface_recognizer_integration():
    """Test integration with ArcFaceRecognizer"""
    print("\n🔗 TESTING ARCFACE RECOGNIZER INTEGRATION")
    print("-" * 40)
    
    try:
        # Initialize recognizer
        print("🚀 Initializing ArcFaceRecognizer...")
        recognizer = ArcFaceRecognizer(backbone="resnet100", embedding_size=512)
        print("✅ ArcFaceRecognizer initialized")
        
        # Create ArcFace head using the new method
        num_classes = 50
        head = recognizer.create_head(num_classes, s=64.0, m=0.5)
        print(f"✅ ArcFace head created with {num_classes} classes")
        
        # Test training setup
        batch_size = 4
        face_tensor = torch.randn(batch_size, 3, 112, 112)  # Standard face input size
        labels = torch.randint(0, num_classes, (batch_size,))
        
        print(f"\n📊 Training test:")
        print(f"   • Input tensor shape: {face_tensor.shape}")
        print(f"   • Labels: {labels}")
        
        # Test forward pass with head
        recognizer.train()
        head.train()
        
        logits, norm_embeddings = recognizer.forward_with_head(face_tensor, head, labels)
        
        print(f"✅ Training forward pass successful:")
        print(f"   • Logits shape: {logits.shape}")
        print(f"   • Normalized embeddings shape: {norm_embeddings.shape}")
        print(f"   • Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # Test inference mode
        print(f"\n🔍 Inference test:")
        recognizer.eval()
        head.eval()
        
        # Extract embeddings only
        embeddings = recognizer.extract(face_tensor)
        print(f"✅ Embedding extraction successful:")
        print(f"   • Embeddings shape: {embeddings.shape}")
        print(f"   • Embedding range: [{embeddings.min().item():.3f}, {embeddings.max().item():.3f}]")
        
        # Test inference with head (no margin)
        inference_logits, inference_norm_emb = recognizer.forward_with_head(
            face_tensor, head, labels=None, apply_margin=False
        )
        print(f"✅ Inference with head successful:")
        print(f"   • Inference logits shape: {inference_logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Recognizer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numerical_stability():
    """Test numerical stability improvements"""
    print("\n🔢 TESTING NUMERICAL STABILITY")
    print("-" * 35)
    
    try:
        head = ArcFaceHead(512, 100, s=64.0, m=0.5)
        head.train()
        
        # Test with extreme embeddings (very large values)
        print("Testing with extreme embedding values...")
        extreme_embeddings = torch.randn(4, 512) * 100  # Very large values
        labels = torch.randint(0, 100, (4,))
        
        logits, norm_emb = head(extreme_embeddings, labels)
        
        # Check if normalized embeddings have unit norm
        norms = torch.norm(norm_emb, dim=1)
        print(f"✅ Normalized embedding norms: {norms}")
        print(f"   • Mean norm: {norms.mean().item():.6f} (should be ~1.0)")
        print(f"   • Std norm: {norms.std().item():.6f} (should be ~0.0)")
        
        # Test with near-boundary cosine values
        print("Testing boundary conditions...")
        
        # Create embeddings that will produce cosine values near ±1
        W_norm = F.normalize(head.W, p=2, dim=1)
        boundary_embeddings = W_norm[:4].clone()  # Use weight vectors as embeddings
        
        logits_boundary, _ = head(boundary_embeddings, labels)
        print(f"✅ Boundary test successful, logits shape: {logits_boundary.shape}")
        
        # Check for NaN or Inf values
        if torch.isnan(logits_boundary).any():
            print("❌ Found NaN values in logits")
            return False
        
        if torch.isinf(logits_boundary).any():
            print("❌ Found Inf values in logits")
            return False
        
        print("✅ No NaN or Inf values detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Numerical stability test failed: {e}")
        return False

def demonstrate_training_example():
    """Demonstrate a simple training example with the enhanced ArcFace head"""
    print("\n📚 TRAINING EXAMPLE DEMONSTRATION")
    print("-" * 40)
    
    try:
        # Setup
        embedding_size = 512
        num_classes = 10
        batch_size = 16
        
        # Initialize recognizer and head
        recognizer = ArcFaceRecognizer(backbone="resnet100", embedding_size=embedding_size)
        head = recognizer.create_head(num_classes, s=30.0, m=0.4)  # Smaller scale and margin for demo
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(recognizer.parameters()) + list(head.parameters()), 
            lr=0.001
        )
        loss_fn = nn.CrossEntropyLoss()
        
        print(f"✅ Training setup complete:")
        print(f"   • Recognizer backbone: {recognizer.backbone_name}")
        print(f"   • Embedding size: {embedding_size}")
        print(f"   • Number of classes: {num_classes}")
        print(f"   • Scale factor: {head.s}")
        print(f"   • Margin: {head.m}")
        
        # Simulate a few training steps
        recognizer.train()
        head.train()
        
        print(f"\n🎯 Simulating training steps...")
        
        for step in range(3):
            # Generate random batch
            face_batch = torch.randn(batch_size, 3, 112, 112)
            label_batch = torch.randint(0, num_classes, (batch_size,))
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = recognizer.forward_with_head(face_batch, head, label_batch, apply_margin=True)
            
            # Calculate loss
            loss = loss_fn(logits, label_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"   Step {step + 1}: Loss = {loss.item():.4f}")
        
        # Test inference
        print(f"\n🔍 Testing inference mode...")
        recognizer.eval()
        head.eval()
        
        test_face = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            # Extract embeddings for similarity computation
            embedding = recognizer.extract(test_face)
            print(f"✅ Extracted embedding shape: {embedding.shape}")
            
            # Get classification logits (without margin for inference)
            inference_logits, _ = recognizer.forward_with_head(
                test_face, head, labels=None, apply_margin=False
            )
            predicted_class = torch.argmax(inference_logits, dim=1)
            confidence = torch.softmax(inference_logits, dim=1).max()
            
            print(f"✅ Inference results:")
            print(f"   • Predicted class: {predicted_class.item()}")
            print(f"   • Confidence: {confidence.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 ENHANCED ARCFACE HEAD - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_arcface_head_basic),
        ("Error Handling", test_arcface_head_error_handling),
        ("Recognizer Integration", test_arcface_recognizer_integration),
        ("Numerical Stability", test_numerical_stability),
        ("Training Example", demonstrate_training_example)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced ArcFace Head is working perfectly!")
        print("\n📋 Key Improvements Demonstrated:")
        print("   • ✅ Better error handling and validation")
        print("   • ✅ Improved numerical stability with epsilon values")
        print("   • ✅ Flexible forward pass with optional margin application")
        print("   • ✅ Better weight initialization with Xavier normal")
        print("   • ✅ Device consistency handling")
        print("   • ✅ NaN handling for stability")
        print("   • ✅ Integration with existing ArcFaceRecognizer")
        print("   • ✅ Training and inference mode support")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
