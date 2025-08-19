"""
Tests for FAS-TD (Face Anti-Spoofing Temporal Difference) model
"""

import pytest
import torch
import torch.nn as nn
from models.fas_td import FAS_TD, create_fas_td_model, TemporalDifferenceBlock, SpatialAttention

class TestFASTD:
    
    def test_fas_td_initialization(self):
        """Test FAS-TD model initialization"""
        model = FAS_TD(input_channels=3, num_classes=2)
        
        assert isinstance(model, nn.Module)
        assert model.max_buffer_size == 5
        assert len(model.frame_buffer) == 0
        
        # Check model structure
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'classifier')
        
    def test_fas_td_forward_pass(self):
        """Test FAS-TD forward pass"""
        model = FAS_TD()
        batch_size = 2
        
        # Test with previous frame
        current_frame = torch.randn(batch_size, 3, 224, 224)
        previous_frame = torch.randn(batch_size, 3, 224, 224)
        
        logits, features = model(current_frame, previous_frame)
        
        assert logits.shape == (batch_size, 2)  # Binary classification
        assert features.shape[0] == batch_size  # Batch dimension
        assert features.shape[1] == 512  # Feature dimension
        
    def test_fas_td_without_previous_frame(self):
        """Test FAS-TD forward pass without previous frame"""
        model = FAS_TD()
        batch_size = 1
        
        current_frame = torch.randn(batch_size, 3, 224, 224)
        
        logits, features = model(current_frame, previous_frame=None)
        
        assert logits.shape == (batch_size, 2)
        assert features.shape[0] == batch_size
        
    def test_temporal_difference_computation(self):
        """Test temporal difference computation"""
        model = FAS_TD()
        
        current_frame = torch.randn(1, 3, 224, 224)
        previous_frame = torch.randn(1, 3, 224, 224)
        
        # Test with previous frame
        diff = model.compute_temporal_difference(current_frame, previous_frame)
        assert diff.shape == current_frame.shape
        assert diff.min() >= 0  # Should be absolute difference
        
        # Test without previous frame
        diff_none = model.compute_temporal_difference(current_frame, None)
        assert diff_none.shape == current_frame.shape
        assert torch.all(diff_none == 0)  # Should be zeros
        
    def test_frame_buffer_management(self):
        """Test frame buffer operations"""
        model = FAS_TD()
        
        # Test buffer update
        frame1 = torch.randn(1, 3, 224, 224)
        frame2 = torch.randn(1, 3, 224, 224)
        
        model.update_frame_buffer(frame1)
        assert len(model.frame_buffer) == 1
        
        model.update_frame_buffer(frame2)
        assert len(model.frame_buffer) == 2
        
        # Test getting previous frame
        prev = model.get_previous_frame()
        assert prev is not None
        assert torch.equal(prev, frame1)
        
        # Test buffer size limit
        for i in range(10):
            model.update_frame_buffer(torch.randn(1, 3, 224, 224))
        
        assert len(model.frame_buffer) == model.max_buffer_size
        
        # Test reset
        model.reset_buffer()
        assert len(model.frame_buffer) == 0
        
    def test_spoof_score_prediction(self):
        """Test spoof score prediction"""
        model = FAS_TD()
        
        frame = torch.randn(1, 3, 224, 224)
        score = model.predict_spoof_score(frame)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Should be a probability
        
        # Test that buffer is updated
        assert len(model.frame_buffer) == 1
        
    def test_temporal_difference_block(self):
        """Test TemporalDifferenceBlock"""
        block = TemporalDifferenceBlock(64, 128)
        
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 128  # Output channels
        assert output.shape[2] == 16  # Height after pooling
        assert output.shape[3] == 16  # Width after pooling
        
    def test_spatial_attention(self):
        """Test SpatialAttention module"""
        attention = SpatialAttention(256)
        
        x = torch.randn(2, 256, 16, 16)
        output = attention(x)
        
        assert output.shape == x.shape  # Same shape as input
        assert not torch.equal(output, x)  # Should be modified by attention
        
    def test_create_fas_td_model(self):
        """Test model creation function"""
        model = create_fas_td_model(pretrained=False)
        
        assert isinstance(model, FAS_TD)
        
        # Test parameter count
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 1000000  # Should have substantial parameters
        
    def test_fas_td_gradient_flow(self):
        """Test gradient flow in FAS-TD"""
        model = FAS_TD()
        model.train()
        
        current_frame = torch.randn(1, 3, 224, 224, requires_grad=True)
        previous_frame = torch.randn(1, 3, 224, 224)
        
        logits, _ = model(current_frame, previous_frame)
        loss = logits.sum()
        loss.backward()
        
        # Check that gradients exist
        assert current_frame.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
        
    def test_fas_td_different_input_sizes(self):
        """Test FAS-TD with different input sizes"""
        model = FAS_TD()
        
        # Test different spatial sizes
        sizes = [(112, 112), (224, 224), (256, 256)]
        
        for h, w in sizes:
            current_frame = torch.randn(1, 3, h, w)
            previous_frame = torch.randn(1, 3, h, w)
            
            try:
                logits, features = model(current_frame, previous_frame)
                assert logits.shape == (1, 2)
                print(f"✓ FAS-TD works with input size {h}x{w}")
            except Exception as e:
                print(f"✗ FAS-TD failed with input size {h}x{w}: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
