"""Tests for video dialogue systems."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.video_dialogue import create_video_dialogue_model, SimpleVideoDialogue
from src.utils.device import get_device, set_seed
from src.utils.video_utils import frames_to_tensor
from src.eval.metrics import VideoDialogueEvaluator


class TestVideoDialogueModels:
    """Test video dialogue models."""
    
    def test_simple_model_creation(self):
        """Test simple model creation."""
        config = {"vocab_size": 1000, "hidden_dim": 128}
        model = create_video_dialogue_model("simple", config)
        
        assert isinstance(model, SimpleVideoDialogue)
        assert model.config == config
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        config = {"vocab_size": 1000, "hidden_dim": 128}
        model = create_video_dialogue_model("simple", config)
        
        # Create dummy video tensor
        video_tensor = torch.randn(1, 8, 3, 224, 224)
        question = "What is happening in this video?"
        
        # Test forward pass
        outputs = model.forward(video_tensor, question)
        
        assert "video_features" in outputs
        assert "text_features" in outputs
        assert "fused_features" in outputs
    
    def test_model_generate_response(self):
        """Test model response generation."""
        config = {"vocab_size": 1000, "hidden_dim": 128}
        model = create_video_dialogue_model("simple", config)
        
        # Create dummy video tensor
        video_tensor = torch.randn(1, 8, 3, 224, 224)
        question = "What is happening in this video?"
        
        # Test response generation
        response = model.generate_response(video_tensor, question)
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestVideoUtils:
    """Test video utility functions."""
    
    def test_frames_to_tensor(self):
        """Test frame to tensor conversion."""
        # Create dummy frames
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        
        # Convert to tensor
        tensor = frames_to_tensor(frames)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (8, 3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_frames_to_tensor_normalize(self):
        """Test frame to tensor conversion with normalization."""
        # Create dummy frames
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
        
        # Convert to tensor with normalization
        tensor = frames_to_tensor(frames, normalize=True)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4, 3, 224, 224)
        # Check if values are in reasonable range after normalization
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(1)
        np_rand = np.random.rand(1)
        
        # Set seed again and generate more
        set_seed(42)
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        # They should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        evaluator = VideoDialogueEvaluator()
        assert evaluator is not None
    
    def test_basic_metrics(self):
        """Test basic evaluation metrics."""
        evaluator = VideoDialogueEvaluator()
        
        predictions = [
            "The video shows a person walking.",
            "There are multiple objects visible.",
            "The scene takes place outdoors."
        ]
        
        references = [
            "The video shows a person walking in a park.",
            "There are several objects visible in the scene.",
            "The scene takes place in an outdoor setting."
        ]
        
        metrics = evaluator.evaluate_batch(predictions, references)
        
        assert "exact_match" in metrics
        assert "f1_score" in metrics
        assert "bleu_1" in metrics
        assert "meteor" in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics["exact_match"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["bleu_1"] <= 1
        assert 0 <= metrics["meteor"] <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline."""
        # Create model
        config = {"vocab_size": 1000, "hidden_dim": 128}
        model = create_video_dialogue_model("simple", config)
        
        # Create dummy video
        video_tensor = torch.randn(1, 8, 3, 224, 224)
        
        # Generate response
        question = "What is happening in this video?"
        response = model.generate_response(video_tensor, question)
        
        # Evaluate response
        evaluator = VideoDialogueEvaluator()
        metrics = evaluator.evaluate_batch(
            [response], 
            ["The video shows various activities."],
            [question]
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "exact_match" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
