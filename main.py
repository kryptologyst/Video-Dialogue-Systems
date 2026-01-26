"""Main entry point for video dialogue systems."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device import get_device, set_seed, print_device_info
from src.models.video_dialogue import create_video_dialogue_model
from src.utils.video_utils import load_video_frames, frames_to_tensor
from src.data.dataset import create_sample_dataset
from src.eval.metrics import VideoDialogueEvaluator, create_evaluation_report


def main():
    """Main function demonstrating video dialogue system."""
    print("=" * 60)
    print("VIDEO DIALOGUE SYSTEM DEMO")
    print("=" * 60)
    
    # Set up device and seed
    device = get_device()
    set_seed(42)
    print_device_info()
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    create_sample_dataset(data_dir, num_samples=50)
    
    # Initialize model
    print("\nInitializing video dialogue model...")
    config = {
        "vocab_size": 10000,
        "hidden_dim": 256,
        "num_layers": 3
    }
    
    model = create_video_dialogue_model("simple", config)
    model.to(device)
    model.eval()
    
    # Demo with sample video (create a dummy video tensor)
    print("\nRunning demo...")
    
    # Create dummy video tensor for demonstration
    batch_size = 1
    num_frames = 8
    height, width = 224, 224
    channels = 3
    
    dummy_video = torch.randn(batch_size, num_frames, channels, height, width).to(device)
    
    # Sample questions
    questions = [
        "What is happening in this video?",
        "How many people are visible?",
        "What objects can you see?",
        "Describe the setting or environment"
    ]
    
    print("\nGenerating responses to sample questions:")
    print("-" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        
        try:
            response = model.generate_response(dummy_video, question)
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Demo evaluation
    print("\n" + "=" * 50)
    print("EVALUATION DEMO")
    print("=" * 50)
    
    # Sample predictions and references
    predictions = [
        "The video shows various activities and scenes.",
        "There are several people visible in the video.",
        "Several objects are visible throughout the video.",
        "The video takes place in a specific environment."
    ]
    
    references = [
        "The video contains multiple scenes with different activities.",
        "Multiple people can be seen in the video content.",
        "Various objects appear throughout the video sequence.",
        "The video is set in a particular location or environment."
    ]
    
    questions_eval = [
        "What is happening in this video?",
        "How many people are visible?",
        "What objects can you see?",
        "Describe the setting or environment"
    ]
    
    # Create evaluation report
    report = create_evaluation_report(
        predictions=predictions,
        references=references,
        questions=questions_eval,
        model_name="SimpleVideoDialogue"
    )
    
    print(report)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo run the interactive demo:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run Streamlit demo: streamlit run demo/streamlit_demo.py")
    print("\nTo train a model:")
    print("1. Create sample data: python src/train/trainer.py --config configs/default.yaml --create_sample_data")
    print("2. Train model: python src/train/trainer.py --config configs/default.yaml")


if __name__ == "__main__":
    import torch
    main()
