"""Streamlit demo for video dialogue systems."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from typing import Optional, Tuple
import time

from src.models.video_dialogue import create_video_dialogue_model
from src.utils.video_utils import load_video_frames, frames_to_tensor
from src.utils.device import get_device, print_device_info


class VideoDialogueDemo:
    """Demo application for video dialogue systems."""
    
    def __init__(self):
        """Initialize the demo."""
        self.device = get_device()
        self.model = None
        self.model_loaded = False
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "video_uploaded" not in st.session_state:
            st.session_state.video_uploaded = False
        if "video_frames" not in st.session_state:
            st.session_state.video_frames = None
    
    def load_model(self, model_type: str = "simple") -> bool:
        """Load the video dialogue model.
        
        Args:
            model_type: Type of model to load.
            
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            config = {
                "vocab_size": 10000,
                "hidden_dim": 256,
                "num_layers": 3
            }
            
            self.model = create_video_dialogue_model(model_type, config)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def process_video(self, video_file) -> Optional[torch.Tensor]:
        """Process uploaded video file.
        
        Args:
            video_file: Uploaded video file.
            
        Returns:
            Processed video tensor or None if error.
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_file.read())
                tmp_path = tmp_file.name
            
            # Load video frames
            frames = load_video_frames(
                tmp_path,
                max_frames=32,
                target_size=(224, 224)
            )
            
            if not frames:
                st.error("Failed to load video frames")
                return None
            
            # Convert to tensor
            video_tensor = frames_to_tensor(frames)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return video_tensor
            
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return None
    
    def generate_response(self, video_tensor: torch.Tensor, question: str) -> str:
        """Generate response for video dialogue.
        
        Args:
            video_tensor: Processed video tensor.
            question: Question about the video.
            
        Returns:
            Generated response.
        """
        if not self.model_loaded:
            return "Model not loaded. Please load a model first."
        
        try:
            with torch.no_grad():
                response = self.model.generate_response(video_tensor, question)
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def run(self):
        """Run the demo application."""
        st.set_page_config(
            page_title="Video Dialogue System",
            page_icon="🎥",
            layout="wide"
        )
        
        st.title("🎥 Video Dialogue System")
        st.markdown("Upload a video and ask questions about its content!")
        
        # Sidebar for model selection
        with st.sidebar:
            st.header("Model Configuration")
            
            model_type = st.selectbox(
                "Select Model Type",
                ["simple", "clip", "blip"],
                help="Choose the video dialogue model to use"
            )
            
            if st.button("Load Model"):
                with st.spinner("Loading model..."):
                    success = self.load_model(model_type)
                    if success:
                        st.success(f"Model '{model_type}' loaded successfully!")
                    else:
                        st.error("Failed to load model")
            
            if self.model_loaded:
                st.success("✅ Model Ready")
            else:
                st.warning("⚠️ Please load a model first")
            
            # Device information
            with st.expander("Device Info"):
                st.text("Device information will be shown here")
                if st.button("Show Device Info"):
                    print_device_info()
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📹 Video Upload")
            
            # Video upload
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file to analyze"
            )
            
            if uploaded_video is not None:
                st.video(uploaded_video)
                
                # Process video
                if st.button("Process Video"):
                    with st.spinner("Processing video..."):
                        video_tensor = self.process_video(uploaded_video)
                        
                        if video_tensor is not None:
                            st.session_state.video_frames = video_tensor
                            st.session_state.video_uploaded = True
                            st.success("Video processed successfully!")
                            
                            # Show video info
                            st.info(f"Video tensor shape: {video_tensor.shape}")
                        else:
                            st.error("Failed to process video")
        
        with col2:
            st.header("💬 Video Dialogue")
            
            if not st.session_state.video_uploaded:
                st.info("Please upload and process a video first")
            elif not self.model_loaded:
                st.info("Please load a model first")
            else:
                # Chat interface
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Question input
                if question := st.chat_input("Ask a question about the video..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    with st.chat_message("user"):
                        st.markdown(question)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Generating response..."):
                            response = self.generate_response(
                                st.session_state.video_frames,
                                question
                            )
                            st.markdown(response)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Example questions
        st.header("💡 Example Questions")
        
        example_questions = [
            "What is happening in this video?",
            "How many people are visible?",
            "What objects can you see?",
            "Describe the setting or environment",
            "What actions are being performed?",
            "What is the mood or atmosphere?",
            "What time of day does this appear to be?",
            "What sounds might you expect to hear?"
        ]
        
        cols = st.columns(4)
        for i, question in enumerate(example_questions):
            with cols[i % 4]:
                if st.button(question, key=f"example_{i}"):
                    if st.session_state.video_uploaded and self.model_loaded:
                        # Add to chat
                        st.session_state.messages.append({"role": "user", "content": question})
                        
                        with st.spinner("Generating response..."):
                            response = self.generate_response(
                                st.session_state.video_frames,
                                question
                            )
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    else:
                        st.warning("Please upload a video and load a model first")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main function to run the demo."""
    demo = VideoDialogueDemo()
    demo.run()


if __name__ == "__main__":
    main()
