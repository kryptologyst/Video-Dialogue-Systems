"""Video processing utilities for dialogue systems."""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Tuple, Union
import decord
from decord import VideoReader


def load_video_frames(
    video_path: str,
    max_frames: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    fps: Optional[float] = None
) -> List[np.ndarray]:
    """Load and preprocess video frames.
    
    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.
        target_size: Target size for frame resizing (height, width).
        fps: Target FPS for frame sampling. If None, uses uniform sampling.
        
    Returns:
        List of preprocessed frames as numpy arrays.
    """
    try:
        # Use decord for efficient video loading
        vr = VideoReader(video_path)
        total_frames = len(vr)
        
        if fps is not None:
            # Sample frames based on target FPS
            video_fps = vr.get_avg_fps()
            frame_indices = np.linspace(0, total_frames - 1, 
                                      int(total_frames * fps / video_fps), 
                                      dtype=int)
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, 
                                      min(max_frames, total_frames), 
                                      dtype=int)
        
        # Limit to max_frames
        frame_indices = frame_indices[:max_frames]
        
        # Extract frames
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Resize frames
        resized_frames = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            resized_frame = cv2.resize(frame_rgb, target_size[::-1])  # OpenCV uses (width, height)
            resized_frames.append(resized_frame)
        
        return resized_frames
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        # Fallback to OpenCV
        return _load_video_frames_opencv(video_path, max_frames, target_size)


def _load_video_frames_opencv(
    video_path: str,
    max_frames: int = 32,
    target_size: Tuple[int, int] = (224, 224)
) -> List[np.ndarray]:
    """Fallback video loading using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 
                              min(max_frames, total_frames), 
                              dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, target_size[::-1])
            frames.append(resized_frame)
    
    cap.release()
    return frames


def frames_to_tensor(
    frames: List[np.ndarray],
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Convert frames to PyTorch tensor.
    
    Args:
        frames: List of frames as numpy arrays.
        normalize: Whether to normalize the frames.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        
    Returns:
        Tensor of shape (T, C, H, W) where T is number of frames.
    """
    # Convert to tensor and rearrange dimensions
    tensor_frames = torch.stack([torch.from_numpy(frame).permute(2, 0, 1) for frame in frames])
    
    # Convert to float and normalize
    tensor_frames = tensor_frames.float() / 255.0
    
    if normalize:
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        tensor_frames = (tensor_frames - mean) / std
    
    return tensor_frames


def extract_optical_flow(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Extract optical flow between consecutive frames.
    
    Args:
        frames: List of frames as numpy arrays.
        
    Returns:
        List of optical flow maps.
    """
    flows = []
    
    for i in range(len(frames) - 1):
        # Convert to grayscale
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        # Alternative: Dense optical flow
        flow_dense = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert flow to RGB visualization
        hsv = np.zeros((flow_dense.shape[0], flow_dense.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow_dense[..., 0], flow_dense[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        flows.append(flow_rgb)
    
    return flows


def create_video_summary(
    frames: List[np.ndarray],
    method: str = "uniform",
    num_frames: int = 8
) -> List[np.ndarray]:
    """Create a video summary by selecting key frames.
    
    Args:
        frames: List of input frames.
        method: Method for frame selection ('uniform', 'random', 'first', 'last').
        num_frames: Number of frames to select.
        
    Returns:
        List of selected frames.
    """
    if len(frames) <= num_frames:
        return frames
    
    if method == "uniform":
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    elif method == "random":
        indices = np.random.choice(len(frames), num_frames, replace=False)
        indices.sort()
    elif method == "first":
        indices = list(range(num_frames))
    elif method == "last":
        indices = list(range(len(frames) - num_frames, len(frames)))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return [frames[i] for i in indices]


def detect_scene_changes(frames: List[np.ndarray], threshold: float = 0.3) -> List[int]:
    """Detect scene changes in video frames.
    
    Args:
        frames: List of frames.
        threshold: Threshold for scene change detection.
        
    Returns:
        List of frame indices where scene changes occur.
    """
    scene_changes = [0]  # First frame is always a scene start
    
    for i in range(1, len(frames)):
        # Convert to grayscale
        gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram difference
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        if correlation < threshold:
            scene_changes.append(i)
    
    return scene_changes
