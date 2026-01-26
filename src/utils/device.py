"""Device management utilities for video dialogue systems."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def get_device() -> torch.device:
    """Get the best available device with fallback priority: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_mixed_precision_dtype() -> torch.dtype:
    """Get the appropriate mixed precision dtype based on device.
    
    Returns:
        torch.dtype: The dtype for mixed precision training.
    """
    if torch.cuda.is_available():
        # Use bfloat16 for newer GPUs, fallback to float16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS supports float16
        return torch.float16
    else:
        # CPU doesn't benefit from mixed precision
        return torch.float32


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    max_batch_size: int = 32,
    dtype: torch.dtype = torch.float32
) -> int:
    """Find the optimal batch size for the given model and device.
    
    Args:
        model: The model to test.
        input_shape: Shape of input tensor (batch_size will be replaced).
        device: Device to test on.
        max_batch_size: Maximum batch size to test.
        dtype: Data type for testing.
        
    Returns:
        int: Optimal batch size.
    """
    model = model.to(device)
    model.eval()
    
    optimal_batch_size = 1
    
    for batch_size in range(1, max_batch_size + 1):
        try:
            # Create dummy input
            dummy_input = torch.randn((batch_size,) + input_shape[1:], dtype=dtype, device=device)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            optimal_batch_size = batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            else:
                raise e
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return optimal_batch_size


def print_device_info() -> None:
    """Print detailed device information."""
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA available: No")
    
    # MPS info
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS available: Yes")
    else:
        print("MPS available: No")
    
    # CPU info
    print(f"CPU cores: {torch.get_num_threads()}")
    
    print("=" * 50)
