#!/usr/bin/env python3
"""Script to run the video dialogue system demo."""

import subprocess
import sys
import os


def main():
    """Run the Streamlit demo."""
    print("Starting Video Dialogue System Demo...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Run the demo
    demo_path = os.path.join(os.path.dirname(__file__), "demo", "streamlit_demo.py")
    
    if not os.path.exists(demo_path):
        print(f"Error: Demo file not found at {demo_path}")
        sys.exit(1)
    
    print(f"Running demo from: {demo_path}")
    print("The demo will open in your web browser.")
    print("Press Ctrl+C to stop the demo.")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", demo_path])
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
