#!/usr/bin/env python3
"""Script to create sample data and test the system."""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def main():
    """Main function for setup and testing."""
    parser = argparse.ArgumentParser(description="Setup and test video dialogue system")
    parser.add_argument("--create-data", action="store_true", help="Create sample dataset")
    parser.add_argument("--test-system", action="store_true", help="Test the system")
    parser.add_argument("--run-demo", action="store_true", help="Run the demo")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    if args.all:
        args.create_data = True
        args.test_system = True
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    if args.create_data:
        print("Creating sample dataset...")
        print("=" * 40)
        
        try:
            # Create data directory
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Run data creation
            cmd = [
                sys.executable, 
                str(project_root / "src" / "train" / "trainer.py"),
                "--config", str(project_root / "configs" / "default.yaml"),
                "--create_sample_data"
            ]
            
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Sample dataset created successfully!")
                print(f"Data saved to: {data_dir}")
            else:
                print("❌ Error creating sample dataset:")
                print(result.stderr)
                return 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    if args.test_system:
        print("\nTesting the system...")
        print("=" * 40)
        
        try:
            # Run main.py to test the system
            cmd = [sys.executable, str(project_root / "main.py")]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ System test completed successfully!")
                print("\nTest output:")
                print(result.stdout)
            else:
                print("❌ Error testing system:")
                print(result.stderr)
                return 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    if args.run_demo:
        print("\nStarting demo...")
        print("=" * 40)
        
        try:
            # Run the demo script
            demo_script = project_root / "scripts" / "run_demo.py"
            subprocess.run([sys.executable, str(demo_script)])
            
        except Exception as e:
            print(f"❌ Error running demo: {e}")
            return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python scripts/run_demo.py")
    print("2. Train a model: python src/train/trainer.py --config configs/default.yaml")
    print("3. Test the system: python main.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
