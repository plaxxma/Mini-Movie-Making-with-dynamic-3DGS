#!/usr/bin/env python3
"""
Inspect COLMAP camera parameters for animation planning
"""
import sys
from pathlib import Path
import numpy as np

# Add gsplat to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
gsplat_path = project_root / "gsplat_repo" / "examples"
sys.path.insert(0, str(gsplat_path))

from datasets.colmap import Parser, Dataset

def inspect_cameras(data_dir: str, num_to_show: int = 10):
    """Inspect COLMAP camera parameters"""
    print(f"Loading COLMAP data from: {data_dir}")
    print("="*80)
    
    # Load with same settings as training
    parser = Parser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8,
    )
    
    # Create dataset to access camera parameters
    dataset = Dataset(parser, split="train")
    
    print(f"Total images: {len(dataset)}")
    print()
    
    # Get intrinsics from first image
    sample = dataset[0]
    K = sample["K"]
    height, width = sample["image"].shape[:2]
    
    print("📷 Camera Intrinsics:")
    print(f"  Width: {width}")
    print(f"  Height: {height}")
    print(f"  K matrix:\n{K}")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print()
    
    # Show some camera extrinsics
    print("📍 Camera Extrinsics (sample):")
    print(f"  Showing first {num_to_show}, middle {num_to_show}, and last {num_to_show} cameras")
    print()
    
    total = len(dataset)
    
    # First N
    print("=" * 80)
    print("FIRST CAMERAS:")
    print("=" * 80)
    for i in range(min(num_to_show, total)):
        data = dataset[i]
        camtoworld = data["camtoworld"]  # [4, 4]
        position = camtoworld[:3, 3]
        
        print(f"\nCamera {i}:")
        print(f"  Position (world): {position.numpy()}")
        print(f"  camtoworld:\n{camtoworld.numpy()}")
    
    # Middle N
    print("\n" + "=" * 80)
    print("MIDDLE CAMERAS:")
    print("=" * 80)
    mid_start = max(0, total // 2 - num_to_show // 2)
    for i in range(mid_start, min(mid_start + num_to_show, total)):
        data = dataset[i]
        camtoworld = data["camtoworld"]
        position = camtoworld[:3, 3]
        
        print(f"\nCamera {i}:")
        print(f"  Position (world): {position.numpy()}")
        print(f"  camtoworld:\n{camtoworld.numpy()}")
    
    # Last N
    print("\n" + "=" * 80)
    print("LAST CAMERAS:")
    print("=" * 80)
    for i in range(max(0, total - num_to_show), total):
        data = dataset[i]
        camtoworld = data["camtoworld"]
        position = camtoworld[:3, 3]
        
        print(f"\nCamera {i}:")
        print(f"  Position (world): {position.numpy()}")
        print(f"  camtoworld:\n{camtoworld.numpy()}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total cameras: {total}")
    print(f"Index range: 0 to {total - 1}")
    print()
    print("💡 Recommendations:")
    print("  - For start camera (showing approach): check cameras 0-20")
    print("  - For center camera (main view): check cameras around middle")
    print(f"  - Middle index: ~{total // 2}")
    print("  - For reference: camera 15 exists" if 15 < total else "  - Warning: camera 15 doesn't exist")
    print("  - For reference: camera 45 exists" if 45 < total else "  - Warning: camera 45 doesn't exist")

if __name__ == "__main__":
    data_dir = "/teamspace/lightning_storage/final_project_3DGS/data/scene_colmap"
    inspect_cameras(data_dir, num_to_show=10)
