#!/usr/bin/env python3
"""
Analyze COLMAP camera orbit parameters for animation reference
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

def analyze_orbit(data_dir: str):
    """Analyze camera orbit around scene center"""
    parser = Parser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8,
    )
    
    dataset = Dataset(parser, split="train")
    
    # Calculate scene center (approximate)
    positions = []
    for i in range(len(dataset)):
        data = dataset[i]
        camtoworld = data["camtoworld"]
        position = camtoworld[:3, 3].numpy()
        positions.append(position)
    
    positions = np.array(positions)
    scene_center = positions.mean(axis=0)
    
    print("📍 Scene Analysis:")
    print(f"  Scene center (approx): {scene_center}")
    print(f"  Camera positions range:")
    print(f"    X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"    Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"    Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
    print()
    
    # Calculate distances from center
    distances = np.linalg.norm(positions - scene_center, axis=1)
    print("📏 Camera distances from center:")
    print(f"  Mean: {distances.mean():.2f}")
    print(f"  Min: {distances.min():.2f}")
    print(f"  Max: {distances.max():.2f}")
    print(f"  Std: {distances.std():.2f}")
    print()
    
    # Analyze camera 15, 45, 60
    for idx in [15, 45, 60]:
        if idx < len(dataset):
            data = dataset[idx]
            camtoworld = data["camtoworld"].numpy()
            position = camtoworld[:3, 3]
            distance = np.linalg.norm(position - scene_center)
            
            # Camera forward direction (negative Z axis in camera space)
            forward = -camtoworld[:3, 2]
            
            print(f"Camera {idx}:")
            print(f"  Position: {position}")
            print(f"  Distance from center: {distance:.2f}")
            print(f"  Height (Z): {position[2]:.2f}")
            print(f"  Forward direction: {forward}")
            print()
    
    # Recommend orbit parameters for cat at [-0.8, 0, 0]
    cat_position = np.array([-0.8, 0, 0])
    print(f"🐱 Cat position: {cat_position}")
    print(f"  Distance from scene center: {np.linalg.norm(cat_position - scene_center):.2f}")
    print()
    
    # Recommend orbit radius similar to camera distances
    recommended_radius = distances.mean() * 0.8  # Slightly closer for better view
    print(f"💡 Recommended orbit parameters:")
    print(f"  Radius: {recommended_radius:.2f} (80% of mean camera distance)")
    print(f"  Height: {positions[:, 2].mean():.2f} (mean camera height)")
    print(f"  Or slightly below cat: {cat_position[2] - 0.2:.2f}")

if __name__ == "__main__":
    data_dir = "/teamspace/lightning_storage/final_project_3DGS/data/scene_colmap"
    analyze_orbit(data_dir)
