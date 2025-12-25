#!/usr/bin/env python3
"""
Scene-Object Composition Script for 3DGS Mini-Movie Project

Combines trained scene and object models with rigid transformations.
Uses gsplat rendering functions for depth-aware composition.

Usage:
    python scripts/compose_scene.py \
        --scene trained_models/scene_training \
        --objects trained_models/cat_training trained_models/tiger_training \
        --output composed_models/test_composition \
        --test-render
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Add gsplat to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
gsplat_path = project_root / "gsplat_repo" / "examples"
sys.path.insert(0, str(gsplat_path))

from datasets.colmap import Dataset, Parser
from gsplat.rendering import rasterization


class ComposedSplats:
    """Composed 3D Gaussian Splats from multiple models"""
    
    def __init__(self):
        self.means = None
        self.quats = None
        self.scales = None
        self.opacities = None
        self.sh0 = None
        self.shN = None
        
    def add_splats(
        self,
        splats: Dict[str, torch.Tensor],
        translation: List[float] = [0, 0, 0],
        rotation: List[float] = [0, 0, 0],  # Euler angles XYZ in degrees
        scale: float = 1.0,
        remove_bright: bool = False,
        brightness_threshold: float = 0.8,
        center_first: bool = False,  # Center to origin before transform
        remove_floor: bool = False,  # Remove floor Gaussians
        floor_threshold: float = -0.3,  # Y threshold for floor removal (after centering)
        filter_outliers: bool = False,  # Use density-based outlier filtering
        density_radius: float = 0.5,  # Radius for density calculation
        min_density: int = 50,  # Minimum neighbors to keep [UNUSED]
        percentile: int = 80,  # Keep N% closest to center
        min_opacity: float = 0.0,  # Minimum opacity threshold (0-1, sigmoid space)
        color_filter_outside_core: bool = False,  # Apply color filter only outside core percentile
        core_percentile: int = 20,  # Core region to preserve (no color filtering)
        max_saturation: float = 0.2,  # Remove low-saturation (gray) Gaussians outside core
    ):
        """
        Add splats from a model with rigid transformation
        
        Args:
            splats: Dictionary with keys 'means', 'quats', 'scales', 'opacities', 'sh0', 'shN'
            translation: [x, y, z] translation vector
            rotation: [rx, ry, rz] Euler angles in degrees (XYZ order)
            scale: Uniform scale factor
            remove_bright: Remove bright Gaussians (e.g., checkered pattern) [DEPRECATED]
            brightness_threshold: Brightness threshold for removal (0-1)
            center_first: If True, center object to origin before applying transforms
            remove_floor: If True, remove Gaussians below floor_threshold (Y axis)
            floor_threshold: Y value below which Gaussians are considered floor
            filter_outliers: Use density-based clustering to remove sparse outliers
            density_radius: Radius for counting neighbors in density calculation
            min_density: Minimum number of neighbors to keep a Gaussian
        """
        # Extract parameters
        means = splats["means"].detach().cpu().numpy()  # [N, 3]
        quats = splats["quats"].detach().cpu().numpy()  # [N, 4]
        scales = splats["scales"].detach().cpu().numpy()  # [N, 3] (log space)
        opacities = splats["opacities"].detach().cpu().numpy()  # [N, 1]
        sh0 = splats["sh0"].detach().cpu().numpy()  # [N, 1, 3]
        shN = splats["shN"].detach().cpu().numpy() if "shN" in splats else None  # [N, K, 3]
        
        original_count = len(means)
        
        # Center object to origin first (important for proper placement)
        if center_first:
            center = means.mean(axis=0)
            means = means - center
            print(f"  🔹 Centered object (moved by {-center.round(3)})")
            print(f"     Mean: {means.mean(axis=0).round(3)}, Std: {means.std(axis=0).round(3)}")
            
            # DENSITY-BASED OUTLIER FILTERING
            if filter_outliers:
                print(f"  🔹 Distance-based outlier filtering...")
                
                # Simple approach: remove Gaussians far from center
                # Much faster and memory-efficient than full density calculation
                distances = np.linalg.norm(means, axis=1)
                
                # Use percentile-based threshold to keep dense core
                distance_threshold = np.percentile(distances, percentile)
                
                # Also apply absolute distance limit
                max_distance = density_radius * 10  # 10x the density radius
                distance_threshold = min(distance_threshold, max_distance)
                
                keep_mask = distances < distance_threshold
                
                removed_outliers = (~keep_mask).sum()
                print(f"     Distance threshold: {distance_threshold:.2f}")
                print(f"     Removed {removed_outliers}/{len(means)} distant Gaussians")
                print(f"     Kept {keep_mask.sum()}/{len(means)} core Gaussians")
                
                means = means[keep_mask]
                quats = quats[keep_mask]
                scales = scales[keep_mask]
                opacities = opacities[keep_mask]
                sh0 = sh0[keep_mask]
                if shN is not None:
                    shN = shN[keep_mask]
            
            # OPACITY FILTERING (remove transparent/semi-transparent Gaussians)
            if min_opacity > 0:
                # opacities are in logit space, convert to sigmoid space for threshold
                import torch
                opacity_sigmoid = torch.sigmoid(torch.from_numpy(opacities)).numpy()
                opacity_mask = opacity_sigmoid.squeeze() >= min_opacity
                
                removed_transparent = (~opacity_mask).sum()
                print(f"  🔹 Opacity filter: removed {removed_transparent}/{len(means)} transparent Gaussians (opacity < {min_opacity:.2f})")
                
                means = means[opacity_mask]
                quats = quats[opacity_mask]
                scales = scales[opacity_mask]
                opacities = opacities[opacity_mask]
                sh0 = sh0[opacity_mask]
                if shN is not None:
                    shN = shN[opacity_mask]
            
            # COLOR FILTERING (only outside core region)
            if color_filter_outside_core and max_saturation > 0:
                print(f"  🔹 Color filter (outside core {core_percentile}%)...")
                
                # Calculate distances from center
                distances = np.linalg.norm(means, axis=1)
                core_threshold = np.percentile(distances, core_percentile)
                
                # Identify core (protected) and outer regions
                is_core = distances < core_threshold
                
                # Calculate color saturation for ALL Gaussians
                # sh0 shape: [N, 1, 3] - convert to RGB
                colors_rgb = sh0.squeeze(1)  # [N, 3]
                
                # Calculate saturation: (max - min) / max
                color_max = colors_rgb.max(axis=1)
                color_min = colors_rgb.min(axis=1)
                saturation = np.where(color_max > 0, (color_max - color_min) / (color_max + 1e-8), 0)
                
                # Remove low-saturation (gray) Gaussians ONLY outside core
                is_colorful = saturation > max_saturation
                keep_mask = is_core | is_colorful  # Keep if core OR colorful
                
                removed_gray = (~keep_mask).sum()
                print(f"     Core region: {is_core.sum()}/{len(means)} Gaussians (protected)")
                print(f"     Removed {removed_gray} low-saturation Gaussians outside core")
                
                means = means[keep_mask]
                quats = quats[keep_mask]
                scales = scales[keep_mask]
                opacities = opacities[keep_mask]
                sh0 = sh0[keep_mask]
                if shN is not None:
                    shN = shN[keep_mask]
            
            # FLOOR REMOVAL (optional, after other filtering)
            if remove_floor:
                floor_mask = means[:, 1] > floor_threshold  # Y > threshold
                removed_floor = (~floor_mask).sum()
                print(f"  🔹 Floor filter: removed {removed_floor}/{len(means)} Gaussians (Y < {floor_threshold})")
                
                means = means[floor_mask]
                quats = quats[floor_mask]
                scales = scales[floor_mask]
                opacities = opacities[floor_mask]
                sh0 = sh0[floor_mask]
                if shN is not None:
                    shN = shN[floor_mask]
            
            if filter_outliers or min_opacity > 0 or color_filter_outside_core or remove_floor:
                print(f"  🔹 Total: {original_count} → {len(means)} Gaussians")
        
        # Apply transformations
        # 1. Scale
        print(f"  🔹 Applying scale: {scale}")
        print(f"     Before scale: mean magnitude={np.linalg.norm(means, axis=1).mean():.3f}")
        means = means * scale
        scales = scales + np.log(scale)  # scales are in log space
        print(f"     After scale: mean magnitude={np.linalg.norm(means, axis=1).mean():.3f}")
        
        # 2. Rotate
        rotation_angles = np.array(rotation)
        if np.any(rotation_angles != 0):
            print(f"  🔹 Applying rotation: {rotation_angles}°")
            rot_matrix = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix()
            means = means @ rot_matrix.T  # Apply rotation to positions
            
            # Rotate quaternions
            rot_quat = R.from_euler('xyz', rotation_angles, degrees=True).as_quat()
            original_rot = R.from_quat(quats)
            new_rot = R.from_quat(rot_quat) * original_rot
            quats = new_rot.as_quat()
            print(f"     After rotation: mean position={means.mean(axis=0).round(3)}")
        
        # 3. Translate
        print(f"  🔹 Applying translation: {translation}")
        means = means + np.array(translation)
        print(f"     Final position: mean={means.mean(axis=0).round(3)}, range=({means.min(axis=0).round(3)}, {means.max(axis=0).round(3)})")
        
        # Concatenate to existing data
        if self.means is None:
            # First model (scene)
            self.means = means
            self.quats = quats
            self.scales = scales
            self.opacities = opacities
            self.sh0 = sh0
            self.shN = shN
        else:
            # Add objects
            self.means = np.concatenate([self.means, means], axis=0)
            self.quats = np.concatenate([self.quats, quats], axis=0)
            self.scales = np.concatenate([self.scales, scales], axis=0)
            self.opacities = np.concatenate([self.opacities, opacities], axis=0)
            self.sh0 = np.concatenate([self.sh0, sh0], axis=0)
            if self.shN is not None and shN is not None:
                self.shN = np.concatenate([self.shN, shN], axis=0)
    
    def to_dict(self, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Convert to dictionary of tensors for rendering"""
        splats = {
            "means": torch.from_numpy(self.means).float().to(device),
            "quats": torch.from_numpy(self.quats).float().to(device),
            "scales": torch.from_numpy(self.scales).float().to(device),
            "opacities": torch.from_numpy(self.opacities).float().to(device),
            "sh0": torch.from_numpy(self.sh0).float().to(device),
        }
        if self.shN is not None:
            splats["shN"] = torch.from_numpy(self.shN).float().to(device)
        return splats
    
    def save_ply(self, path: str):
        """Save as PLY file using gsplat's export_splats"""
        from gsplat import export_splats
        
        # Convert to tensors on CPU
        means = torch.from_numpy(self.means).float()
        scales = torch.from_numpy(self.scales).float()
        quats = torch.from_numpy(self.quats).float()
        opacities = torch.from_numpy(self.opacities).float().squeeze()
        sh0 = torch.from_numpy(self.sh0).float()
        
        # Handle shN (can be None)
        if self.shN is not None:
            shN = torch.from_numpy(self.shN).float()
        else:
            # Create dummy shN if not present (required by export_splats)
            shN = torch.zeros(len(means), 15, 3)  # 15 = (sh_degree+1)^2 - 1 for degree 3
        
        # Export to PLY
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=str(path),
        )
        print(f"Saved composed model: {path}")
    
    def save_filtered_object(self, path: str, object_idx: int = 0):
        """Save individual filtered object as PLY (for reuse)"""
        from gsplat import export_splats
        
        # This would require tracking which Gaussians belong to which object
        # For now, save the full composed model
        print(f"Note: Saving full composed model to {path}")
        self.save_ply(path)


def save_filtered_splats(
    splats: Dict[str, torch.Tensor],
    output_path: Path,
    filter_params: Dict,
):
    """
    Save filtered object splats to PLY file for reuse
    
    Args:
        splats: Input splats dictionary
        output_path: Path to save filtered PLY
        filter_params: Filtering parameters used
    """
    from gsplat import export_splats
    
    # Apply same filtering as in add_splats
    means = splats["means"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    opacities = splats["opacities"].detach().cpu().numpy()
    sh0 = splats["sh0"].detach().cpu().numpy()
    shN = splats["shN"].detach().cpu().numpy() if "shN" in splats else None
    
    # Center
    if filter_params.get("center_first", False):
        center = means.mean(axis=0)
        means = means - center
        print(f"Centered object: moved by {-center.round(3)}")
        
        # Density filtering
        if filter_params.get("filter_outliers", False):
            density_radius = filter_params.get("density_radius", 0.5)
            
            # Distance-based filtering (fast and memory-efficient)
            distances = np.linalg.norm(means, axis=1)
            threshold_percentile = 80
            distance_threshold = np.percentile(distances, threshold_percentile)
            max_distance = density_radius * 10
            distance_threshold = min(distance_threshold, max_distance)
            
            keep_mask = distances < distance_threshold
            
            print(f"Distance filtering: {keep_mask.sum()}/{len(means)} kept (threshold={distance_threshold:.2f})")
            
            means = means[keep_mask]
            quats = quats[keep_mask]
            scales = scales[keep_mask]
            opacities = opacities[keep_mask]
            sh0 = sh0[keep_mask]
            if shN is not None:
                shN = shN[keep_mask]
    
    # Convert to tensors
    means_t = torch.from_numpy(means).float()
    scales_t = torch.from_numpy(scales).float()
    quats_t = torch.from_numpy(quats).float()
    opacities_t = torch.from_numpy(opacities).float().squeeze()
    sh0_t = torch.from_numpy(sh0).float()
    
    if shN is not None:
        shN_t = torch.from_numpy(shN).float()
    else:
        shN_t = torch.zeros(len(means), 15, 3)
    
    # Save
    export_splats(
        means=means_t,
        scales=scales_t,
        quats=quats_t,
        opacities=opacities_t,
        sh0=sh0_t,
        shN=shN_t,
        format="ply",
        save_to=str(output_path),
    )
    print(f"✅ Saved filtered object: {output_path}")


def load_checkpoint(ckpt_path: Path, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load checkpoint and extract splats"""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    splats = ckpt["splats"]
    num_gaussians = splats["means"].shape[0]
    print(f"  Loaded {num_gaussians:,} Gaussians")
    
    return splats


def compose_scene(
    scene_dir: Path,
    object_dirs: List[Path],
    object_transforms: List[Dict],
    output_dir: Path,
    remove_bright: bool = False,
    brightness_threshold: float = 0.8,
    save_filtered_objects: bool = False,  # Save filtered objects as separate PLY files
) -> ComposedSplats:
    """
    Compose scene with objects
    
    Args:
        scene_dir: Path to scene training directory
        object_dirs: List of paths to object training directories
        object_transforms: List of transform dicts with 'translation', 'rotation', 'scale'
        output_dir: Output directory
        remove_bright: Remove bright Gaussians from objects [DEPRECATED]
        brightness_threshold: Brightness threshold (0-1)
        save_filtered_objects: Save filtered objects as individual PLY files
    
    Returns:
        ComposedSplats object
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scene
    print(f"\n📦 Loading scene from {scene_dir}")
    scene_ckpt = scene_dir / "ckpts" / "ckpt_29999_rank0.pt"
    if not scene_ckpt.exists():
        # Try to find any checkpoint
        ckpt_files = sorted((scene_dir / "ckpts").glob("ckpt_*_rank0.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {scene_dir}/ckpts")
        scene_ckpt = ckpt_files[-1]
    
    scene_splats = load_checkpoint(scene_ckpt)
    
    # Create composed model
    composed = ComposedSplats()
    
    # Add scene (no transformation)
    print("Adding scene...")
    composed.add_splats(scene_splats)
    
    # Add objects
    for i, (obj_dir, transform) in enumerate(zip(object_dirs, object_transforms)):
        print(f"\n📦 Loading object {i+1} from {obj_dir}")
        
        obj_ckpt = obj_dir / "ckpts" / "ckpt_29999_rank0.pt"
        if not obj_ckpt.exists():
            ckpt_files = sorted((obj_dir / "ckpts").glob("ckpt_*_rank0.pt"))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint found in {obj_dir}/ckpts")
            obj_ckpt = ckpt_files[-1]
        
        obj_splats = load_checkpoint(obj_ckpt)
        
        translation = transform.get("translation", [0, 0, 0])
        rotation = transform.get("rotation", [0, 0, 0])
        scale = transform.get("scale", 1.0)
        
        # Filtering options
        filter_outliers = transform.get("filter_outliers", False)
        density_radius = transform.get("density_radius", 0.5)
        min_density = transform.get("min_density", 50)
        percentile = transform.get("percentile", 80)
        min_opacity = transform.get("min_opacity", 0.0)
        color_filter_outside_core = transform.get("color_filter_outside_core", False)
        core_percentile = transform.get("core_percentile", 20)
        max_saturation = transform.get("max_saturation", 0.2)
        remove_floor = transform.get("remove_floor", False)
        floor_threshold = transform.get("floor_threshold", -0.3)
        
        print(f"  Translation: {translation}")
        print(f"  Rotation: {rotation}°")
        print(f"  Scale: {scale}")
        if filter_outliers:
            print(f"  Density filtering: enabled (radius={density_radius}, percentile={percentile})")
        if min_opacity > 0:
            print(f"  Opacity filtering: enabled (min_opacity={min_opacity})")
        if color_filter_outside_core:
            print(f"  Color filtering: enabled (core={core_percentile}%, max_saturation={max_saturation})")
        if remove_floor:
            print(f"  Floor removal: enabled (threshold={floor_threshold})")
        
        composed.add_splats(
            obj_splats,
            translation=translation,
            rotation=rotation,
            scale=scale,
            remove_bright=False,  # Deprecated
            brightness_threshold=0.8,
            center_first=True,  # Center object before transforming
            remove_floor=remove_floor,
            floor_threshold=floor_threshold,
            filter_outliers=filter_outliers,
            density_radius=density_radius,
            min_density=min_density,
            percentile=percentile,
            min_opacity=min_opacity,
            color_filter_outside_core=color_filter_outside_core,
            core_percentile=core_percentile,
            max_saturation=max_saturation,
        )
        
        # Save filtered object separately if requested
        if save_filtered_objects and (filter_outliers or remove_floor):
            obj_name = obj_dir.name  # e.g., "cat_training"
            filtered_path = output_dir / f"{obj_name}_filtered.ply"
            save_filtered_splats(
                obj_splats,
                filtered_path,
                {
                    "center_first": True,
                    "filter_outliers": filter_outliers,
                    "density_radius": density_radius,
                    "min_density": min_density,
                    "remove_floor": remove_floor,
                    "floor_threshold": floor_threshold,
                }
            )
    
    # Save composed model
    print(f"\n💾 Saving composed model to {output_dir}")
    composed.save_ply(output_dir / "composed.ply")
    
    # Save composition metadata
    metadata = {
        "scene": str(scene_dir),
        "objects": [str(d) for d in object_dirs],
        "transforms": object_transforms,
        "num_gaussians": len(composed.means),
        "remove_bright": remove_bright,
        "brightness_threshold": brightness_threshold,
    }
    with open(output_dir / "composition.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Composition complete! Total Gaussians: {len(composed.means):,}")
    
    return composed


def render_test_views(
    composed: ComposedSplats,
    scene_dir: Path,
    output_dir: Path,
    num_views: int = 5,
    camera_indices: Optional[List[int]] = None,
):
    """
    Render test views using scene cameras for verification
    
    Args:
        composed: Composed splats model
        scene_dir: Scene training directory (for camera parameters)
        output_dir: Output directory
        num_views: Number of views to render (if camera_indices not specified)
        camera_indices: Specific camera indices to render
    """
    print(f"\n📷 Rendering test views...")
    
    # Load scene config (use unsafe_load for Python objects)
    cfg_path = scene_dir / "cfg.yml"
    with open(cfg_path, "r") as f:
        cfg = yaml.unsafe_load(f)
    
    data_dir = cfg["data_dir"]
    data_factor = cfg.get("data_factor", 1)
    normalize = cfg.get("normalize_world_space", True)
    
    # Load dataset to get cameras
    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=normalize,
        test_every=8,
    )
    trainset = Dataset(parser, split="train")
    
    # Select cameras to render
    if camera_indices is None:
        step = len(trainset) // num_views
        camera_indices = list(range(0, len(trainset), step))[:num_views]
    
    print(f"Rendering cameras: {camera_indices}")
    
    # Convert to GPU tensors
    splats_dict = composed.to_dict(device="cuda")
    
    # Render each view
    render_dir = output_dir / "test_renders"
    render_dir.mkdir(exist_ok=True)
    
    for idx in tqdm(camera_indices, desc="Rendering"):
        data = trainset[idx]
        
        # Extract camera parameters
        camtoworld = data["camtoworld"].to("cuda")  # [4, 4]
        K = data["K"].to("cuda")  # [3, 3]
        image = data["image"]  # [H, W, 3]
        height, width = image.shape[:2]
        
        # Prepare for rasterization
        camtoworlds = camtoworld.unsqueeze(0)  # [1, 4, 4]
        Ks = K.unsqueeze(0)  # [1, 3, 3]
        
        # Combine SH coefficients: sh0 [N, 1, 3] + shN [N, K, 3] -> [N, K+1, 3]
        sh0 = splats_dict["sh0"]  # [N, 1, 3]
        shN = splats_dict.get("shN", None)
        if shN is not None:
            colors = torch.cat([sh0, shN], dim=1)  # [N, K+1, 3]
            sh_degree = int((colors.shape[1] ** 0.5) - 1)  # K+1 = (degree+1)^2
        else:
            colors = sh0  # [N, 1, 3]
            sh_degree = 0
        
        # Rasterization with SH
        with torch.no_grad():
            renders, alphas, info = rasterization(
                means=splats_dict["means"],
                quats=F.normalize(splats_dict["quats"], dim=-1),
                scales=torch.exp(splats_dict["scales"]),
                opacities=torch.sigmoid(splats_dict["opacities"]),
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [1, 4, 4]
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree,  # Enable SH evaluation
                packed=False,
            )
        
        # Save render
        # renders shape: [1, H, W, 3] -> need [H, W, 3]
        render_img = renders[0].detach().cpu().clamp(0, 1)  # [H, W, 3]
        render_img = (render_img.numpy() * 255).astype(np.uint8)
        
        output_path = render_dir / f"camera_{idx:04d}.png"
        imageio.imwrite(output_path, render_img)
    
    print(f"✅ Test renders saved to: {render_dir}")
    print(f"   Check these images to verify composition!")


def main():
    parser = argparse.ArgumentParser(description="Compose scene with objects")
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to scene training directory (e.g., trained_models/scene_training)",
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        required=True,
        help="Paths to object training directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for composed model",
    )
    parser.add_argument(
        "--remove-bright",
        action="store_true",
        help="Remove bright Gaussians from objects (e.g., checkered patterns)",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=0.8,
        help="Brightness threshold for removal (0-1, default: 0.8)",
    )
    parser.add_argument(
        "--test-render",
        action="store_true",
        help="Render test views for verification",
    )
    parser.add_argument(
        "--num-test-views",
        type=int,
        default=5,
        help="Number of test views to render (default: 5)",
    )
    parser.add_argument(
        "--camera-indices",
        type=int,
        nargs="+",
        help="Specific camera indices to render",
    )
    parser.add_argument(
        "--save-filtered-objects",
        action="store_true",
        help="Save filtered objects as separate PLY files for reuse",
    )
    
    args = parser.parse_args()
    
    # Convert paths
    scene_dir = Path(args.scene).resolve()
    object_dirs = [Path(p).resolve() for p in args.objects]
    output_dir = Path(args.output).resolve()
    
    # Validate inputs
    if not scene_dir.exists():
        print(f"Error: Scene directory not found: {scene_dir}")
        sys.exit(1)
    
    for obj_dir in object_dirs:
        if not obj_dir.exists():
            print(f"Error: Object directory not found: {obj_dir}")
            sys.exit(1)
    
    # Define transformations for each object
    transforms = [
        # Curby - with smart color filtering
        {
            "translation": [-0.8, 0, 0],
            "rotation": [150, 0, 30],
            "scale": 0.5,
            "filter_outliers": True,
            "density_radius": 0.15,
            "percentile": 40,  # Keep 40% by distance
            "min_opacity": 0.0,
            "color_filter_outside_core": True,  # Enable smart color filter
            "core_percentile": 20,  # Protect center 20% (includes eyes)
            "max_saturation": 0.3,  # Remove gray pixels outside core
        },
        # cat
        {
            "translation": [-0.8, 0.0, 0],  # backward, right, above
            "rotation": [180, 0, 60],  # pitch, roll, yaw
            "scale": 0.5,
            "filter_outliers": True,  # Enable distance filtering
            "density_radius": 0.3,  # For distance calculation
            "percentile": 70,  # Keep 60% by distance
            "min_opacity": 0.0,
            "color_filter_outside_core": True,  # Enable smart color filter
            "core_percentile": 65,  # Protect center 40% (includes eyes)
            "max_saturation": 0.4,  # Remove gray pixels outside core
        },

    ]
    
    # Ensure enough transforms
    while len(transforms) < len(object_dirs):
        transforms.append({
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0],
            "scale": 1.0,
        })
    
    transforms = transforms[:len(object_dirs)]
    
    print("="*80)
    print("3DGS Scene-Object Composition")
    print("="*80)
    print(f"Scene: {scene_dir}")
    for i, (obj_dir, transform) in enumerate(zip(object_dirs, transforms)):
        print(f"Object {i+1}: {obj_dir}")
        print(f"  Transform: {transform}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Compose scene
    composed = compose_scene(
        scene_dir=scene_dir,
        object_dirs=object_dirs,
        object_transforms=transforms,
        output_dir=output_dir,
        remove_bright=args.remove_bright,
        brightness_threshold=args.brightness_threshold,
        save_filtered_objects=args.save_filtered_objects,
    )
    
    # Render test views
    if args.test_render:
        render_test_views(
            composed=composed,
            scene_dir=scene_dir,
            output_dir=output_dir,
            num_views=args.num_test_views,
            camera_indices=args.camera_indices,
        )


if __name__ == "__main__":
    main()
