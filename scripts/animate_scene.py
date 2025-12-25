#!/usr/bin/env python3
"""
Animate 3DGS Mini-Movie with Dynamic Objects and Camera Motion

Creates a 14-second animated sequence (420 frames @ 30fps):
1. Camera move (2s, 60f): Camera 15 → Camera 45, Curby static at [-0.8, 0, 0]
2. Curby spin + morph (3s, 90f): Curby spins 3×360° (yaw), then becomes Cat
3. Cat scale up (4s, 120f): Cat scales from 0.5 → 2.0 (4×)
4. Orbit camera (3s, 90f): Camera orbits around Cat (-60° → +60°)
5. Cat exit + camera return (2s, 60f): Cat moves to [1, 0, 2], camera jumps to 45

Usage:
    python scripts/animate_scene.py \
        --scene trained_models/scene_training \
        --curby trained_models/curby_training \
        --cat trained_models/cat_training \
        --output animated_movie \
        --fps 30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_checkpoint(ckpt_path: Path, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load checkpoint and extract splats"""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    splats = ckpt["splats"]
    num_gaussians = splats["means"].shape[0]
    print(f"  Loaded {num_gaussians:,} Gaussians")
    return splats


def apply_filtering(
    splats: Dict[str, torch.Tensor],
    filter_params: Dict,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Apply outlier filtering to object splats
    Same filtering as in compose_scene.py
    """
    means = splats["means"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    opacities = splats["opacities"].detach().cpu().numpy()
    sh0 = splats["sh0"].detach().cpu().numpy()
    shN = splats["shN"].detach().cpu().numpy() if "shN" in splats else None
    
    original_count = len(means)
    
    # Center to origin
    if filter_params.get("center_first", True):
        center = means.mean(axis=0)
        means = means - center
        print(f"  Centered: moved by {-center.round(3)}")
        
        # Distance-based filtering
        if filter_params.get("filter_outliers", False):
            distances = np.linalg.norm(means, axis=1)
            percentile = filter_params.get("percentile", 80)
            density_radius = filter_params.get("density_radius", 0.5)
            
            distance_threshold = np.percentile(distances, percentile)
            max_distance = density_radius * 10
            distance_threshold = min(distance_threshold, max_distance)
            
            keep_mask = distances < distance_threshold
            print(f"  Distance filter: {keep_mask.sum()}/{len(means)} kept (threshold={distance_threshold:.2f})")
            
            means = means[keep_mask]
            quats = quats[keep_mask]
            scales = scales[keep_mask]
            opacities = opacities[keep_mask]
            sh0 = sh0[keep_mask]
            if shN is not None:
                shN = shN[keep_mask]
        
        # Opacity filtering
        min_opacity = filter_params.get("min_opacity", 0.0)
        if min_opacity > 0:
            opacity_sigmoid = torch.sigmoid(torch.from_numpy(opacities)).numpy()
            opacity_mask = opacity_sigmoid.squeeze() >= min_opacity
            print(f"  Opacity filter: {opacity_mask.sum()}/{len(means)} kept")
            
            means = means[opacity_mask]
            quats = quats[opacity_mask]
            scales = scales[opacity_mask]
            opacities = opacities[opacity_mask]
            sh0 = sh0[opacity_mask]
            if shN is not None:
                shN = shN[opacity_mask]
        
        # Color filtering (outside core)
        if filter_params.get("color_filter_outside_core", False):
            core_percentile = filter_params.get("core_percentile", 20)
            max_saturation = filter_params.get("max_saturation", 0.2)
            
            distances = np.linalg.norm(means, axis=1)
            core_threshold = np.percentile(distances, core_percentile)
            is_core = distances < core_threshold
            
            colors_rgb = sh0.squeeze(1)
            color_max = colors_rgb.max(axis=1)
            color_min = colors_rgb.min(axis=1)
            saturation = np.where(color_max > 0, (color_max - color_min) / (color_max + 1e-8), 0)
            
            is_colorful = saturation > max_saturation
            keep_mask = is_core | is_colorful
            print(f"  Color filter: {keep_mask.sum()}/{len(means)} kept (core={is_core.sum()})")
            
            means = means[keep_mask]
            quats = quats[keep_mask]
            scales = scales[keep_mask]
            opacities = opacities[keep_mask]
            sh0 = sh0[keep_mask]
            if shN is not None:
                shN = shN[keep_mask]
        
        print(f"  Total: {original_count} → {len(means)} Gaussians")
    
    # Convert back to tensors
    filtered = {
        "means": torch.from_numpy(means).float().to(device),
        "quats": torch.from_numpy(quats).float().to(device),
        "scales": torch.from_numpy(scales).float().to(device),
        "opacities": torch.from_numpy(opacities).float().to(device),
        "sh0": torch.from_numpy(sh0).float().to(device),
    }
    if shN is not None:
        filtered["shN"] = torch.from_numpy(shN).float().to(device)
    
    return filtered


def transform_splats(
    splats: Dict[str, torch.Tensor],
    translation: np.ndarray,
    rotation: np.ndarray,  # Euler angles in degrees
    scale: float,
) -> Dict[str, torch.Tensor]:
    """
    Apply rigid transformation to splats
    """
    means = splats["means"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    scales_log = splats["scales"].detach().cpu().numpy()
    
    # Scale
    means = means * scale
    scales_log = scales_log + np.log(scale)
    
    # Rotate
    if np.any(rotation != 0):
        rot_matrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
        means = means @ rot_matrix.T
        
        # Rotate quaternions
        rot_quat = R.from_euler('xyz', rotation, degrees=True).as_quat()
        original_rot = R.from_quat(quats)
        new_rot = R.from_quat(rot_quat) * original_rot
        quats = new_rot.as_quat()
    
    # Translate
    means = means + translation
    
    # Update tensors
    result = splats.copy()
    result["means"] = torch.from_numpy(means).float().to(splats["means"].device)
    result["quats"] = torch.from_numpy(quats).float().to(splats["quats"].device)
    result["scales"] = torch.from_numpy(scales_log).float().to(splats["scales"].device)
    
    return result


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation"""
    return a + (b - a) * t


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation for quaternions"""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate angle between quaternions
    dot = np.dot(q1, q2)
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product
    dot = np.clip(dot, -1, 1)
    
    # Calculate interpolation
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2


def camera_matrix_from_params(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Create 4×4 camera-to-world matrix from position and quaternion"""
    # Quaternion to rotation matrix (wxyz format)
    rot = R.from_quat(quaternion).as_matrix()
    
    # Build camtoworld matrix
    camtoworld = np.eye(4)
    camtoworld[:3, :3] = rot
    camtoworld[:3, 3] = position
    
    return camtoworld


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """
    Create a camera-to-world matrix that looks at target from eye position
    """
    # Forward direction (from eye to target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Right direction
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Build camera-to-world matrix
    camtoworld = np.eye(4)
    camtoworld[:3, 0] = right
    camtoworld[:3, 1] = up
    camtoworld[:3, 2] = -forward  # Camera looks down -Z axis
    camtoworld[:3, 3] = eye
    
    return camtoworld


def generate_animation_sequence(
    scene_dataset: Dataset,
    curby_splats: Dict[str, torch.Tensor],
    cat_splats: Dict[str, torch.Tensor],
    fps: int = 30,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate per-frame animation parameters
    
    Returns:
        camera_frames: List of camera parameters per frame
        object_frames: List of object parameters per frame
    """
    # Animation parameters
    CAT_POSITION = np.array([-0.8, 0.0, 0.0])
    CURBY_ROTATION_INIT = np.array([150, 0, 30])  # roll, pitch, yaw
    CAT_ROTATION_INIT = np.array([180, 0, 60])
    CURBY_SCALE = 0.5
    CAT_SCALE_START = 0.5
    CAT_SCALE_END = 2.0
    CAT_EXIT_POSITION = np.array([-5.0, 0.0, 2.0]) #change
    
    # Orbit parameters
    ORBIT_RADIUS = 0.8  # Similar to mean camera distance
    ORBIT_HEIGHT = CAT_POSITION[2] - 0.2  # Slightly below cat
    ORBIT_ANGLE_START = -60  # degrees
    ORBIT_ANGLE_END = 60
    
    # Get reference cameras
    cam15_data = scene_dataset[15]
    cam45_data = scene_dataset[45]
    
    cam15_matrix = cam15_data["camtoworld"].numpy()
    cam45_matrix = cam45_data["camtoworld"].numpy()
    
    cam15_pos = cam15_matrix[:3, 3]
    cam45_pos = cam45_matrix[:3, 3]
    
    # Convert to quaternions for SLERP
    cam15_quat = R.from_matrix(cam15_matrix[:3, :3]).as_quat()  # xyzw
    cam45_quat = R.from_matrix(cam45_matrix[:3, :3]).as_quat()
    
    camera_frames = []
    object_frames = []
    
    # Stage 1: Camera move (2s, 60 frames)
    print("Stage 1: Camera move (15 → 45)")
    for i in range(60):
        t = i / 59.0
        
        # Camera interpolation (LERP + SLERP)
        cam_pos = lerp(cam15_pos, cam45_pos, t)
        cam_quat = slerp(cam15_quat, cam45_quat, t)
        cam_matrix = camera_matrix_from_params(cam_pos, cam_quat)
        
        camera_frames.append({"camtoworld": cam_matrix})
        
        # Objects: Curby at initial position
        object_frames.append({
            "curby": {
                "visible": True,
                "translation": CAT_POSITION,
                "rotation": CURBY_ROTATION_INIT,
                "scale": CURBY_SCALE,
            },
            "cat": {"visible": False},
        })
    
    # Stage 2: Curby spin + morph to Cat (3s, 90 frames)
    print("Stage 2: Curby spin + morph")
    for i in range(90):
        t = i / 89.0
        
        # Camera: stay at 45
        camera_frames.append({"camtoworld": cam45_matrix})
        
        # Curby spins 3×360° = 1080°
        yaw_rotation = CURBY_ROTATION_INIT[2] + 1080 * t
        
        if i < 89:  # Curby visible until last frame
            object_frames.append({
                "curby": {
                    "visible": True,
                    "translation": CAT_POSITION,
                    "rotation": np.array([CURBY_ROTATION_INIT[0], CURBY_ROTATION_INIT[1], yaw_rotation]),
                    "scale": CURBY_SCALE,
                },
                "cat": {"visible": False},
            })
        else:  # Last frame: Cat appears
            object_frames.append({
                "curby": {"visible": False},
                "cat": {
                    "visible": True,
                    "translation": CAT_POSITION,
                    "rotation": CAT_ROTATION_INIT,
                    "scale": CAT_SCALE_START,
                },
            })
    
    # Stage 3: Cat scale up (4s, 120 frames)
    print("Stage 3: Cat scale up")
    for i in range(120):
        t = i / 119.0
        
        # Camera: stay at 45
        camera_frames.append({"camtoworld": cam45_matrix})
        
        # Cat scales up
        cat_scale = lerp(CAT_SCALE_START, CAT_SCALE_END, t)
        object_frames.append({
            "curby": {"visible": False},
            "cat": {
                "visible": True,
                "translation": CAT_POSITION,
                "rotation": CAT_ROTATION_INIT,
                "scale": cat_scale,
            },
        })
    
    # Stage 4: Orbit camera using real COLMAP cameras (3s, 90 frames)
    # Camera 60 → 105 → 60 (forward then backward)
    print("Stage 4: Orbit camera (COLMAP 60→105→60)")
    
    # Get camera data for 60 and 105
    cam60_data = scene_dataset[60]
    cam105_data = scene_dataset[105]
    
    for i in range(90):
        t = i / 89.0
        
        # First half: 60 → 105 (forward)
        # Second half: 105 → 60 (backward)
        if t <= 0.5:
            # Forward motion: 0 → 0.5 maps to camera 60 → 105
            local_t = t * 2.0  # 0 → 1
            cam_start_idx = 60
            cam_end_idx = 105
            
            # Interpolate between actual camera indices
            cam_idx = int(cam_start_idx + (cam_end_idx - cam_start_idx) * local_t)
            cam_idx = min(cam_idx, cam_end_idx)
        else:
            # Backward motion: 0.5 → 1.0 maps to camera 105 → 60
            local_t = (t - 0.5) * 2.0  # 0 → 1
            cam_start_idx = 105
            cam_end_idx = 60
            
            cam_idx = int(cam_start_idx + (cam_end_idx - cam_start_idx) * local_t)
            cam_idx = max(cam_idx, cam_end_idx)
        
        # Get camera matrix for current index
        cam_data = scene_dataset[cam_idx]
        cam_matrix = cam_data["camtoworld"].numpy()
        
        camera_frames.append({"camtoworld": cam_matrix})
        
        # Cat stays at final scale
        object_frames.append({
            "curby": {"visible": False},
            "cat": {
                "visible": True,
                "translation": CAT_POSITION,
                "rotation": CAT_ROTATION_INIT,
                "scale": CAT_SCALE_END,
            },
        })
    
    # Stage 5: Cat exit + camera return (2s, 60 frames)
    print("Stage 5: Cat exit + camera return to 45")
    for i in range(60):
        t = i / 59.0
        
        # Camera: smoothly return to 45 from 60
        cam60_matrix = scene_dataset[60]["camtoworld"].numpy()
        cam_matrix = lerp(cam60_matrix, cam45_matrix, t)
        
        camera_frames.append({"camtoworld": cam_matrix})
        
        # Cat moves to exit position
        cat_pos = lerp(CAT_POSITION, CAT_EXIT_POSITION, t)
        object_frames.append({
            "curby": {"visible": False},
            "cat": {
                "visible": True,
                "translation": cat_pos,
                "rotation": CAT_ROTATION_INIT,
                "scale": CAT_SCALE_END,
            },
        })
    
    total_frames = len(camera_frames)
    print(f"\nTotal frames: {total_frames} ({total_frames / fps:.1f}s @ {fps}fps)")
    
    return camera_frames, object_frames


def render_frame(
    scene_splats: Dict[str, torch.Tensor],
    curby_splats: Dict[str, torch.Tensor],
    cat_splats: Dict[str, torch.Tensor],
    camera_params: Dict,
    object_params: Dict,
    K: torch.Tensor,
    width: int,
    height: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Render a single frame with composed objects
    """
    # Start with scene splats
    combined_means = [scene_splats["means"]]
    combined_quats = [scene_splats["quats"]]
    combined_scales = [scene_splats["scales"]]
    combined_opacities = [scene_splats["opacities"]]
    combined_sh0 = [scene_splats["sh0"]]
    combined_shN = [scene_splats.get("shN")]
    
    # Add Curby if visible
    if object_params["curby"]["visible"]:
        curby_transformed = transform_splats(
            curby_splats,
            translation=object_params["curby"]["translation"],
            rotation=object_params["curby"]["rotation"],
            scale=object_params["curby"]["scale"],
        )
        combined_means.append(curby_transformed["means"])
        combined_quats.append(curby_transformed["quats"])
        combined_scales.append(curby_transformed["scales"])
        combined_opacities.append(curby_transformed["opacities"])
        combined_sh0.append(curby_transformed["sh0"])
        if "shN" in curby_transformed:
            if combined_shN[0] is not None:
                combined_shN.append(curby_transformed["shN"])
    
    # Add Cat if visible
    if object_params["cat"]["visible"]:
        cat_transformed = transform_splats(
            cat_splats,
            translation=object_params["cat"]["translation"],
            rotation=object_params["cat"]["rotation"],
            scale=object_params["cat"]["scale"],
        )
        combined_means.append(cat_transformed["means"])
        combined_quats.append(cat_transformed["quats"])
        combined_scales.append(cat_transformed["scales"])
        combined_opacities.append(cat_transformed["opacities"])
        combined_sh0.append(cat_transformed["sh0"])
        if "shN" in cat_transformed:
            if combined_shN[0] is not None:
                combined_shN.append(cat_transformed["shN"])
    
    # Concatenate all splats
    means = torch.cat(combined_means, dim=0)
    quats = torch.cat(combined_quats, dim=0)
    scales = torch.cat(combined_scales, dim=0)
    opacities = torch.cat(combined_opacities, dim=0)
    sh0 = torch.cat(combined_sh0, dim=0)
    
    # Handle shN
    if combined_shN[0] is not None:
        shN = torch.cat([s for s in combined_shN if s is not None], dim=0)
        colors = torch.cat([sh0, shN], dim=1)
        sh_degree = int((colors.shape[1] ** 0.5) - 1)
    else:
        colors = sh0
        sh_degree = 0
    
    # Camera parameters
    camtoworld = torch.from_numpy(camera_params["camtoworld"]).float().to(device)
    viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)  # [1, 4, 4]
    K_batch = K.unsqueeze(0)  # [1, 3, 3]
    
    # Rasterization
    with torch.no_grad():
        renders, alphas, info = rasterization(
            means=means,
            quats=F.normalize(quats, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmats=viewmat,
            Ks=K_batch,
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
        )
    
    # Convert to numpy image
    render_img = renders[0].detach().cpu().clamp(0, 1).numpy()
    render_img = (render_img * 255).astype(np.uint8)
    
    return render_img


def main():
    parser = argparse.ArgumentParser(description="Animate 3DGS mini-movie")
    parser.add_argument("--scene", type=str, required=True, help="Scene training directory")
    parser.add_argument("--curby", type=str, required=True, help="Curby training directory")
    parser.add_argument("--cat", type=str, required=True, help="Cat training directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup paths
    scene_dir = Path(args.scene).resolve()
    curby_dir = Path(args.curby).resolve()
    cat_dir = Path(args.cat).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    
    print("="*80)
    print("3DGS Mini-Movie Animation")
    print("="*80)
    print(f"Scene: {scene_dir}")
    print(f"Curby: {curby_dir}")
    print(f"Cat: {cat_dir}")
    print(f"Output: {output_dir}")
    print(f"FPS: {args.fps}")
    print("="*80)
    
    # Load scene checkpoint
    print("\n📦 Loading scene...")
    scene_ckpt = scene_dir / "ckpts" / "ckpt_29999_rank0.pt"
    if not scene_ckpt.exists():
        ckpt_files = sorted((scene_dir / "ckpts").glob("ckpt_*_rank0.pt"))
        scene_ckpt = ckpt_files[-1]
    scene_splats = load_checkpoint(scene_ckpt, device=device)
    
    # Load Curby checkpoint
    print("\n📦 Loading Curby...")
    curby_ckpt = curby_dir / "ckpts" / "ckpt_29999_rank0.pt"
    if not curby_ckpt.exists():
        ckpt_files = sorted((curby_dir / "ckpts").glob("ckpt_*_rank0.pt"))
        curby_ckpt = ckpt_files[-1]
    curby_splats_raw = load_checkpoint(curby_ckpt, device=device)
    
    # Apply Curby filtering
    print("  Applying Curby filtering...")
    curby_filter_params = {
        "center_first": True,
        "filter_outliers": True,
        "density_radius": 0.15,
        "percentile": 40,
        "min_opacity": 0.0,
        "color_filter_outside_core": True,
        "core_percentile": 20,
        "max_saturation": 0.3,
    }
    curby_splats = apply_filtering(curby_splats_raw, curby_filter_params, device=device)
    
    # Load Cat checkpoint
    print("\n📦 Loading Cat...")
    cat_ckpt = cat_dir / "ckpts" / "ckpt_29999_rank0.pt"
    if not cat_ckpt.exists():
        ckpt_files = sorted((cat_dir / "ckpts").glob("ckpt_*_rank0.pt"))
        cat_ckpt = ckpt_files[-1]
    cat_splats_raw = load_checkpoint(cat_ckpt, device=device)
    
    # Apply Cat filtering
    print("  Applying Cat filtering...")
    cat_filter_params = {
        "center_first": True,
        "filter_outliers": True,
        "density_radius": 0.3,
        "percentile": 70,
        "min_opacity": 0.0,
        "color_filter_outside_core": True,
        "core_percentile": 65,
        "max_saturation": 0.4,
    }
    cat_splats = apply_filtering(cat_splats_raw, cat_filter_params, device=device)
    
    # Load scene dataset for cameras
    print("\n📷 Loading camera parameters...")
    cfg_path = scene_dir / "cfg.yml"
    with open(cfg_path, "r") as f:
        cfg = yaml.unsafe_load(f)
    
    data_dir = cfg["data_dir"]
    data_factor = cfg.get("data_factor", 1)
    normalize = cfg.get("normalize_world_space", True)
    
    parser_obj = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=normalize,
        test_every=8,
    )
    scene_dataset = Dataset(parser_obj, split="train")
    
    # Get camera intrinsics
    sample = scene_dataset[0]
    K = sample["K"].to(device)
    height, width = sample["image"].shape[:2]
    
    print(f"  Resolution: {width} × {height}")
    print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    # Generate animation sequence
    print("\n🎬 Generating animation sequence...")
    camera_frames, object_frames = generate_animation_sequence(
        scene_dataset,
        curby_splats,
        cat_splats,
        fps=args.fps,
    )
    
    # Save JSON files
    print("\n💾 Saving animation parameters...")
    
    # Save camera trajectory
    camera_json = []
    for i, cam in enumerate(camera_frames):
        camera_json.append({
            "frame": i,
            "camtoworld": cam["camtoworld"].tolist(),
        })
    
    with open(output_dir / "camera_trajectory.json", "w") as f:
        json.dump(camera_json, f, indent=2)
    print(f"  Saved: {output_dir / 'camera_trajectory.json'}")
    
    # Save object motion
    object_json = []
    for i, obj in enumerate(object_frames):
        frame_data = {"frame": i}
        
        if obj["curby"]["visible"]:
            frame_data["curby"] = {
                "visible": True,
                "translation": obj["curby"]["translation"].tolist(),
                "rotation": obj["curby"]["rotation"].tolist(),
                "scale": float(obj["curby"]["scale"]),
            }
        else:
            frame_data["curby"] = {"visible": False}
        
        if obj["cat"]["visible"]:
            frame_data["cat"] = {
                "visible": True,
                "translation": obj["cat"]["translation"].tolist(),
                "rotation": obj["cat"]["rotation"].tolist(),
                "scale": float(obj["cat"]["scale"]),
            }
        else:
            frame_data["cat"] = {"visible": False}
        
        object_json.append(frame_data)
    
    with open(output_dir / "object_motion.json", "w") as f:
        json.dump(object_json, f, indent=2)
    print(f"  Saved: {output_dir / 'object_motion.json'}")
    
    # Render frames
    print("\n🎥 Rendering frames...")
    frames = []
    
    for i in tqdm(range(len(camera_frames)), desc="Rendering"):
        frame_img = render_frame(
            scene_splats=scene_splats,
            curby_splats=curby_splats,
            cat_splats=cat_splats,
            camera_params=camera_frames[i],
            object_params=object_frames[i],
            K=K,
            width=width,
            height=height,
            device=device,
        )
        frames.append(frame_img)
        
        # Save individual frame
        if i % 30 == 0:  # Save every 1 second
            frame_path = output_dir / "frames" / f"frame_{i:04d}.png"
            frame_path.parent.mkdir(exist_ok=True)
            imageio.imwrite(frame_path, frame_img)
    
    # Save as video
    print("\n💾 Saving video...")
    video_path = output_dir / "mini_movie.mp4"
    imageio.mimwrite(video_path, frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"  Saved: {video_path}")
    
    # Save metadata
    metadata = {
        "scene": str(scene_dir),
        "curby": str(curby_dir),
        "cat": str(cat_dir),
        "fps": args.fps,
        "total_frames": len(frames),
        "duration_seconds": len(frames) / args.fps,
        "resolution": [width, height],
        "curby_filter_params": curby_filter_params,
        "cat_filter_params": cat_filter_params,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Animation complete!")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / args.fps:.1f}s")
    print(f"  Output: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
