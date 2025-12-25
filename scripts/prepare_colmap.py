#!/usr/bin/env python3
"""
COLMAP preprocessing script for 3DGS mini-movie project.
Handles both scene and object videos with appropriate parameters.

For objects: Uses SAM segmentation to remove background before COLMAP.
For scenes: Extracts frames directly without segmentation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps=None):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        target_fps: Target extraction rate (None = use all frames)
    
    Returns:
        Number of frames extracted
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.2f}s")
    
    # Calculate frame interval
    if target_fps is None:
        frame_interval = 1
    else:
        frame_interval = max(1, int(original_fps / target_fps))
    
    print(f"\nExtracting every {frame_interval} frame(s) (target: {target_fps} fps)")
    
    frame_idx = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_name = f"frame_{saved_count:05d}.png"
                frame_path = output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"\n✓ Extracted {saved_count} frames to {output_dir}")
    return saved_count


def extract_frames_with_sam(video_path, output_dir, target_fps=None, 
                            skip_first_n=90, sam_model="facebook/sam-vit-base",
                            prompt_type="center", prompt_value=None,
                            interactive=False):
    """
    Extract frames from video with SAM segmentation for object extraction.
    Background is replaced with white color.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        target_fps: Target extraction rate
        skip_first_n: Skip first N frames (for ID card display, ~3s at 30fps)
        sam_model: SAM model to use
        prompt_type: "center", "point", or "box"
        prompt_value: Prompt value (e.g., "x,y" for point)
        interactive: Interactive mode for prompt selection
    
    Returns:
        Number of frames extracted
    """
    # Import SAM module
    try:
        from sam_segmentation import (
            SAMSegmenter, segment_frame, interactive_prompt_selection
        )
    except ImportError:
        print("Error: sam_segmentation.py not found in the same directory")
        print("Make sure sam_segmentation.py exists in scripts/")
        sys.exit(1)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {original_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.2f}s")
    print(f"  Skipping first {skip_first_n} frames (ID card)")
    
    # Calculate frame interval
    if target_fps is None:
        frame_interval = 1
    else:
        frame_interval = max(1, int(original_fps / target_fps))
    
    print(f"\nExtracting every {frame_interval} frame(s) (target: {target_fps} fps)")
    
    # Interactive prompt selection
    if interactive:
        # Read first valid frame for preview
        for _ in range(skip_first_n):
            cap.read()
        ret, first_frame = cap.read()
        if ret:
            prompt_type, prompt_value = interactive_prompt_selection(first_frame)
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize SAM segmenter
    print(f"\nInitializing SAM segmenter ({sam_model})...")
    segmenter = SAMSegmenter(model_name=sam_model)
    
    frame_idx = 0
    saved_count = 0
    
    print(f"\nProcessing frames with SAM segmentation...")
    with tqdm(total=total_frames, desc="Segmenting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip first N frames (ID card)
            if frame_idx < skip_first_n:
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Sample at target FPS
            if (frame_idx - skip_first_n) % frame_interval == 0:
                # Segment frame
                masked_frame = segment_frame(
                    frame, segmenter, 
                    prompt_type=prompt_type,
                    prompt_value=prompt_value,
                    background_color=(255, 255, 255)  # White background
                )
                
                # Save frame
                frame_name = f"frame_{saved_count:05d}.png"
                frame_path = output_dir / frame_name
                cv2.imwrite(str(frame_path), masked_frame)
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"\n✓ Extracted and segmented {saved_count} frames to {output_dir}")
    return saved_count


def run_colmap_feature_extraction(workspace_dir, camera_model="PINHOLE", single_camera=True):
    """Run COLMAP feature extraction with GPU support."""
    database_path = workspace_dir / "database.db"
    images_dir = workspace_dir / "images"
    
    # Remove existing database
    if database_path.exists():
        database_path.unlink()
        print("  Removed existing database")
    
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--SiftExtraction.use_gpu", "0",  # CPU mode (OpenGL issues in headless)
    ]
    
    print(f"\nRunning: colmap feature_extractor")
    print(f"  Database: {database_path}")
    print(f"  Images: {images_dir}")
    print(f"  Camera model: {camera_model}")
    print(f"  GPU: Disabled (headless environment)")
    
    # Set environment for headless execution
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    try:
        # Real-time output: no buffering
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr to stdout for real-time output
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # Monitor progress in real-time
        print("\n  Progress (this will take 1-3 minutes for 56 images):")
        for line in process.stdout:
            line = line.strip()
            # Filter out noise but show all relevant progress
            if line and 'XDG_RUNTIME_DIR' not in line and 'QStandardPaths' not in line:
                print(f"  {line}", flush=True)  # flush=True for immediate output
        
        process.wait()
        
        if process.returncode == 0:
            print("✓ Feature extraction completed")
            return True
        else:
            print(f"✗ Feature extraction failed (exit code: {process.returncode})")
            return False
            
    except FileNotFoundError:
        print("\n✗ Error: COLMAP not found!")
        print("  conda install -c conda-forge colmap")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_colmap_matcher(workspace_dir, matcher_type="sequential"):
    """Run COLMAP feature matching with GPU support."""
    database_path = workspace_dir / "database.db"
    
    if matcher_type == "exhaustive":
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "0",  # CPU mode (OpenGL issues)
        ]
    elif matcher_type == "sequential":
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", "10",
            "--SiftMatching.use_gpu", "0",  # CPU mode (OpenGL issues)
        ]
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")
    
    print(f"\nRunning: colmap {matcher_type}_matcher")
    print(f"  Database: {database_path}")
    print(f"  GPU: Disabled (headless environment)")
    
    # Set environment for headless execution
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # Monitor progress in real-time
        print("\n  Progress (this will take 30 seconds to 2 minutes):")
        for line in process.stdout:
            line = line.strip()
            # Show all relevant matching progress
            if line and 'XDG_RUNTIME_DIR' not in line and 'QStandardPaths' not in line:
                print(f"  {line}", flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            print("✓ Feature matching completed")
            return True
        else:
            print(f"✗ Feature matching failed (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_colmap_mapper(workspace_dir):
    """Run COLMAP incremental mapping."""
    database_path = workspace_dir / "database.db"
    images_dir = workspace_dir / "images"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ]
    
    print(f"\nRunning: colmap mapper")
    print(f"  Database: {database_path}")
    print(f"  Images: {images_dir}")
    print(f"  Output: {sparse_dir}")
    print("\n  This may take several minutes...")
    
    # Set environment for headless execution
    env = os.environ.copy()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # Monitor progress in real-time
        print("\n  Progress (this will take 2-5 minutes):")
        for line in process.stdout:
            line = line.strip()
            # Show all reconstruction progress
            if line and 'XDG_RUNTIME_DIR' not in line and 'QStandardPaths' not in line:
                print(f"  {line}", flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            # Check if reconstruction was created
            model_dirs = list(sparse_dir.glob("*"))
            if model_dirs:
                print(f"✓ Sparse reconstruction completed")
                print(f"  Found {len(model_dirs)} reconstruction model(s)")
                return True
            else:
                print("✗ No reconstruction model created")
                return False
        else:
            print(f"✗ Mapper failed (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def process_video(video_path, output_dir, data_type="scene", 
                  target_fps=None, camera_model="PINHOLE", matcher_type="sequential",
                  use_sam=None, sam_model="facebook/sam-vit-base",
                  sam_prompt_type="center", sam_prompt_value=None,
                  sam_skip_frames=90, sam_interactive=False):
    """
    Complete COLMAP pipeline for a video.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for COLMAP data
        data_type: "scene" or "object"
        target_fps: Target FPS for frame extraction
        camera_model: COLMAP camera model
        matcher_type: Feature matching strategy
        use_sam: Use SAM segmentation (None = auto based on data_type)
        sam_model: SAM model name
        sam_prompt_type: "center", "point", or "box"
        sam_prompt_value: Prompt value for point/box
        sam_skip_frames: Skip first N frames (ID card)
        sam_interactive: Interactive prompt selection
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Auto-detect SAM usage based on data type
    if use_sam is None:
        use_sam = (data_type == "object")
    
    # Set default FPS based on data type
    if target_fps is None:
        if data_type == "scene":
            target_fps = 2  # Scene: slow motion, less frames
        elif data_type == "object":
            target_fps = 6  # Object: rotation, more frames
        else:
            target_fps = 3
    
    print("\n" + "="*70)
    print(f"COLMAP PROCESSING: {data_type.upper()}")
    if use_sam:
        print("  [SAM Segmentation ENABLED - background will be removed]")
    print("="*70)
    print(f"  Video: {video_path.name}")
    print(f"  Output: {output_dir}")
    print(f"  Target FPS: {target_fps}")
    print(f"  Camera model: {camera_model}")
    print(f"  Matcher: {matcher_type}")
    if use_sam:
        print(f"  SAM Model: {sam_model}")
        print(f"  SAM Prompt: {sam_prompt_type}")
        print(f"  Skip first {sam_skip_frames} frames (ID card)")
    print("="*70)
    
    # Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse"
    sparse_0_dir = sparse_dir / "0"
    
    # Step 1: Extract frames (with or without SAM segmentation)
    print("\n[1/4] EXTRACTING FRAMES")
    if use_sam:
        print("       (with SAM segmentation)")
    print("-" * 70)
    
    if use_sam:
        num_frames = extract_frames_with_sam(
            video_path, images_dir, 
            target_fps=target_fps,
            skip_first_n=sam_skip_frames,
            sam_model=sam_model,
            prompt_type=sam_prompt_type,
            prompt_value=sam_prompt_value,
            interactive=sam_interactive
        )
    else:
        num_frames = extract_frames(video_path, images_dir, target_fps=target_fps)
    
    if num_frames < 10:
        print(f"\n✗ ERROR: Only {num_frames} frames extracted. Need at least 10.")
        return False
    
    # Step 2: Feature extraction
    print("\n[2/4] FEATURE EXTRACTION")
    print("-" * 70)
    if not run_colmap_feature_extraction(output_dir, camera_model=camera_model):
        print("\n✗ ERROR: Feature extraction failed")
        return False
    
    # Step 3: Feature matching
    print("\n[3/4] FEATURE MATCHING")
    print("-" * 70)
    if not run_colmap_matcher(output_dir, matcher_type=matcher_type):
        print("\n✗ ERROR: Feature matching failed")
        return False
    
    # Step 4: Sparse reconstruction
    print("\n[4/4] SPARSE RECONSTRUCTION")
    print("-" * 70)
    if not run_colmap_mapper(output_dir):
        print("\n✗ ERROR: Sparse reconstruction failed")
        return False
    
    # Verify output
    print("\n" + "="*70)
    print("VERIFYING OUTPUT")
    print("="*70)
    
    if not sparse_0_dir.exists():
        print(f"✗ ERROR: Model directory not found: {sparse_0_dir}")
        return False
    
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [f for f in required_files if not (sparse_0_dir / f).exists()]
    
    if missing:
        print(f"✗ ERROR: Missing required files: {missing}")
        return False
    
    print("✓ All required files present:")
    for f in required_files:
        fpath = sparse_0_dir / f
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    
    # Save metadata
    metadata = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "data_type": data_type,
        "target_fps": target_fps,
        "num_frames_extracted": num_frames,
        "camera_model": camera_model,
        "matcher_type": matcher_type,
        "use_sam": use_sam,
    }
    
    if use_sam:
        metadata.update({
            "sam_model": sam_model,
            "sam_prompt_type": sam_prompt_type,
            "sam_prompt_value": sam_prompt_value,
            "sam_skip_frames": sam_skip_frames,
        })
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved: {metadata_path}")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/          ({num_frames} frames)")
    print(f"    ├── sparse/")
    print(f"    │   └── 0/           (reconstruction)")
    print(f"    ├── database.db")
    print(f"    └── metadata.json")
    print("\n" + "="*70 + "\n")
    
    return True


def main():
    print("\n" + "!"*70)
    print("WARNING: This process will take 5-10 minutes to complete.")
    print("DO NOT interrupt with Ctrl+C unless absolutely necessary!")
    print("Progress will be shown in real-time.")
    print("!"*70 + "\n")
    
    parser = argparse.ArgumentParser(
        description="COLMAP preprocessing for 3DGS mini-movie project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process scene video (no SAM segmentation)
  python prepare_colmap.py --video video/KU_statue2.mp4 --output data/scene_colmap --type scene
  
  # Process object video (SAM segmentation enabled by default for objects)
  python prepare_colmap.py --video video/cat.mp4 --output data/cat_colmap --type object
  
  # Object with interactive SAM prompt selection
  python prepare_colmap.py --video video/tiger.mp4 --output data/tiger_colmap --type object --sam-interactive
  
  # Object with custom point prompt
  python prepare_colmap.py --video video/cat.mp4 --output data/cat_colmap --type object \\
      --sam-prompt-type point --sam-prompt-value "640,360"
"""
    )
    
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for COLMAP data")
    parser.add_argument("--type", type=str, choices=["scene", "object"], default="scene",
                       help="Type of capture (affects processing parameters)")
    parser.add_argument("--fps", type=float, default=None,
                       help="Target FPS for frame extraction (default: auto based on type)")
    parser.add_argument("--camera-model", type=str, default="PINHOLE",
                       choices=["PINHOLE", "SIMPLE_PINHOLE", "OPENCV", "OPENCV_FISHEYE"],
                       help="COLMAP camera model")
    parser.add_argument("--matcher", type=str, default="sequential",
                       choices=["sequential", "exhaustive"],
                       help="Feature matching strategy")
    
    # SAM segmentation options
    sam_group = parser.add_argument_group("SAM Segmentation Options (for objects)")
    sam_group.add_argument("--no-sam", action="store_true",
                           help="Disable SAM segmentation for objects")
    sam_group.add_argument("--sam-model", type=str, default="facebook/sam-vit-base",
                           choices=["facebook/sam-vit-base", "facebook/sam-vit-large", "facebook/sam-vit-huge"],
                           help="SAM model to use (default: facebook/sam-vit-base)")
    sam_group.add_argument("--sam-prompt-type", type=str, default="center",
                           choices=["center", "point", "box"],
                           help="SAM prompt type (default: center)")
    sam_group.add_argument("--sam-prompt-value", type=str, default=None,
                           help="SAM prompt value (x,y for point; x1,y1,x2,y2 for box)")
    sam_group.add_argument("--sam-skip-frames", type=int, default=90,
                           help="Skip first N frames for ID card (default: 90 = 3s at 30fps)")
    sam_group.add_argument("--sam-interactive", action="store_true",
                           help="Interactive mode for SAM prompt selection")
    
    args = parser.parse_args()
    
    # Validate video exists
    if not os.path.exists(args.video):
        print(f"✗ ERROR: Video file not found: {args.video}")
        sys.exit(1)
    
    # Check COLMAP installation
    try:
        subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            check=True,
            timeout=5
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ ERROR: COLMAP not found!")
        print("  conda install -c conda-forge colmap")
        sys.exit(1)
    
    # Determine SAM usage
    use_sam = None  # Auto-detect based on type
    if args.no_sam:
        use_sam = False
    
    # Process video
    success = process_video(
        video_path=args.video,
        output_dir=args.output,
        data_type=args.type,
        target_fps=args.fps,
        camera_model=args.camera_model,
        matcher_type=args.matcher,
        use_sam=use_sam,
        sam_model=args.sam_model,
        sam_prompt_type=args.sam_prompt_type,
        sam_prompt_value=args.sam_prompt_value,
        sam_skip_frames=args.sam_skip_frames,
        sam_interactive=args.sam_interactive,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
