#!/usr/bin/env python3
"""
3DGS Training Wrapper for Mini-Movie Project

Wraps simple_trainer.py with project-specific settings:
- Headless mode (disable_viewer=True)
- Consistent world space scaling (normalize_world_space=False)
- Random background for objects (random_bkgd=True)
- Fixed background for scene (random_bkgd=False)

Usage:
    # Train scene
    python scripts/train_model.py --data data/scene_colmap --output trained_models/scene.pt --type scene --steps 30000
    
    # Train object (transparent background)
    python scripts/train_model.py --data data/cat_colmap --output trained_models/cat.pt --type object --steps 30000
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3DGS model with project-specific settings")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to COLMAP data directory (e.g., data/scene_colmap)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for trained model .pt file (e.g., trained_models/scene.pt)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["scene", "object"],
        required=True,
        help="Type of data: 'scene' (solid background) or 'object' (transparent background)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30000,
        help="Number of training steps (default: 30000)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["default", "mcmc"],
        default="mcmc",
        help="Training strategy: 'default' or 'mcmc' (default: mcmc)",
    )
    parser.add_argument(
        "--eval-steps",
        type=str,
        default="7000,30000",
        help="Comma-separated eval steps (default: 7000,30000)",
    )
    parser.add_argument(
        "--save-steps",
        type=str,
        default="5000,10000,15000,20000,25000,30000",
        help="Comma-separated save steps (default: every 5000 steps)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training (default: 1)",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Spherical harmonics degree (default: 3)",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Save .ply files at save steps (can be large)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    gsplat_repo = project_root / "gsplat_repo"
    trainer_script = gsplat_repo / "examples" / "simple_trainer.py"
    
    data_dir = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    if not (data_dir / "sparse" / "0").exists():
        print(f"Error: COLMAP sparse reconstruction not found in {data_dir}/sparse/0")
        sys.exit(1)
    
    if not trainer_script.exists():
        print(f"Error: simple_trainer.py not found: {trainer_script}")
        sys.exit(1)
    
    # Parse eval and save steps
    eval_steps = [int(x) for x in args.eval_steps.split(",")]
    save_steps = [int(x) for x in args.save_steps.split(",")]
    
    # Create result directory name
    result_name = output_path.stem  # e.g., "scene" from "scene.pt"
    result_dir = output_dir / f"{result_name}_training"
    
    # Check for resume
    resume_ckpt = None
    # Auto-detect latest checkpoint if result_dir exists
    ckpt_dir = result_dir / "ckpts"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
        if ckpts:
            resume_ckpt = ckpts[-1]
            print(f"Found existing training, resuming from: {resume_ckpt}")
    
    # Build command
    cmd = [
        sys.executable,  # Python executable
        str(trainer_script),
        args.strategy,  # "default" or "mcmc"
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(args.steps),
        "--data_factor", "1",  # Use original resolution (no downsampling)
        "--disable_viewer",  # Headless mode (no value needed)
        "--batch_size", str(args.batch_size),
        "--sh_degree", str(args.sh_degree),
        "--disable_video",  # Skip video during training (no value needed)
    ]
    
    # Add resume checkpoint
    if resume_ckpt:
        cmd.extend(["--resume", str(resume_ckpt)])
    
    # Add eval steps
    for step in eval_steps:
        cmd.extend(["--eval_steps", str(step)])
    
    # Add save steps
    for step in save_steps:
        cmd.extend(["--save_steps", str(step)])
    
    # Type-specific settings
    if args.type == "object":
        # Objects need transparent background
        cmd.extend(["--random_bkgd"])
        print("Training OBJECT with random background (for transparency)")
    else:
        # Scene has solid background (no --random_bkgd flag)
        print("Training SCENE with solid background")
    
    # PLY export
    if args.save_ply:
        cmd.extend(["--save_ply"])
        for step in save_steps:
            cmd.extend(["--ply_steps", str(step)])
    
    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Print configuration
    print("\n" + "="*80)
    print("3DGS Training Configuration")
    print("="*80)
    print(f"Data directory:      {data_dir}")
    print(f"Result directory:    {result_dir}")
    print(f"Output model:        {output_path}")
    print(f"Type:                {args.type}")
    print(f"Strategy:            {args.strategy}")
    print(f"Training steps:      {args.steps}")
    print(f"Eval steps:          {eval_steps}")
    print(f"Save steps:          {save_steps}")
    print(f"Batch size:          {args.batch_size}")
    print(f"SH degree:           {args.sh_degree}")
    print(f"Random background:   {args.type == 'object'}")
    print(f"Normalize world:     False (consistent scaling)")
    print(f"GPU:                 {args.gpu}")
    if resume_ckpt:
        print(f"Resume from:         {resume_ckpt}")
    print("="*80)
    if resume_ckpt:
        print("\nResuming training from checkpoint...")
    else:
        print("\nStarting training... This will take several hours.")
    print("Estimated time: 1-3 hours depending on data size")
    print("Checkpoints saved every 5000 steps (auto-resume on restart)")
    print("\nCheckpoints will be saved in:", result_dir / "ckpts")
    print("Progress logged to:", result_dir / "tb")
    print("\nYou can monitor with: tensorboard --logdir", result_dir / "tb")
    print("Press Ctrl+C to stop (safe to resume later)")
    print("="*80 + "\n")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(gsplat_repo / "examples"),  # Run from examples directory
            check=True,
        )
        
        print("\n" + "="*80)
        print("Training completed successfully!")
        print("="*80)
        
        # Find the final checkpoint
        ckpt_dir = result_dir / "ckpts"
        final_step = max(save_steps)
        ckpt_file = ckpt_dir / f"ckpt_{final_step}_rank0.pt"
        
        if ckpt_file.exists():
            # Copy checkpoint to output location
            import shutil
            shutil.copy(ckpt_file, output_path)
            print(f"\nFinal model saved to: {output_path}")
            
            # Save training metadata
            metadata = {
                "data_dir": str(data_dir),
                "result_dir": str(result_dir),
                "type": args.type,
                "strategy": args.strategy,
                "steps": args.steps,
                "final_step": final_step,
                "random_bkgd": args.type == "object",
                "normalize_world_space": False,
                "checkpoint": str(ckpt_file),
            }
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
            
            # Print model info
            import torch
            ckpt = torch.load(output_path, map_location="cpu", weights_only=True)
            num_gaussians = ckpt["splats"]["means"].shape[0]
            print(f"\nModel statistics:")
            print(f"  Number of Gaussians: {num_gaussians:,}")
            print(f"  Training step: {ckpt['step']}")
        else:
            print(f"\nWarning: Checkpoint file not found: {ckpt_file}")
            print("Training completed but output may be in:", ckpt_dir)
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("Training failed!")
        print("="*80)
        print(f"Error: {e}")
        print(f"\nCheck logs in: {result_dir}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("Training interrupted by user")
        print("="*80)
        print(f"Partial results may be in: {result_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
