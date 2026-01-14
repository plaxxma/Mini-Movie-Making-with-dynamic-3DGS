# Mini-Movie-Making-with-dynamic-3DGS

A complete pipeline for creating dynamic cinematic animations using 3D Gaussian Splatting (3DGS) technology. This project demonstrates real-world scene and object reconstruction, depth-aware composition, and JSON-driven animation rendering.

## Overview

This project implements an end-to-end system for generating dynamic mini-movies by reconstructing real environments and objects using 3D Gaussian Splatting, then composing them with programmatic camera motion and object animation. The final output is a 14-second cinematic sequence (420 frames at 30fps) featuring multiple objects with transformations, scaling, and camera movements.

The pipeline consists of three major stages:
1. **Structure-from-Motion (SfM)** using COLMAP for camera pose estimation
2. **3D Gaussian Splatting reconstruction** using the gsplat library
3. **Depth-aware composition and dynamic rendering** with JSON-based animation control

## Key Features

- **Real-world reconstruction**: Captures and reconstructs actual scenes and objects from mobile phone videos
- **Multi-object composition**: Combines independently trained 3DGS models with depth-aware occlusion handling
- **Dynamic animation system**: JSON-based animation framework supporting object transformations (translation, rotation, scale) and camera trajectories
- **Cinematic rendering**: 420-frame sequence with smooth camera movements and object animations
- **Complete reproducibility**: All animations defined declaratively in JSON files
- **Adaptive filtering**: Multi-stage object extraction using density-based and color-based filtering

## Technical Pipeline

### 1. Data Acquisition and Preprocessing

Videos are captured using a mobile phone with specific camera trajectories:
- **Scene video**: Semi-circular trajectory around the subject (filmed at Hana Square, Korea University)
- **Object videos**: 360-degree rotation capture for complete geometric coverage

Frame extraction rates:
- Scene: 2fps
- Objects: 6fps

### 2. Structure-from-Motion with COLMAP

COLMAP performs feature extraction, feature matching, and sparse reconstruction to generate:
- Camera intrinsics and extrinsics for each frame
- Sparse 3D point cloud for initialization

Each asset (scene and objects) is processed independently.

### 3. 3D Gaussian Splatting Reconstruction

Training parameters:
- **Framework**: gsplat library
- **Iterations**: 30,000
- **Loss function**: L1 + SSIM photometric loss
- **Optimization**: Adaptive density control with dynamic Gaussian addition/removal

Each 3D Gaussian primitive is parameterized by:
- Position (mean)
- Covariance matrix
- Color (spherical harmonics)
- Opacity

### 4. Scene Composition with Depth-Aware Occlusion

**Object Filtering Strategy** (multi-stage):
1. Center object at origin
2. Density-based outlier removal for sparse Gaussians
3. Color-based filtering to eliminate background artifacts
   - Kirby: 40th percentile threshold
   - Cat: 70th percentile threshold

**Rendering**: Depth-aware rasterization with automatic occlusion through depth-sorted alpha blending. Object transforms controlled via 4×4 transformation matrices.

### 5. Dynamic Animation and Cinematic Rendering

**Video Specifications**:
- Duration: 14 seconds
- Frame rate: 30fps
- Total frames: 420
- Resolution: 1280×720

**Animation Sequences**:

| Sequence | Duration | Frames | Description |
|----------|----------|--------|-------------|
| 1 | 2s | 60 | Camera movement through scene, Kirby introduction |
| 2 | 3s | 90 | Kirby 1080° rotation with fade-out, Cat fade-in (morph effect) |
| 3 | 4s | 120 | Cat scale transformation (0.5× → 2.0×, 4× total change) |
| 4 | 3s | 90 | 120° orbital camera motion around Cat (-60° to +60°) |
| 5 | 2s | 60 | Cat exit to position [1, 0, 2], camera return to origin |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- COLMAP
- FFmpeg

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Mini-Movie-Making-with-dynamic-3DGS.git
cd Mini-Movie-Making-with-dynamic-3DGS
```

2. Install dependencies:
```bash
bash scripts/install_dependencies.sh
```

3. Install gsplat:
```bash
cd gsplat_repo
pip install -e .
```

## Usage

### 1. Prepare COLMAP Data

Extract frames and run COLMAP Structure-from-Motion:

```bash
python scripts/prepare_colmap.py --video_path <path_to_video> --output_dir <colmap_output>
```

### 2. Train 3DGS Models

Train individual models for scene and objects:

```bash
python scripts/train_model.py --colmap_dir <colmap_output> --output_dir trained_models/<model_name> --iterations 30000
```

### 3. Compose Scene

Combine trained models with filtering and object placement:

```bash
python scripts/compose_scene.py \
  --scene_model trained_models/scene_training \
  --object_models trained_models/curby_training trained_models/cat_training \
  --output_dir composed_models/final_scene
```

### 4. Generate Animation

Create the dynamic mini-movie with JSON-defined animations:

```bash
python scripts/animate_scene.py \
  --composition composed_models/final_scene/composition.json \
  --camera_trajectory animated_movie2/camera_trajectory.json \
  --object_motion animated_movie2/object_motion.json \
  --output_video final_output.mp4
```

## Project Structure

```
.
├── animated_movie2/          # Animation definitions
│   ├── camera_trajectory.json
│   ├── object_motion.json
│   └── metadata.json
├── composed_models/          # Composed scene configurations
│   ├── curby_cat_scene/
│   ├── cat_only_filtered/
│   └── test_tiger_filtered/
├── gsplat_repo/              # 3DGS implementation (gsplat library)
├── scripts/                  # Pipeline scripts
│   ├── prepare_colmap.py
│   ├── train_model.py
│   ├── compose_scene.py
│   ├── animate_scene.py
│   └── sam_segmentation.py
├── trained_models/           # Trained 3DGS checkpoints
│   ├── scene_training/
│   ├── curby_training/
│   └── cat_training/
└── raw_video/                # Source video files
```

## Results

The final mini-movie demonstrates:
- High-quality 3DGS reconstruction with photorealistic rendering
- Smooth object transformations and camera movements
- Proper depth-based occlusion between scene and objects
- Complete narrative flow across five distinct sequences

**Rendering Performance**: 420 frames rendered in approximately 10 minutes on a modern GPU.

## Challenges and Solutions

### Challenge 1: Background Removal

**Problem**: Objects were trained with their backgrounds, causing hazy artifacts when composed into the scene.

**Solution**: Multi-stage filtering approach combining:
- Density-based outlier removal
- Color-based filtering with object-specific percentile thresholds
- Manual threshold tuning for each object

### Challenge 2: Automatic Segmentation Failure

**Problem**: SAM (Segment Anything Model) failed to accurately isolate objects from video frames.

**Solution**: 
- Switched to white background for cleaner capture (Kirby)
- Developed post-training filtering strategy for objects filmed on colored backgrounds (Cat)

## Limitations

1. **Manual filtering required**: Object extraction thresholds need manual tuning for each asset
2. **Rigid transformations only**: Current implementation supports only rigid body transformations (no non-rigid deformation)
3. **Background dependency**: Filtering effectiveness depends on background-object color contrast

## Future Work

- Integration of learning-based segmentation methods for automatic object extraction
- Implementation of 4D Gaussian Splatting for non-rigid deformations
- Real-time rendering optimization
- Interactive animation editing interface
- Support for physics-based object interactions

## Technical Details

### JSON Animation Format

**Camera Trajectory** (`camera_trajectory.json`):
```json
{
  "frames": [
    {
      "frame_id": 0,
      "camera_to_world": [[...], [...], [...], [...]]
    }
  ]
}
```

**Object Motion** (`object_motion.json`):
```json
{
  "objects": [
    {
      "object_id": "curby",
      "keyframes": [
        {
          "frame": 0,
          "position": [x, y, z],
          "rotation": [rx, ry, rz],
          "scale": [sx, sy, sz],
          "opacity": 1.0
        }
      ]
    }
  ]
}
```

## Acknowledgments

- **gsplat library**: Primary 3DGS implementation framework
- **COLMAP**: Structure-from-Motion pipeline
- Filmed at Hana Square, Korea University

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [gsplat: Open-source library for Gaussian Splatting](https://github.com/nerfstudio-project/gsplat)
- [COLMAP: Structure-from-Motion and Multi-View Stereo](https://colmap.github.io/)

## License

This project is for educational purposes. Please check individual component licenses (gsplat, COLMAP) for their respective terms.

---

**Note**: The character "Kirby" is referred to as "curby" in some code implementations due to initial spelling uncertainty during development.
