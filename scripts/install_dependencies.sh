#!/bin/bash
# Install all required dependencies for 3DGS training

set -e

echo "Installing dependencies..."

# Install remaining Python packages
pip install viser \
    imageio[ffmpeg] \
    opencv-python \
    tyro \
    "torchmetrics[image]" \
    tensorboard \
    pyyaml \
    matplotlib \
    tqdm \
    scikit-learn \
    tensorly \
    splines \
    Pillow

# Install nerfview
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"

# Install gsplat in editable mode
cd /teamspace/lightning_storage/final_project_3DGS/gsplat_repo
pip install --no-build-isolation -e .

echo "All dependencies installed successfully!"
