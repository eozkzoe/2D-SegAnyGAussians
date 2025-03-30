#!/bin/bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

cd ./2D-gaussian/
pip install -r requirements.txt
pip install submodules/simple-knn
bash update_pkg.sh

cd ./SegAnyGAussians/
pip install hdbscan
pip install matplotlib
pip install submodules/diff-gaussian-rasterization_contrastive_f
pip install submodules/diff-gaussian-rasterization-depth
pip install third_party/segment-anything
pip install dearpygui
pip install open-clip-torch
pip install joblib==1.1.0
