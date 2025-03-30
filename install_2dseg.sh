cd /workspace/2D-SegAnyGAussians/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

cd /workspace/2D-SegAnyGAussians/2D-gaussian/
pip install submodules/simple-knn
bash update_pkg.sh

cd /workspace/2D-SegAnyGAussians/SegAnyGAussians/
pip install hdbscan
pip install matplotlib
pip install submodules/diff-gaussian-rasterization_contrastive_f
pip install submodules/diff-gaussian-rasterization-depth
pip install third_party/segment-anything
pip install dearpygui
pip install open-clip-torch
pip install joblib==1.1.0
