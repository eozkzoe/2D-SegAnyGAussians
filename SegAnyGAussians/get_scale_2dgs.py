import torch
import numpy as np
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from utils.image_utils import psnr
import json

def extract_scales(scene_path, iteration=None):
    # Load scene
    if iteration is None:
        # Find latest iteration
        checkpoints = [os.path.splitext(f)[0] for f in os.listdir(scene_path) if f.endswith('.ply')]
        iterations = [int(c.split('_')[-1]) for c in checkpoints]
        iteration = max(iterations) if iterations else 7000

    # Set up model parameters
    parser = ArgumentParser(description="Scale extraction for 2D Gaussian Splatting")
    ModelParams.init_parser(parser)
    PipelineParams.init_parser(parser)
    OptimizationParams.init_parser(parser)
    args = parser.parse_args(['--model_path', scene_path])
    
    # Initialize system state
    safe_state(True)
    
    # Create Gaussian model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians)
    
    # Get Gaussian parameters
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    rotation = gaussians.get_rotation.detach().cpu().numpy()
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy()
    
    # Extract scale information from surfels
    scales = []
    for i in range(len(xyz)):
        # Compute scale from surfel parameters
        scale = np.mean(scaling[i])  # Average scale across dimensions
        normal = rotation[i]  # Normal direction from rotation
        
        scales.append({
            'position': xyz[i].tolist(),
            'scale': float(scale),
            'normal': normal.tolist(),
            'opacity': float(opacity[i])
        })
    
    # Save scales to JSON file
    output_path = os.path.join(scene_path, f'scales_{iteration}.json')
    with open(output_path, 'w') as f:
        json.dump(scales, f)
    
    print(f"Saved {len(scales)} scales to {output_path}")
    return scales

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract scales from 2D Gaussian scene")
    parser.add_argument("--scene_path", required=True, help="Path to the scene directory")
    parser.add_argument("--iteration", type=int, help="Iteration to load (default: latest)")
    args = parser.parse_args()
    
    extract_scales(args.scene_path, args.iteration) 