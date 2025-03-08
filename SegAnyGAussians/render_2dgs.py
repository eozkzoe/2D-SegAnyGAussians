import torch
import numpy as np
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
import cv2
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
import open3d as o3d

def render_segmented_mesh(scene_path, mask_path, output_path, iteration=None, mesh_res=1024, num_clusters=50):
    # Load scene
    if iteration is None:
        checkpoints = [os.path.splitext(f)[0] for f in os.listdir(scene_path) if f.endswith('.ply')]
        iterations = [int(c.split('_')[-1]) for c in checkpoints]
        iteration = max(iterations) if iterations else 7000
    
    # Set up model parameters
    parser = ArgumentParser()
    ModelParams.init_parser(parser)
    PipelineParams.init_parser(parser)
    OptimizationParams.init_parser(parser)
    args = parser.parse_args(['--model_path', scene_path])
    
    # Initialize system state
    safe_state(True)
    
    # Create Gaussian model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians)
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32) / 255.0
    
    # Create GaussianExtractor
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    gaussExtractor = GaussianExtractor(gaussians, render, scene.pipe, bg_color=bg_color)
    
    # Set active_sh to 0 for diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    
    # Get cameras for reconstruction
    cameras = scene.getTrainCameras()
    gaussExtractor.reconstruction(cameras)
    
    # Extract mesh
    print("Extracting mesh...")
    mesh = gaussExtractor.extract_mesh_unbounded(resolution=mesh_res)
    
    # Apply segmentation mask
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    
    # Project vertices to mask space
    cam = cameras[0]  # Use first camera for projection
    cam_vertices = gaussExtractor.to_camera(vertices, cam)
    cam_vertices = cam_vertices[:, :2] / cam_vertices[:, 2:3]
    
    # Scale to mask dimensions
    h, w = mask.shape
    cam_vertices[:, 0] = (cam_vertices[:, 0] + 1) * w / 2
    cam_vertices[:, 1] = (cam_vertices[:, 1] + 1) * h / 2
    
    # Sample mask values at vertex positions
    vertex_mask = np.zeros(len(vertices))
    valid = (cam_vertices[:, 0] >= 0) & (cam_vertices[:, 0] < w) & \
            (cam_vertices[:, 1] >= 0) & (cam_vertices[:, 1] < h)
    coords = cam_vertices[valid].astype(int)
    vertex_mask[valid] = mask[coords[:, 1], coords[:, 0]]
    
    # Create segmented meshes
    fg_vertices = vertices[vertex_mask > 0.5]
    fg_colors = vertex_colors[vertex_mask > 0.5]
    
    # Create foreground mesh
    fg_mesh = o3d.geometry.TriangleMesh()
    fg_mesh.vertices = o3d.utility.Vector3dVector(fg_vertices)
    fg_mesh.vertex_colors = o3d.utility.Vector3dVector(fg_colors)
    
    # Reconstruct topology
    fg_mesh = post_process_mesh(fg_mesh, cluster_to_keep=num_clusters)
    
    # Save meshes
    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    fg_path = os.path.join(output_path, f"{base_name}_fg.ply")
    o3d.io.write_triangle_mesh(fg_path, fg_mesh)
    
    print(f"Saved segmented mesh to {fg_path}")
    return fg_mesh

if __name__ == "__main__":
    parser = ArgumentParser(description="Render segmented mesh from 2D Gaussian scene")
    parser.add_argument("--scene_path", required=True, help="Path to the scene directory")
    parser.add_argument("--mask_path", required=True, help="Path to the segmentation mask")
    parser.add_argument("--output_path", required=True, help="Path to save the output meshes")
    parser.add_argument("--iteration", type=int, help="Iteration to load (default: latest)")
    parser.add_argument("--mesh_res", type=int, default=1024, help="Resolution for mesh extraction")
    parser.add_argument("--num_clusters", type=int, default=50, help="Number of clusters to keep in post-processing")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    render_segmented_mesh(
        args.scene_path,
        args.mask_path,
        args.output_path,
        args.iteration,
        args.mesh_res,
        args.num_clusters
    ) 