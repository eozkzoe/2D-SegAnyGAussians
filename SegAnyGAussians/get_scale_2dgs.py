import torch
import numpy as np
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from utils.image_utils import psnr
import json
import cv2
from tqdm import tqdm

def get_combined_args(parser: ArgumentParser, model_path: str):
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    # Override model_path with model_path
    args_cmdline.model_path = model_path
    
    target_cfg_file = "cfg_args"
    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found, using defaults")
        pass
    
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
            
    # Add default values for required parameters if not present
    if 'allow_principle_point_shift' not in merged_dict:
        merged_dict['allow_principle_point_shift'] = False
    
    return Namespace(**merged_dict)

def extract_scales(model_path, image_root=None):
    # Verify input.ply exists
    if not os.path.exists(os.path.join(model_path, 'input.ply')):
        raise FileNotFoundError(f"Could not find input.ply in {model_path}")

    # Set up model parameters
    parser = ArgumentParser(description="Scale extraction for 2D Gaussian Splatting")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--image_root", default=None, type=str)
    parser.add_argument("--allow_principle_point_shift", action="store_true", help="Allow camera principle point shift")
    
    args = get_combined_args(parser, model_path)
    
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
        scale = float(np.mean(scaling[i]).item())  # Average scale across dimensions
        normal = rotation[i].tolist()  # Normal direction from rotation
        
        scales.append({
            'position': xyz[i].tolist(),
            'scale': scale,
            'normal': normal,
            'opacity': float(opacity[i].item())  # Properly extract scalar value
        })
    
    # Save scales to appropriate location
    if image_root is not None:
        # If image_root is provided, save in mask_scales directory like original SAGA
        output_dir = os.path.join(image_root, 'mask_scales')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save per-view scales if we have image data
        if os.path.exists(os.path.join(image_root, 'images')):
            cameras = scene.getTrainCameras()
            background = torch.zeros(gaussians.get_xyz.shape[0], 3, device='cuda')
            
            for view in tqdm(cameras, desc="Processing views"):
                # Render view
                render_pkg = render(view, gaussians, pipeline.extract(args), background)
                
                # Get depth information
                depth = render_pkg['surf_depth'].cpu()
                
                # Calculate view-dependent scales
                view_scales = []
                for scale_info in scales:
                    pos = torch.tensor(scale_info['position'])
                    # Project position to view space
                    view_pos = view.world_to_camera(pos.unsqueeze(0))
                    view_depth = view_pos[0, 2].item()
                    
                    # Scale the base scale by depth
                    view_scale = scale_info['scale'] * view_depth
                    view_scales.append(view_scale)
                
                # Save view-specific scales
                torch.save(torch.tensor(view_scales), 
                         os.path.join(output_dir, f"{view.image_name}.pt"))
        
        # Also save the base scales
        with open(os.path.join(output_dir, 'base_scales.json'), 'w') as f:
            json.dump(scales, f)
    else:
        # If no image_root, save in scene directory
        output_path = os.path.join(model_path, 'scales.json')
        with open(output_path, 'w') as f:
            json.dump(scales, f)
    
    print(f"Saved {len(scales)} scales")
    return scales

if __name__ == "__main__":
    parser = ArgumentParser(description="Extract scales from 2D Gaussian scene")
    parser.add_argument("--model_path", required=True, help="Path to the scene directory")
    parser.add_argument("--image_root", default=None, help="Path to image root directory (optional)")
    parser.add_argument("--allow_principle_point_shift", action="store_true", help="Allow camera principle point shift")
    args = parser.parse_args()
    
    extract_scales(args.model_path, args.image_root)