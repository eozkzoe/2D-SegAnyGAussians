import torch
import numpy as np
from argparse import ArgumentParser, Namespace
import os
from tqdm import tqdm
import cv2
from gaussian_2d.gaussian_model import GaussianModel2D
from arguments import ModelParams, PipelineParams

def get_combined_args(parser: ArgumentParser):
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    target_cfg_file = "cfg_args"
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found")
        pass
    
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
    grid = torch.stack(grid, dim=-1)
    return grid

if __name__ == '__main__':
    parser = ArgumentParser(description="Get scales for SAM masks (2D Gaussian version)")
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)
    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)

    args = get_combined_args(parser)
    
    # Initialize 2D Gaussian model with the same parameters as 3D
    gaussians = GaussianModel2D(dataset.sh_degree)
    gaussians.load_ply(args.model_path)
    
    # Get 2D specific parameters
    xyz = gaussians.get_xyz
    scaling = gaussians.get_scaling
    rotation = gaussians.get_rotation
    
    # Calculate scale differently for 2D Gaussians
    scales = []
    for i in range(len(xyz)):
        scale = float(np.mean(scaling[i]).item())
        normal = rotation[i].tolist()
        scales.append({
            'position': xyz[i].tolist(),
            'scale': scale,
            'normal': normal
        })
    
    return scales