#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import open3d as o3d
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import fov2focal, getWorld2View


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    colors = []
    depths = []
    poses = []
    K = torch.zeros(4, 4)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx == 20:
        #     break
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        colors.append(rendering)
        depths.append(render_pkg["depth"])
        pose = getWorld2View(view.R, view.T)
        poses.append(pose)
        K = visualize_pc(render_pkg, view, idx)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
    torch.save(depths, "test/data/depths.pt")
    torch.save(colors, "test/data/colors.pt")
    torch.save(poses, "test/data/poses.pt")
    torch.save(K, "test/data/K.pt")
    mesh_extraction(depths, colors, K, poses)


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )


def visualize_pc(render_pkg, view, idx):
    # def visualize(idx=0):
    """
    Input:
    rendering_pkg: dict
        rendering_pkg["render"]: torch.Tensor, shape (3, H, W)
        rendering_pkg["depth"]: torch.Tensor, shape (1, H, W)
    view: Camera object
    idx: int

    return: torch.Tensor, shape (4, 4)
    """
    rendering = render_pkg["render"]
    depth = render_pkg["depth"]
    scale = view.scale
    H, W = rendering.shape[1:]
    # camera intrinsic
    fx = fov2focal(view.FoVx, W)
    fy = fov2focal(view.FoVy, H)
    cx = W / 2
    cy = H / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # torch.save(rendering, 'test/rendering.pt')
    # torch.save(depth, 'test/depth.pt')
    # torch.save(K, 'test/K.pt')
    # load data
    # rendering = torch.load('test/rendering.pt')
    # depth = torch.load('test/depth.pt')
    # K = torch.load('test/K.pt')
    # fx=K[0,0]
    # fy=K[1,1]
    # cx=K[0,2]
    # cy=K[1,2]
    # W=rendering.shape[2]
    # H=rendering.shape[1]
    x = torch.arange(W).reshape(1, -1).repeat(H, 1)
    y = torch.arange(H).reshape(-1, 1).repeat(1, W)
    X = (x - cx) / fx
    Y = (y - cy) / fy

    Z = depth.squeeze().cpu().numpy()
    X = (X * Z).numpy()
    Y = (Y * Z).numpy()
    # create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    colors = rendering.permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
    colors_clip = np.clip(colors, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors_clip)
    o3d.io.write_point_cloud(f"test/point_cloud_result/pc_{idx}.ply", pcd)
    return K


def mesh_extraction(depths, colors, K, RTs):
    # Create necessary directories
    os.makedirs("test/data", exist_ok=True)
    os.makedirs("test/mesh_result", exist_ok=True)

    # Voxel size used in TSDF fusion.
    voxel_length = 0.004
    # Truncation threshold for TSDF.
    sdf_trunc = 0.02

    try:
        # Initialize a TSDF volume
        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        H, W = depths[0].shape[1:]
        img_num = len(depths)
        idx = np.arange(0, min(190, img_num), 5)  # Ensure we don't exceed array bounds

        # Process each depth image
        for i in idx:
            depth = depths[i]
            RT = RTs[i]
            color = colors[i]
            print("Integrate {:d}-th image into the volume.".format(i))

            # Ensure data is on CPU and in correct format
            color_np = color.reshape(H, W, 3).cpu().numpy()
            color_clip = np.clip(color_np, 0.0, 1.0) * 255
            depth_np = depth.squeeze(0).cpu().numpy()

            # Skip if depth is invalid
            if np.isnan(depth_np).any() or np.isinf(depth_np).any():
                print(f"Skipping frame {i} due to invalid depth values")
                continue

            depth_o3d = o3d.geometry.Image(depth_np.astype(np.float32))
            color_o3d = o3d.geometry.Image(color_clip.astype(np.uint8))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=5.0,  # Add depth truncation
                convert_rgb_to_intensity=False,
            )

            # Create a camera intrinsic object
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
            intrinsic_o3d.set_intrinsics(
                W, H, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
            )

            # Ensure RT matrix is in correct format
            RT_o3d = np.array(RT).astype(np.float64)

            tsdf_volume.integrate(rgbd_image, intrinsic_o3d, RT_o3d)

        # Extract and optimize the mesh
        print("Extracting mesh...")
        mesh = tsdf_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Save the mesh
        print("Saving mesh...")
        o3d.io.write_triangle_mesh(f"test/mesh_result/mesh_{img_num}.ply", mesh)
        print(
            f"Mesh saved successfully with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles"
        )

    except Exception as e:
        print(f"Error during mesh extraction: {str(e)}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian Splatting Visualization")
    parser.add_argument("--load", action="store_true", help="Load pre-rendered data")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument(
        "--iteration", default=-1, type=int, help="Model iteration to load"
    )
    parser.add_argument(
        "--skip_train", action="store_true", help="Skip training set rendering"
    )
    parser.add_argument(
        "--skip_test", action="store_true", help="Skip test set rendering"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.004, help="Voxel size for TSDF fusion"
    )
    parser.add_argument(
        "--sdf_trunc", type=float, default=0.02, help="Truncation value for TSDF fusion"
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=5,
        help="Stride for selecting frames in mesh extraction",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument(
        "--output_dir", type=str, default="test", help="Output directory"
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(f"{args.output_dir}/data", exist_ok=True)
    os.makedirs(f"{args.output_dir}/point_cloud_result", exist_ok=True)
    os.makedirs(f"{args.output_dir}/mesh_result", exist_ok=True)

    if args.load:
        try:
            depths = torch.load(f"{args.output_dir}/data/depths.pt")
            poses = torch.load(f"{args.output_dir}/data/poses.pt")
            colors = torch.load(f"{args.output_dir}/data/colors.pt")
            K = torch.load(f"{args.output_dir}/data/K.pt")
            mesh_extraction(depths, colors, K, poses)
        except Exception as e:
            print(f"Error loading pre-rendered data: {str(e)}")
    else:
        try:
            # Set up model parameters
            model = ModelParams(parser, sentinel=True)
            pipeline = PipelineParams(parser)
            model_args = get_combined_args(parser)

            if not args.quiet:
                print("Rendering " + args.model_path)

            # Initialize system state (RNG)
            safe_state(args.quiet)

            # Render the scenes
            render_sets(
                model.extract(model_args),
                args.iteration,
                pipeline.extract(model_args),
                args.skip_train,
                args.skip_test,
            )
        except Exception as e:
            print(f"Error during rendering: {str(e)}")
