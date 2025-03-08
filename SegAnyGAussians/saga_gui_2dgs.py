import os
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
import json
from PIL import Image
import cv2
from typing import List, Tuple, Dict
import torch.nn.functional as F

class SAGAGUI:
    def __init__(self, scene_path: str, feature_path: str, iteration: int = None):
        self.scene_path = scene_path
        self.feature_path = feature_path
        
        # Load scene
        if iteration is None:
            checkpoints = [os.path.splitext(f)[0] for f in os.listdir(scene_path) if f.endswith('.ply')]
            iterations = [int(c.split('_')[-1]) for c in checkpoints]
            self.iteration = max(iterations) if iterations else 7000
        else:
            self.iteration = iteration
            
        # Set up model parameters
        parser = ArgumentParser()
        ModelParams.init_parser(parser)
        PipelineParams.init_parser(parser)
        OptimizationParams.init_parser(parser)
        self.args = parser.parse_args(['--model_path', scene_path])
        
        # Initialize system state
        safe_state(True)
        
        # Create Gaussian model
        self.gaussians = GaussianModel(self.args.sh_degree)
        self.scene = Scene(self.args, self.gaussians)
        
        # Load features
        checkpoint = torch.load(feature_path)
        self.features = checkpoint['features']
        
        # Initialize GUI state
        self.selected_points = set()
        self.view_params = {
            'camera_distance': 2.0,
            'camera_angle': 0.0,
            'camera_height': 0.0
        }
        self.current_mask = None
        self.current_view = None
        self.current_depth = None
        self.current_normal = None
        
        # Initialize GUI
        self.init_gui()
        
    def init_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="SAGA GUI - 2D Gaussian Splatting", width=1280, height=720)
        
        # Main window
        with dpg.window(label="SAGA Controls", width=300, height=720, pos=(0, 0)):
            # View controls
            with dpg.collapsing_header(label="View Controls", default_open=True):
                dpg.add_slider_float(label="Camera Distance", default_value=2.0,
                                   min_value=0.5, max_value=5.0,
                                   callback=self.update_view_params)
                dpg.add_slider_float(label="Camera Angle", default_value=0.0,
                                   min_value=-180.0, max_value=180.0,
                                   callback=self.update_view_params)
                dpg.add_slider_float(label="Camera Height", default_value=0.0,
                                   min_value=-2.0, max_value=2.0,
                                   callback=self.update_view_params)
            
            # Selection controls
            with dpg.collapsing_header(label="Selection Controls", default_open=True):
                dpg.add_button(label="Clear Selection", callback=self.clear_selection)
                dpg.add_button(label="Grow Selection", callback=self.grow_selection)
                dpg.add_button(label="Save Mask", callback=self.save_mask)
            
            # Visualization controls
            with dpg.collapsing_header(label="Visualization", default_open=True):
                dpg.add_radio_button(items=["RGB", "Depth", "Normal"], default_value="RGB",
                                   callback=self.update_visualization)
        
        # Render window
        with dpg.window(label="Render View", width=960, height=720, pos=(300, 0)):
            with dpg.texture_registry():
                dpg.add_raw_texture(width=960, height=720, default_value=np.zeros((720, 960, 4), dtype=np.float32),
                                  format=dpg.mvFormat_Float_rgba, tag="render_texture")
            
            dpg.add_image("render_texture", width=960, height=720)
            with dpg.item_handler_registry(tag="image_handler"):
                dpg.add_item_clicked_handler(callback=self.handle_click)
            dpg.bind_item_handler_registry("render_texture", "image_handler")
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        
        # Initial render
        self.update_view()
        
    def update_view_params(self, sender, app_data):
        param_name = dpg.get_item_label(sender).lower().replace(" ", "_")
        self.view_params[param_name] = app_data
        self.update_view()
        
    def update_view(self):
        # Create camera parameters
        distance = self.view_params['camera_distance']
        angle = np.radians(self.view_params['camera_angle'])
        height = self.view_params['camera_height']
        
        camera_position = torch.tensor([
            distance * np.cos(angle),
            height,
            distance * np.sin(angle)
        ], device='cuda')
        
        # Render scene
        render_pkg = render(camera_position, self.gaussians, self.scene.pipe, self.scene.background)
        
        self.current_view = render_pkg["render"]
        self.current_depth = render_pkg["surf_depth"]
        self.current_normal = render_pkg["surf_normal"] * 0.5 + 0.5
        
        # Update visualization
        self.update_visualization()
        
    def update_visualization(self, sender=None, app_data=None):
        if sender is not None:
            vis_mode = app_data
        else:
            vis_mode = "RGB"
            
        if vis_mode == "RGB":
            image = self.current_view
        elif vis_mode == "Depth":
            image = self.current_depth.repeat(3, 1, 1)
        else:  # Normal
            image = self.current_normal
            
        # Convert to numpy and apply mask highlight
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        if self.current_mask is not None:
            image = image * 0.7 + image * self.current_mask[:, :, None] * 0.3
            
        # Add alpha channel
        image = np.concatenate([image, np.ones_like(image[:, :, :1])], axis=2)
        
        # Update texture
        dpg.set_value("render_texture", image.astype(np.float32))
        
    def handle_click(self, sender, app_data):
        if not dpg.is_item_hovered("render_texture"):
            return
            
        # Get click position
        mouse_pos = dpg.get_mouse_pos()
        rel_pos = dpg.get_item_pos("render_texture")
        click_x = (mouse_pos[0] - rel_pos[0]) / 960
        click_y = (mouse_pos[1] - rel_pos[1]) / 720
        
        # Find closest point in view space
        with torch.no_grad():
            view_points = self.current_view.view(3, -1).t()
            click_point = torch.tensor([click_x, click_y], device=view_points.device)
            
            distances = torch.norm(view_points[:, :2] - click_point, dim=1)
            closest_idx = distances.argmin().item()
            
            # Toggle point selection
            if closest_idx in self.selected_points:
                self.selected_points.remove(closest_idx)
            else:
                self.selected_points.add(closest_idx)
            
            # Update mask
            self.update_mask()
            
    def update_mask(self):
        if not self.selected_points:
            self.current_mask = None
            self.update_visualization()
            return
            
        # Get features of selected points
        selected_features = self.features[list(self.selected_points)]
        
        # Compute similarity to all points
        similarities = F.cosine_similarity(
            self.features.unsqueeze(0),
            selected_features.mean(0).unsqueeze(0)
        )
        
        # Create mask
        mask = (similarities > 0.8).float()
        self.current_mask = mask.view(self.current_view.shape[1], self.current_view.shape[2])
        
        # Update visualization
        self.update_visualization()
        
    def clear_selection(self):
        self.selected_points.clear()
        self.current_mask = None
        self.update_visualization()
        
    def grow_selection(self):
        if self.current_mask is None:
            return
            
        # Find points with high similarity to current selection
        mask_indices = self.current_mask.nonzero().squeeze()
        if mask_indices.dim() == 0:
            return
            
        selected_features = self.features[mask_indices]
        mean_feature = selected_features.mean(0)
        
        # Find new similar points
        similarities = F.cosine_similarity(self.features, mean_feature.unsqueeze(0))
        new_mask = (similarities > 0.7).float()
        
        # Update mask
        self.current_mask = new_mask.view(self.current_view.shape[1], self.current_view.shape[2])
        self.update_visualization()
        
    def save_mask(self):
        if self.current_mask is None:
            print("No mask to save")
            return
            
        # Save mask as PNG
        mask_image = (self.current_mask.cpu().numpy() * 255).astype(np.uint8)
        mask_path = os.path.join(self.scene_path, f'mask_{len(os.listdir(self.scene_path))}.png')
        cv2.imwrite(mask_path, mask_image)
        print(f"Saved mask to {mask_path}")
        
    def run(self):
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

if __name__ == "__main__":
    parser = ArgumentParser(description="SAGA GUI for 2D Gaussian Splatting")
    parser.add_argument("--scene_path", required=True, help="Path to the scene directory")
    parser.add_argument("--feature_path", required=True, help="Path to the trained features")
    parser.add_argument("--iteration", type=int, help="Iteration to load (default: latest)")
    args = parser.parse_args()
    
    gui = SAGAGUI(args.scene_path, args.feature_path, args.iteration)
    gui.run() 