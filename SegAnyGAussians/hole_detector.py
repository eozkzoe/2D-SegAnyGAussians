import os
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from scene import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal

class HoleDetector:
    def __init__(self, scene_path, mask_path, output_dir="./hole_detection_results"):
        self.scene_path = scene_path
        self.mask_path = mask_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the scene
        self.gaussian_model = GaussianModel(3)  # sh_degree=3
        self.gaussian_model.load_ply(scene_path)
        
        # Load the mask
        self.mask = torch.load(mask_path)
        if torch.count_nonzero(self.mask) == 0:
            print("Mask is empty, inverting mask")
            self.mask = ~self.mask
        
        # Get scene bounds for camera placement
        self.xyz = self.gaussian_model.get_xyz
        self.center = self.xyz.mean(dim=0)
        self.scale = (self.xyz.max(dim=0).values - self.xyz.min(dim=0).values).max()
        
        # Rendering parameters
        self.width = 800
        self.height = 800
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # Optimization parameters
        self.best_circularity = 0.0
        self.best_camera = None
        self.best_ellipse = None
        self.best_render = None

    def create_camera(self, position, target, up_vector, fovy=60):
        """Create a camera at a specific position looking at a target"""
        position = np.array(position)
        target = np.array(target)
        up_vector = np.array(up_vector)
        
        # Calculate camera orientation
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix
        R_mat = np.stack([right, up, -forward], axis=1)
        t = -R_mat @ position
        
        # Create camera
        ss = math.pi / 180.0
        fovy_rad = fovy * ss
        
        fy = fov2focal(fovy_rad, self.height)
        fovx = focal2fov(fy, self.width)
        
        cam = Camera(
            colmap_id=0,
            R=R_mat,
            T=t,
            FoVx=fovx,
            FoVy=fovy_rad,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam
    
    def generate_viewpoint(self, random_offset=True, iteration=0):
        """Generate a camera viewpoint looking at the scene center"""
        # Calculate distance based on scene scale
        distance = self.scale * 2.0
        
        if iteration == 0 and not random_offset:
            # First try: look along Z axis
            camera_pos = self.center + torch.tensor([0, 0, distance], device="cuda")
            up_vector = np.array([0, 1, 0])
        else:
            # Random viewpoint around the object
            if random_offset:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                
                x = distance * np.sin(phi) * np.cos(theta)
                y = distance * np.sin(phi) * np.sin(theta)
                z = distance * np.cos(phi)
                
                camera_pos = self.center + torch.tensor([x, y, z], device="cuda")
                up_vector = np.array([0, 1, 0])
            else:
                # Systematic exploration around Z axis
                angle = (iteration / 8) * 2 * np.pi
                rotation = R.from_rotvec(angle * np.array([0, 1, 0]))
                offset_dir = rotation.apply(np.array([1, 0, 0]))
                camera_pos = self.center + torch.tensor(
                    offset_dir * distance * 0.6 + np.array([0, 0, distance * 0.8]), 
                    device="cuda"
                )
                up_vector = np.array([0, 1, 0])
        
        return self.create_camera(camera_pos.cpu().numpy(), self.center.cpu().numpy(), up_vector)
    
    def detect_ellipse(self, image):
        """Detect ellipses in the rendered image"""
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to isolate the object
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for holes (contours with holes inside)
        for contour in contours:
            # Check if contour is large enough
            area = cv2.contourArea(contour)
            if area < 100:  # Skip tiny contours
                continue
                
            # Create a mask for this contour
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Check if there's a hole inside
            # Invert the mask to find holes
            mask_inv = cv2.bitwise_not(mask)
            hole_contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small holes and those touching the border
            for hole in hole_contours:
                hole_area = cv2.contourArea(hole)
                if hole_area < 50 or hole_area > area * 0.9:  # Skip tiny holes or those too large
                    continue
                
                # Check if hole is not touching the border
                x, y, w, h = cv2.boundingRect(hole)
                if x <= 1 or y <= 1 or x + w >= binary.shape[1] - 1 or y + h >= binary.shape[0] - 1:
                    continue
                
                # Try to fit an ellipse
                if len(hole) >= 5:  # Need at least 5 points to fit an ellipse
                    try:
                        ellipse = cv2.fitEllipse(hole)
                        (x, y), (width, height), angle = ellipse
                        
                        # Calculate circularity (1.0 for perfect circle)
                        circularity = min(width, height) / max(width, height)
                        
                        return {
                            "ellipse": ellipse,
                            "circularity": circularity,
                            "center": (float(x), float(y)),
                            "width": float(width),
                            "height": float(height),
                            "angle": float(angle)
                        }
                    except:
                        continue
        
        # If we get here, no suitable ellipse was found
        return None
    
    def optimize_viewpoint(self, ellipse_info, camera):
        """Optimize the camera position to get a more circular view of the hole"""
        # Extract ellipse parameters
        (x, y), (width, height), angle = ellipse_info["ellipse"]
        
        # Calculate the center of the ellipse in normalized device coordinates
        center_x = (x / self.width) * 2 - 1
        center_y = (y / self.height) * 2 - 1
        
        # Get camera parameters
        cam_pos = -np.linalg.inv(camera.R) @ camera.T
        cam_dir = self.pose["centroid"] - cam_pos
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        
        # Calculate the major axis direction of the ellipse
        major_axis_angle = angle * np.pi / 180.0
        major_axis = np.array([np.cos(major_axis_angle), np.sin(major_axis_angle), 0])
        
        # Calculate a new camera position that's more aligned with the hole normal
        # Move in the direction perpendicular to the major axis
        move_dir = np.array([center_x, center_y, 0])
        move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-6)
        
        # Adjust camera position
        new_cam_pos = cam_pos + move_dir * 0.1  # Small step
        
        # Create a new camera
        return self.create_camera(new_cam_pos, self.pose["centroid"], np.array([0, 1, 0]))
    
    def render_view(self, camera):
        """Render the scene from a given camera viewpoint"""
        # Create pipeline parameters
        class PipelineParams:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False
        
        pipeline = PipelineParams()
        
        # Apply the mask to the scene
        original_mask = self.gaussian_model._mask.clone()
        self.gaussian_model._mask = self.mask
        
        # Render
        with torch.no_grad():
            outputs = render(camera, self.gaussian_model, pipeline, self.bg_color)
        
        # Restore original mask
        self.gaussian_model._mask = original_mask
        
        # Get the rendered image
        img = outputs["render"].permute(1, 2, 0).cpu().numpy()
        
        return img
    
    def detect_hole(self, max_iterations=20, max_optimizations=5):
        """
        Detect a circular hole in the segmented object
        """
        print(f"Detecting holes in segmented object...")
        
        # Try different viewpoints
        for i in tqdm(range(max_iterations)):
            camera = self.generate_viewpoint(random_offset=(i > 8), iteration=i)
            render_img = self.render_view(camera)
            ellipse_info = self.detect_ellipse(render_img)
            
            if ellipse_info is not None:
                print(f"Ellipse detected with circularity: {ellipse_info['circularity']:.3f}")
                
                # Save initial detection
                if ellipse_info["circularity"] > self.best_circularity:
                    self.best_circularity = ellipse_info["circularity"]
                    self.best_camera = camera
                    self.best_ellipse = ellipse_info
                    self.best_render = render_img
                
                # Try to optimize the viewpoint
                current_camera = camera
                current_ellipse = ellipse_info
                current_render = render_img
                
                for j in range(max_optimizations):
                    # If circularity is already good, stop optimizing
                    if current_ellipse["circularity"] > 0.95:
                        break
                    
                    # Optimize viewpoint
                    new_camera = self.optimize_viewpoint(current_ellipse, current_camera)
                    
                    # Render new view
                    new_render = self.render_view(new_camera)
                    
                    # Detect ellipse in new view
                    new_ellipse = self.detect_ellipse(new_render)
                    
                    if new_ellipse is None:
                        break
                    
                    print(f"  Optimization step {j+1}: circularity = {new_ellipse['circularity']:.3f}")
                    
                    # If new view is better, update
                    if new_ellipse["circularity"] > current_ellipse["circularity"]:
                        current_camera = new_camera
                        current_ellipse = new_ellipse
                        current_render = new_render
                        
                        # Update best if this is better
                        if current_ellipse["circularity"] > self.best_circularity:
                            self.best_circularity = current_ellipse["circularity"]
                            self.best_camera = current_camera
                            self.best_ellipse = current_ellipse
                            self.best_render = current_render
                    else:
                        # No improvement, stop optimizing
                        break
                
                # If we found a very good circle, stop early
                if self.best_circularity > 0.95:
                    break
        
        # Save results
        if self.best_ellipse is not None:
            self.save_results()
            return {
                "found": True,
                "circularity": self.best_circularity,
                "center": self.best_ellipse["center"],
                "width": self.best_ellipse["width"],
                "height": self.best_ellipse["height"],
                "camera_position": -np.linalg.inv(self.best_camera.R) @ self.best_camera.T,
                "camera_direction": self.center.cpu().numpy() - (-np.linalg.inv(self.best_camera.R) @ self.best_camera.T)
            }
        else:
            return {"found": False}
    
    def save_results(self):
        """Save detection results"""
        if self.best_ellipse is None:
            return
            
        # Save the best render with ellipse visualization
        output_img = self.best_render.copy()
        
        # Draw the detected ellipse
        cv2.ellipse(
            (output_img * 255).astype(np.uint8),
            self.best_ellipse["ellipse"],
            (0, 255, 0),
            2
        )
        
        # Save visualization
        cv2.imwrite(
            os.path.join(self.output_dir, "best_view.png"),
            cv2.cvtColor((output_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        
        # Save detection results
        results = {
            "circularity": float(self.best_circularity),
            "ellipse": {
                "center": self.best_ellipse["center"],
                "width": float(self.best_ellipse["width"]),
                "height": float(self.best_ellipse["height"]),
                "angle": float(self.best_ellipse["angle"])
            },
            "camera": {
                "position": (-np.linalg.inv(self.best_camera.R) @ self.best_camera.T).tolist(),
                "direction": (self.pose["centroid"] - (-np.linalg.inv(self.best_camera.R) @ self.best_camera.T)).tolist(),
                "up": self.best_camera.R[:, 1].tolist()
            },
            "object": {
                "centroid": self.pose["centroid"].tolist(),
                "normal": self.pose["normal"].tolist(),
                "bbox_dimensions": self.bbox["dimensions"].tolist()
            }
        }
        
        with open(os.path.join(self.output_dir, "hole_detection.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to {self.output_dir}")
        print(f"Best circularity: {self.best_circularity:.3f}")