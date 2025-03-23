import os
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from scene import Scene, GaussianModel
from gaussian_renderer import render, render_mask
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter


class HoleDetector:
    def __init__(
        self,
        scene_path,
        model_path,
        mask_path,
        output_dir="./hole_detection_results",
        debug=False,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Rendering parameters
        self.width = 800
        self.height = 800
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # Load the scene and cameras
        class DummyArgs:
            def __init__(self):
                self.source_path = scene_path  # COLMAP directory
                self.model_path = model_path  # Pre-trained model directory
                self.images = "images"
                self.eval = False
                self.sh_degree = 3
                self.white_background = False
                self.feature_dim = 256
                self.load_iteration = -1
                self.allow_principle_point_shift = False
                self.need_features = False
                self.need_masks = False
                self.resolution = 1
                self.data_device = "cuda"

        args = DummyArgs()
        self.gaussian_model = GaussianModel(3)  # sh_degree=3
        self.gaussian_model.load_ply(
            os.path.join(model_path, "point_cloud", "iteration_9000", "point_cloud.ply")
        )
        scene = Scene(args, self.gaussian_model, None, load_iteration=-1, shuffle=False)

        # Get cameras from scene
        self.cameras = scene.getTrainCameras()
        if not self.cameras:
            print("Warning: No scene cameras found, will use generated viewpoints only")

        # Load the mask and ensure it matches Gaussian count
        self.mask = torch.load(mask_path)
        if torch.count_nonzero(self.mask) == 0:
            print("Mask is empty, inverting mask")
            self.mask = ~self.mask

        # Get the number of Gaussians
        n_gaussians = len(self.gaussian_model.get_xyz)

        # Resize mask if needed
        if len(self.mask.flatten()) != n_gaussians:
            print(f"Resizing mask from {len(self.mask.flatten())} to {n_gaussians}")
            # Take first n_gaussians elements if mask is too large, or pad with False if too small
            flat_mask = self.mask.flatten()
            if len(flat_mask) > n_gaussians:
                self.mask = flat_mask[:n_gaussians]
            else:
                self.mask = torch.cat(
                    [
                        flat_mask,
                        torch.zeros(n_gaussians - len(flat_mask), dtype=torch.bool),
                    ]
                )
            self.mask = self.mask.to(device="cuda")

        # Get scene bounds for camera placement
        self.xyz = self.gaussian_model.get_xyz.detach()
        self.center = self.xyz.mean(dim=0)

        # Calculate radius that encompasses 80% of points
        distances = torch.norm(self.xyz - self.center.unsqueeze(0), dim=1)
        self.radius = torch.quantile(distances, 0.8)  # 80th percentile
        self.scale = (self.xyz.max(dim=0).values - self.xyz.min(dim=0).values).max()

        # Optimization parameters
        self.best_circularity = 0.0
        self.best_camera = None
        self.best_ellipse = None
        self.best_render = None
        self.debug = debug
        self.debug_renders = []  # Store debug renders

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
        ss = np.pi / 180.0
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

    def generate_viewpoint(self, random_offset=True, iteration=0, max_iterations=20):
        """Generate a camera viewpoint looking at the scene center"""
        # Use the 80th percentile radius instead of max scale
        distance = float(self.radius.cpu()) * 2.0  # Multiply by 2 for better visibility

        if iteration == 0 and not random_offset:
            # First try: look along Z axis
            camera_pos = self.center + torch.tensor([0, 0, distance], device="cuda")
            up_vector = np.array([0, 1, 0])
        else:
            # Random viewpoint around the object
            if random_offset:
                # Generate points on unit sphere using fibonacci spiral
                golden_ratio = (1 + 5**0.5) / 2
                i = iteration + np.random.uniform(-0.5, 0.5)  # Add some randomness
                theta = 2 * np.pi * i / golden_ratio
                phi = np.arccos(1 - 2 * (i + 0.5) / max_iterations)

                x = distance * np.sin(phi) * np.cos(theta)
                y = distance * np.sin(phi) * np.sin(theta)
                z = distance * np.cos(phi)

                camera_pos = self.center + torch.tensor([x, y, z], device="cuda")
                up_vector = np.array([0, 1, 0])
            else:
                # Systematic exploration around Z axis at 45-degree elevation
                angle = (iteration / 8) * 2 * np.pi
                phi = np.pi / 4  # 45-degree elevation

                x = distance * np.sin(phi) * np.cos(angle)
                y = distance * np.sin(phi) * np.sin(angle)
                z = distance * np.cos(phi)

                camera_pos = self.center + torch.tensor([x, y, z], device="cuda")
                up_vector = np.array([0, 1, 0])

        return self.create_camera(
            camera_pos.detach().cpu().numpy(),
            self.center.detach().cpu().numpy(),
            up_vector,
        )

    def detect_ellipse(self, image):
        """Detect elliptical holes in the rendered image using Hough Ellipse Transform"""
        # Convert to grayscale and uint8
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Downscale image significantly for initial processing
        scale_factor = 0.25  # Process at 1/4 resolution
        height, width = int(gray.shape[0] * scale_factor), int(gray.shape[1] * scale_factor)
        small_gray = cv2.resize(gray, (width, height))
        
        # Apply stronger blur to reduce noise and edges
        blurred = cv2.GaussianBlur(small_gray, (5, 5), 3)
        
        # More aggressive edge detection parameters
        edges = canny(blurred, sigma=2.0, low_threshold=20, high_threshold=80)

        if self.debug:
            edge_debug = (edges * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_dir, "edge_detection_debug.png"), edge_debug)
        
        # Detect ellipses with optimized parameters
        result = hough_ellipse(
            edges,
            accuracy=15,      # Reduced accuracy for speed
            threshold=25,     # Lower threshold to detect more candidates
            min_size=30,      # Adjusted for scaled image
            max_size=int(min(height, width) // 2)
        )

        if result:
            # Scale back the results to original image size
            best = list(result)[0]
            yc, xc, a, b, orientation = best[1], best[0], best[3], best[2], best[4]
            
            # Scale coordinates back to original image size
            xc /= scale_factor
            yc /= scale_factor
            a /= scale_factor
            b /= scale_factor

            # Get the best ellipse (first result)
            best = list(result)[0]
            yc, xc, a, b, orientation = best[1], best[0], best[3], best[2], best[4]

            # Calculate circularity (ratio of minor to major axis)
            # Perfect circle has circularity of 1.0
            circularity = min(a, b) / max(a, b)

            # Draw the ellipse for debug visualization
            if self.debug:
                debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                rr, cc = ellipse_perimeter(
                    int(yc), int(xc), int(a), int(b), orientation=orientation
                )
                # Keep only valid coordinates
                valid = (rr >= 0) & (rr < self.height) & (cc >= 0) & (cc < self.width)
                rr, cc = rr[valid], cc[valid]
                debug_img[rr, cc] = [0, 255, 0]  # Green color
                cv2.imwrite(
                    os.path.join(self.output_dir, "ellipse_detection_debug.png"),
                    debug_img,
                )

            return {
                "ellipse": (
                    (float(xc), float(yc)),
                    (float(a * 2), float(b * 2)),
                    float(orientation * 180 / np.pi),
                ),
                "circularity": float(circularity),
                "center": (float(xc), float(yc)),
                "width": float(a * 2),
                "height": float(b * 2),
                "angle": float(orientation * 180 / np.pi),
                "radius": float((a + b) / 2),
            }

        # If no ellipses found
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
        cam_dir = self.center.cpu().numpy() - cam_pos
        cam_dir = cam_dir / np.linalg.norm(cam_dir)

        # Calculate a new camera position that's more aligned with the hole normal
        # Move in the direction perpendicular to the major axis
        move_dir = np.array([center_x, center_y, 0])
        move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-6)

        # Adjust camera position
        new_cam_pos = cam_pos + move_dir * 0.1  # Small step

        # Create a new camera
        return self.create_camera(
            new_cam_pos, self.center.cpu().numpy(), np.array([0, 1, 0])
        )

    def render_view(self, camera):
        """Render the scene from a given camera viewpoint"""

        # Create pipeline parameters
        class PipelineParams:
            def __init__(self):
                self.convert_SHs_python = False
                self.compute_cov3D_python = False
                self.debug = False

        pipeline = PipelineParams()

        # Convert mask to float for render_mask
        float_mask = self.mask.float()

        # Render the masked view using render_mask
        with torch.no_grad():
            mask_res = render_mask(
                camera,
                self.gaussian_model,
                pipeline,
                self.bg_color,
                precomputed_mask=float_mask,
            )
            outputs = render(camera, self.gaussian_model, pipeline, self.bg_color)

        # Apply the rendered mask to the image
        rendering = outputs["render"]
        mask = mask_res["mask"]
        mask[mask < 0.5] = 0
        mask[mask != 0] = 1
        rendering = rendering * mask

        # Get the rendered image
        img = rendering.permute(1, 2, 0).cpu().numpy()

        return img

    def detect_hole(self, max_iterations=20, max_optimizations=5):
        print(f"Detecting holes in segmented object...")
        all_detections = []

        # First try existing scene cameras
        if self.cameras:
            print("Trying existing scene cameras...")
            for i, camera in enumerate(tqdm(self.cameras)):
                render_img = self.render_view(camera)

                # Save all renders
                if self.debug:
                    debug_img = render_img.copy()
                    cv2.putText(
                        debug_img,
                        f"Scene View {i}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        3,
                    )
                    cv2.putText(
                        debug_img,
                        f"Scene View {i}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (1, 1, 1),
                        1,
                    )
                    self.debug_renders.append(debug_img)

                ellipse_info = self.detect_ellipse(render_img)
                if ellipse_info is not None:
                    print(
                        f"Hole detected in scene camera {i} with circularity: {ellipse_info['circularity']:.3f}"
                    )
                    all_detections.append(
                        {
                            "circularity": ellipse_info["circularity"],
                            "camera": camera,
                            "ellipse": ellipse_info,
                            "render": render_img,
                        }
                    )

        # Try generated viewpoints
        print("Trying generated viewpoints...")
        for i in tqdm(range(max_iterations)):
            camera = self.generate_viewpoint(
                random_offset=(i > 8), iteration=i, max_iterations=max_iterations
            )
            render_img = self.render_view(camera)

            # Save debug render
            if self.debug:
                debug_img = render_img.copy()
                cv2.putText(
                    debug_img,
                    f"View {i}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    debug_img,
                    f"View {i}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (1, 1, 1),
                    1,
                )
                self.debug_renders.append(debug_img)

            ellipse_info = self.detect_ellipse(render_img)
            if ellipse_info is not None:
                all_detections.append(
                    {
                        "circularity": ellipse_info["circularity"],
                        "camera": camera,
                        "ellipse": ellipse_info,
                        "render": render_img,
                    }
                )

                # Try to optimize this viewpoint
                current_camera = camera
                current_ellipse = ellipse_info
                current_render = render_img

                for j in range(max_optimizations):
                    new_camera = self.optimize_viewpoint(
                        current_ellipse, current_camera
                    )
                    new_render = self.render_view(new_camera)
                    new_ellipse = self.detect_ellipse(new_render)

                    if new_ellipse is None:
                        break

                    if new_ellipse["circularity"] > current_ellipse["circularity"]:
                        all_detections.append(
                            {
                                "circularity": new_ellipse["circularity"],
                                "camera": new_camera,
                                "ellipse": new_ellipse,
                                "render": new_render,
                            }
                        )
                        current_camera = new_camera
                        current_ellipse = new_ellipse
                        current_render = new_render
                    else:
                        break

        # Select the best detection
        if all_detections:
            best_detection = max(all_detections, key=lambda x: x["circularity"])
            self.best_circularity = best_detection["circularity"]
            self.best_camera = best_detection["camera"]
            self.best_ellipse = best_detection["ellipse"]
            self.best_render = best_detection["render"]

        # Save debug renders if enabled
        if self.debug and self.debug_renders:
            debug_dir = os.path.join(self.output_dir, "debug_views")
            os.makedirs(debug_dir, exist_ok=True)

            for i, render in enumerate(self.debug_renders):
                cv2.imwrite(
                    os.path.join(debug_dir, f"view_{i:03d}.png"),
                    cv2.cvtColor((render * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
            print(f"Debug views saved to {debug_dir}")

        # Save results
        if self.best_ellipse is not None:
            self.save_results()
            return {
                "found": True,
                "circularity": self.best_circularity,
                "center": self.best_ellipse["center"],
                "width": self.best_ellipse["width"],
                "height": self.best_ellipse["height"],
                "camera_position": -np.linalg.inv(self.best_camera.R)
                @ self.best_camera.T,
                "camera_direction": self.center.cpu().numpy()
                - (-np.linalg.inv(self.best_camera.R) @ self.best_camera.T),
            }
        else:
            return {"found": False}

    def save_results(self):
        """Save detection results"""
        if self.best_ellipse is None:
            return

        # Save the best render with ellipse visualization
        output_img = self.best_render.copy()

        # Convert to uint8 for drawing
        img_uint8 = (output_img * 255).astype(np.uint8)

        # Draw the detected ellipse with thicker line and brighter color
        cv2.ellipse(
            img_uint8,
            self.best_ellipse["ellipse"],
            (0, 255, 0),  # Green color
            3,  # Thicker line
        )

        # Add text showing the circularity
        cv2.putText(
            img_uint8,
            f"Circularity: {self.best_circularity:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),  # Green color
            2,
        )

        # Save visualization
        cv2.imwrite(
            os.path.join(self.output_dir, "best_view.png"),
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
        )

        # Get camera position and direction
        camera_position = -np.linalg.inv(self.best_camera.R) @ self.best_camera.T
        camera_direction = self.center.cpu().numpy() - camera_position

        # Save detection results
        results = {
            "circularity": float(self.best_circularity),
            "ellipse": {
                "center": self.best_ellipse["center"],
                "width": float(self.best_ellipse["width"]),
                "height": float(self.best_ellipse["height"]),
                "angle": float(self.best_ellipse["angle"]),
            },
            "camera": {
                "position": camera_position.tolist(),
                "direction": camera_direction.tolist(),
                "up": self.best_camera.R[:, 1].tolist(),
            },
            "object": {
                "centroid": self.center.cpu().numpy().tolist(),
                "bbox_dimensions": (
                    self.xyz.max(dim=0).values - self.xyz.min(dim=0).values
                )
                .cpu()
                .numpy()
                .tolist(),
            },
        }

        with open(os.path.join(self.output_dir, "hole_detection.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")
        print(f"Best circularity: {self.best_circularity:.3f}")
