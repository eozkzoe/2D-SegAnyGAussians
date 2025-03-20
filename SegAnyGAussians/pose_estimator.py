import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

class PoseEstimator:
    def __init__(self, gaussians, mask):
        """
        Initialize PoseEstimator
        Args:
            gaussians: GaussianModel instance
            mask: Boolean mask for selecting Gaussians
        """
        self.gaussians = gaussians
        
        # Handle different Gaussian representations (SAGA vs 2D-GS)
        if len(gaussians._xyz.shape) == 1:  # SAGA format (flattened)
            # Reshape mask if needed
            if len(mask.shape) == 1 and len(mask) == len(gaussians._xyz) // 3:
                # Expand mask to match flattened coordinates
                mask = mask.repeat_interleave(3)
            elif len(mask.shape) == 1 and len(mask) != len(gaussians._xyz):
                raise ValueError(f"Mask size {len(mask)} does not match flattened Gaussian size {len(gaussians._xyz)}")
        else:  # 2D-GS format (N, 3)
            if len(mask) != len(gaussians._xyz):
                raise ValueError(f"Mask size {len(mask)} does not match number of Gaussians {len(gaussians._xyz)}")
        
        # Convert mask to correct type and device
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.to(device=gaussians._xyz.device, dtype=torch.bool)
        
        self.mask = mask
        
    def estimate_pose(self):
        """Estimate pose (position and normal) of segmented Gaussians"""
        # Validate that we have points selected
        if not torch.any(self.mask):
            raise ValueError("No points selected in mask")
            
        # Get positions
        xyz = self.gaussians._xyz
        if len(xyz.shape) == 1:  # SAGA format
            # Reshape to (N, 3)
            xyz = xyz.reshape(-1, 3)
            mask_3d = self.mask.reshape(-1, 3)
            # Take only the first component of each point's mask (they're repeated)
            mask = mask_3d[:, 0]
            xyz = xyz[mask]
        else:  # 2D-GS format
            xyz = xyz[self.mask]
            
        points = xyz.detach().cpu().numpy()
        centroid = np.mean(points, axis=0)
        
        # Get normals directly from rotation matrices
        rotations = self.gaussians.get_rotation
        if len(rotations.shape) == 1:  # SAGA format
            rotations = rotations.reshape(-1, 3)
            mask_3d = self.mask.reshape(-1, 3)
            mask = mask_3d[:, 0]
            rotations = rotations[mask]
        else:  # 2D-GS format
            rotations = rotations[self.mask]
            
        normals = rotations.detach().cpu().numpy()
        
        # Get opacity weights for weighted calculations
        opacities = self.gaussians.get_opacity
        if len(opacities.shape) == 1 and len(xyz.shape) == 1:  # SAGA format
            opacities = opacities.reshape(-1, 1)
            mask_1d = self.mask.reshape(-1, 3)[:, 0]
            opacities = opacities[mask_1d]
        else:  # 2D-GS format
            opacities = opacities[self.mask]
            
        opacities = opacities.detach().cpu().numpy()
        weights = opacities / opacities.sum()
        
        # Calculate weighted centroid and normal
        weighted_centroid = np.sum(points * weights[:, None], axis=0)
        # Compute weighted normal directly
        weighted_normal = np.zeros(3)
        for i in range(len(normals)):
            weighted_normal += normals[i] * weights[i]
        weighted_normal = weighted_normal / np.linalg.norm(weighted_normal)
        
        # Also compute unweighted mean normal
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        
        # Calculate planarity using PCA for completeness
        pca = PCA(n_components=3)
        pca.fit(points - centroid)
        planarity = 1 - pca.explained_variance_ratio_[2]
        
        return {
            'centroid': weighted_centroid,
            'normal': weighted_normal,
            'unweighted_normal': mean_normal,
            'planarity': planarity,
            'points': points,
            'normals': normals
        }
        
    def get_oriented_bbox(self):
        """Get oriented bounding box of segmented Gaussians"""
        pose = self.estimate_pose()
        points = pose['points']
        
        # Use normal as primary axis
        normal = pose['normal']
        # Find perpendicular vectors to form coordinate system
        v1 = np.array([1, 0, 0]) if abs(normal[1]) > abs(normal[0]) else np.array([0, 1, 0])
        axis1 = np.cross(normal, v1)
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = np.cross(normal, axis1)
        
        # Create transformation matrix
        axes = np.vstack([axis1, axis2, normal])
        
        # Project points onto new coordinate system
        projected = (points - pose['centroid']) @ axes.T
        
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        dimensions = max_proj - min_proj
        print(f"center: {pose['centroid']}, 'axes': {axes}, 'dims': {dimensions}")
        
        return {
            'center': pose['centroid'],
            'axes': axes,
            'dimensions': dimensions
        }
        
    def get_mean_normal(self):
        """Get the mean normal direction of segmented Gaussians, weighted by opacity"""
        # Use estimate_pose to handle different formats consistently
        pose = self.estimate_pose()
        return pose['normal']
